import os
import numpy as np 
import fire
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import json
import time
# import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random

from torch.utils.data import ConcatDataset

# zero2 只分片优化器状态，参数仍然每卡全量


class PeriodicEmptyCache(TrainerCallback):
    """SFT 显存保险：每 every 步 empty_cache 释放池子空闲块 + 打点（5090×2 micro24 稳态 ~30.4G/32.6G，
    余量仅 ~2G；eval/save 瞬态是主要风险）。empty_cache 只回收池内空闲，不降低活集本身。"""

    def __init__(self, every: int = 20):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0 or not torch.cuda.is_available():
            return
        gb = 2**30
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[mem][step {state.global_step}] allocated="
                  f"{torch.cuda.memory_allocated()/gb:.2f}GiB reserved="
                  f"{torch.cuda.memory_reserved()/gb:.2f}GiB", flush=True)
        torch.cuda.empty_cache()

# === 监控修复（2026-08-30 19:5x，详见 MONITORING_LOG.md）===
# 环境 torch=2.3.0 < 2.6，transformers 4.57.3 的 CVE-2025-32434 安全检查
# (check_torch_load_is_safe) 会阻止 resume 时 torch.load 本地 checkpoint(optimizer.pt)。
# 加载的是本机自生成的可信文件，故绕过该检查（不影响 safetensors 路径）。
import transformers.utils.import_utils as _transformers_iu
import transformers.trainer as _transformers_trainer
_transformers_iu.check_torch_load_is_safe = lambda: None
_transformers_trainer.check_torch_load_is_safe = lambda: None

# 监控修复续：resume 时跳过 optimizer/scheduler 加载（只恢复模型权重）。
# 原因：DDP 下 Trainer 把 optimizer.pt（fp32 m+v ≈2.4GB/卡）加载到 GPU，
# 24GB 显存被挤爆导致第一个训练步 OOM（详见 MONITORING_LOG.md）。已训练进度仅 5%，动量丢失影响很小。
def _skip_optimizer_scheduler_load(self, checkpoint):
    print("[MONITORING] 跳过 optimizer/scheduler 加载（显存限制），仅恢复模型权重")
_transformers_trainer.Trainer._load_optimizer_and_scheduler = _skip_optimizer_scheduler_load

# === zero2 配套补丁（ds engine ckpt 跳过 / 末端 best HF 直载）——与 rl.py 共用 ds_zero2_patches.py ===
# 背景：engine ckpt 9.2G/次写满单卷磁盘；load_best_model_at_end 的 ds 分支要 engine 文件 → 早停后崩。
# 详见 ds_zero2_patches.py 头注释与 PROGRESS 2026-09-07 条目。
from ds_zero2_patches import apply_ds_zero2_patches, make_run_dir_name
apply_ds_zero2_patches()


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            # {"0": ["<a_5>", "<b_23>", "<c_55>"], "1": ["<a_0>", "<b_7>", "<c_12>"], ... }
            self.indices = json.load(f)  

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():  # 遍历每个item的sid（token列表）
            for token in index: # 遍历该sid中的每个token
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,  # 最大输入 token 数，尾部截断
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb 参数
    wandb_project: str = "MiniOneRec", 
    wandb_run_name: str = "",  # 设置后启用 wandb（需先 wandb login）；空 = 只写 tensorboard，不发 wandb
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    deepspeed_config: str = "",  # transformers 格式 ds json（config/ds_zero2.json）；空 = 不启用（单卡/未装 deepspeed 环境）
    train_tasks: str = "all",    # "all"=三任务混合（NTP+sid-title 互译+历史sid→title，默认）；"ntp"=仅 next-item 预测（消融）
):
    set_seed(seed)
    if wandb_run_name:
        os.environ['WANDB_PROJECT'] = wandb_project or "MiniOneRec"
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (base_model), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # === 为每次运行生成独立输出目录（run_时间戳），避免不同 run 的 checkpoint 互相覆盖 ===
    # 例如 --output_dir ./outputs/ → ./outputs/run_20260830_215501/
    output_dir = os.path.join(output_dir, make_run_dir_name())  # rank0 广播时间戳，防跨秒竞态（2026-09-08）

    # === 根据 micro batch size 计算梯度累积步数 ===
    # 一个 batch 太大放不进显存，于是每次前向只跑 micro_batch 条，然后累积梯度后再更新参数
    # 效果等价完整 batch。micro-batch 梯度之和 = 32 × 完整 batch 梯度。qwen模型中也不涉及 batch norm。
    gradient_accumulation_steps = batch_size // micro_batch_size

    # === 多卡管理（但是貌似是死码，在Trainer中会再处理） ===
    device_map = "auto" 
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 进程数
    ddp = world_size != 1  # 如果是多卡训练（DistributedDataParallel）
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # ""表示整个模型，LoCAL_RANK是本进程GPU编号
        gradient_accumulation_steps = gradient_accumulation_steps // world_size  # 如果多卡，micro batch 就可以再分

    # === 加载 plm_model ===
    if not train_from_scratch:  #  从磁盘加载架构和预训练权重
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
        )
    else:  # 只加载架构，不加载权重，从零开始训练  
        config = AutoConfig.from_pretrained(base_model)   # 只读 config.json，不碰 safetensors
        model = AutoModelForCausalLM.from_config(config)  # 按架构"现搭"一个模型
        print("Training from scratch!")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    original_vocab_size = len(tokenizer)

    # === 扩容词表，将 SID 中的 token 注册到 tokenizer，相应扩展模型嵌入层大小 ===
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()  # 所有新增 token
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            # 新 token 注册进 tokenizer （只是写入内存，未持久化到磁盘）
            tokenizer.add_tokens(new_tokens)  # 注册进 tokenizer
            # 模型 emebedding 矩阵从原来的 vocab 扩展到新词表大小，基于旧词表均值、协方差进行正态分布随机初始化
            model.resize_token_embeddings(len(tokenizer)) 
        print("EXTEND VOLAB FINISHED")

    # === 如果只训练新增token的embedding ===
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # === 多个数据集对应多任务指令微调 —— LLM for rec 中的标配做法  ===
    # 3个训练目标，共享同一个模型、同一套 SID token embedding，只是输入形式、输出形式、监督信号来源不同
    # 动机：
    # 1. SID token 是新增的，初始 embedding 是语义空白的；如果只用 NTP 训练，模型只能学到"符号→符号"的共现转移，但完全不知道这些符号指什么物品。通过 sid-title 互译任务来让模型理解符号的内容
    # 2. 监督信号互补（协同vs内容）：NTP的信号来自交互数据（稀疏），冷门item的sid完全没有监督；sid-title 互译是基于全量item的metadata，冷门 item 也能被理解；历史 sid-> target title 是进一步联系起来前两者。
    # 3. 共享 embedding 空间的多任务塑造

    train_datasets = []
    # 1. next-item 预测: 历史 sid 序列 -> target sid
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    # 2. sid-title互译（对齐）: sid -> title, title -> sid
    # 3. 历史 sid 序列 -> target title
    # 消融（--train_tasks ntp）：只保留 1（next-item 预测），隔离其余两个 metadata 语义任务的贡献
    if train_tasks == "all":
        train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
        train_datasets.append(train_data2)
        train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
        train_datasets.append(train_data3)
    
    # train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data4)
    # train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    # train_datasets.append(train_data5)

    # 训练集混合三个任务
    train_data = ConcatDataset(train_datasets)  # 这里是均匀采样，数据集比例 5:1:5，目前实现没有显式加权
    # 验证集只评估主任务 NTP
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)

    print("LOAD DATA FINISHED")    

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    # train_data[0]返回一个样本，形如：
    # {'input_ids':      [样本1的ids, 样本2的ids, ..., 样本282457的ids],
    # 'attention_mask': [样本1的mask, 样本2的mask, ...],
    # 'labels':         [样本1的labels, 样本2的labels, ...]}


    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        tokenizer=tokenizer,  # save_model 时把 tokenizer 一起写进模型目录（RL/eval 从该目录加载；历史 final_checkpoint 自包含即因此）
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            deepspeed=(deepspeed_config or None),  # zero2（config/ds_zero2.json）：优化器状态 4 卡分片；None = 关闭
            run_name=wandb_run_name or None,  # 传给 wandb/tensorboard 的 run 名（wandb 面板按此区分实验）
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=4,  # 早停 patience=3 ⇒ best 之后至多 3 次保存；limit 4 保证 best 目录不被轮换删（×1.5G≈6G）
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=(["wandb", "tensorboard"] if wandb_run_name else ["tensorboard"]),  # 传 --wandb_run_name 才启用 wandb；勿用 None（HF 里 None 回落到 "all"）
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3),
                     PeriodicEmptyCache(every=100)],  #
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False  # KV cache 推理时才用

    # batch 语义核对（zero2 只分片优化器状态，batch 语义必须与预期一致；rank0 打一行）：
    # sft.py 约定 --batch_size = 全局 batch：gas = batch//micro 再 //world 分到每卡
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[cfg] per_device={trainer.args.per_device_train_batch_size} "
              f"grad_accum={trainer.args.gradient_accumulation_steps} world={world_size} "
              f"ds={deepspeed_config or 'off'} | 全局 batch = "
              f"{trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps * world_size}",
              flush=True)

    print('开始训练')
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
