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


def init_new_emb_from_codebook(model, tokenizer, original_vocab_size,
                               sid_index_path, codebook_path="", shuffle=False,
                               shuffle_seed=42):
    """用 RQ-KMeans 码本向量初始化新增 SID token 的 embedding（2026-09-14）。

    动机：resize_token_embeddings 给新 token 的是"老词表均值/协方差随机采样"，语义为空白。
    而 SFT 第 2 个任务（sid↔title 互译）的全部意义就是让模型自己把 SID 语义学出来——用码本
    初始化等于把这个先验直接给模型。对那 267 个落在"预训练空槽行"（std 0.0095，从未被训练过）
    上的 token 更是质变。

    实测依据（2026-09-14）：
      - 码本 (256,1024) × 3 级，与 LLM hidden 同为 1024
      - 两空间语义结构实质对齐：线性对齐留出 R²=0.40（打乱对照 -0.03）；
        level-0 簇命中率 47.3%（随机 0.39%，122×）；level-1 严格检验（固定 <a_x> 上下文）
        82.4%（随机 25%，3.3×）
      - 尺度差 71 倍（码本范数 65.9 vs 老词表 0.926）→ **必须重标定**，否则 logits 爆炸

    ⚠️ 重标定的目标尺度是 **resize 默认初始化给新 token 的范数 (≈0.30)**，不是老词表范数 (0.926)。
       2026-09-15 实测修正：对齐老词表会让首步 loss 从 15.6 涨到 24.1，对齐新 token 初始化
       范数则降到 13.2（比基线还低）。细节与机理见函数体内注释。

    映射关系（code ↔ token 字符串）不靠"猜 +1 偏移"，而是从 codes + index.json 直接推导
    并断言自洽 —— 少一个隐含假设。

    tie 红利：Qwen3 tie_word_embeddings=True 时 lm_head 与 embed_tokens 是同一块 tensor，
    写 emb.weight 会同时覆盖输入侧与输出侧，无需分别处理。

    ⚠️ 这会改变实验前提（本任务就是在"学 SID 语义"），故默认关闭，必须作为独立消融，
    不能与使用随机初始化的全参 SFT 直接比较。
    """
    if not codebook_path:
        # 命名约定：<dataset>.index.json ↔ <dataset>.codebooks_constrained.npz（同目录）
        codebook_path = sid_index_path.replace(".index.json", ".codebooks_constrained.npz")
    codes_path = sid_index_path.replace(".index.json", ".codes_constrained.npy")
    for p in (codebook_path, codes_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"codebook 初始化需要 {p}（可用 --codebook_path 指定）")

    cb = np.load(codebook_path)
    codes = np.load(codes_path)
    index = json.load(open(sid_index_path))
    n_levels = len([k for k in cb.files if k.startswith("codebook_")])

    # --- 从数据推导 {level: {code: token字符串}}，并断言自洽 ---
    level_map = [dict() for _ in range(n_levels)]
    for i in range(len(codes)):
        toks = index[str(i)]
        for lv in range(n_levels):
            code, tok = int(codes[i, lv]), toks[lv]
            prev = level_map[lv].setdefault(code, tok)
            if prev != tok:
                raise ValueError(
                    f"码本映射不自洽：level {lv} code {code} 同时对应 {prev!r} 与 {tok!r}"
                    f"（item {i}）——codes 与 index.json 可能不同源")

    emb = model.get_input_embeddings()

    # --- 尺度基准（2026-09-15 修正，实测驱动）---
    # ⚠️ 曾经的错误：把码本向量重标定到**老词表**行范数 (0.926)。那是模型**训练出来的自信**，
    #    不是新 token 该有的起点。实测后果（真实数据 16×136 batch，尺度扫描）：
    #      对齐老词表 0.926 → 新 token 区间 logits mean −2.26 / std 6.61 / max 16.6 → 首步 loss 24.13
    #      对齐新 token 0.303 → 同区间                −0.74 /     2.13 /     5.19 → 首步 loss 13.16
    #      （resize 基线：mean 3.55 / std 3.17 / max 10.50 → loss 15.64）
    #    机理：码本方向与 hidden 只是**部分**对齐（R²=0.40）。范数一大，未被对齐的那 60%
    #    就变成大幅随机 logits，把正确答案淹没 —— 尺度是放大器，方向对错都被它放大。
    #    现在改为对齐 **resize 默认初始化给新 token 的范数**：与基线同起点，唯一变量只剩"方向"
    #    ——这才是干净的 A/B。
    #    ⚠️ 必须在写入码本**之前**测：写完就再也测不到这个基准了。
    with torch.no_grad():
        new_init_norm = emb.weight.data[original_vocab_size:].float().norm(dim=1).mean().item()
    if not (new_init_norm > 0):     # resize 未产生新行（异常）时的兜底
        new_init_norm = 0.3
    old_norm = emb.weight.data[:original_vocab_size].float().norm(dim=1).mean().item()
    added = tokenizer.added_tokens_encoder
    written_ids, missing, scales = set(), [], []

    # --- shuffle=True：对照组（2026-09-15）---
    # 目的：分离"**码本的语义**"和"**方向的分布形状**"这两个解释。
    # 做法：**在同一级内**把 code→token 的对应关系错位重排 —— 向量集合、范数、层级结构全不变，
    #       只有"哪个语义向量给了哪个 token"被破坏。
    # 约束：强制 derangement（无不动点），否则有 token 会碰巧保留自己的正确向量，稀释对照强度。
    # 若打乱后效果不变 ⇒ 起作用的是"像真 embedding 的方向分布"；若明显变差 ⇒ 语义对齐真的起作用。
    perm_maps = None
    if shuffle:
        rng = np.random.RandomState(shuffle_seed)
        perm_maps = []
        for lv in range(n_levels):
            codes_lv = sorted(level_map[lv].keys())
            n = len(codes_lv)
            idx = np.arange(n)
            perm = rng.permutation(n)
            for _ in range(10000):               # 拒绝采样直到得到 derangement
                if not (perm == idx).any():
                    break
                perm = rng.permutation(n)
            else:
                raise RuntimeError(f"level {lv}: 无法生成 derangement（n={n}）")
            perm_maps.append({codes_lv[i]: codes_lv[perm[i]] for i in range(n)})
        n_fixed = sum(1 for lv in range(n_levels)
                      for k, v in perm_maps[lv].items() if k == v)
        print(f"[init] ⚠️ 对照组模式（codebook_shuffled, seed={shuffle_seed}）："
              f"码本向量在**同一级内**错位重排，破坏语义对应；"
              f"向量集合/范数/层级结构不变。derangement 校验：不动点 {n_fixed} 个（应为 0）",
              flush=True)

    with torch.no_grad():
        for lv in range(n_levels):
            C = torch.tensor(cb[f"codebook_{lv}"], dtype=torch.float32)
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12) * new_init_norm  # ← 尺度重标定
            scales.append(C.norm(dim=1).mean().item())
            for code, tok in level_map[lv].items():
                tid = added.get(tok)
                if tid is None:
                    missing.append(tok)
                    continue
                src = perm_maps[lv][code] if shuffle else code     # ← 对照组取别人的向量
                emb.weight.data[tid] = C[src].to(emb.weight.dtype)
                written_ids.add(tid)

    n_new = emb.weight.shape[0] - original_vocab_size

    # --- 无码本对应的新 token（主要是 15 个 <d_*> 消歧 token）：随机方向保留，但尺度对齐 ---
    # <d_*> 是前缀碰撞的消歧 token（约 27% 的商品 SID 需要它）。我们没有它们的语义先验，
    # 但至少不该引入**额外的**尺度偏置 —— 故拉到与其它新 token 相同的范数，方向仍是随机的。
    rest = [i for i in range(original_vocab_size, emb.weight.shape[0]) if i not in written_ids]
    with torch.no_grad():
        for i in rest:
            v = emb.weight.data[i].float()
            emb.weight.data[i] = (v / v.norm().clamp_min(1e-12) * new_init_norm).to(emb.weight.dtype)

    print(f"[init] 码本初始化：写入 {len(written_ids)} 行 / 新增 token {n_new} 个；"
          f"尺度重标定 {[round(s, 3) for s in scales]}"
          f"（目标=新 token 初始化范数 {new_init_norm:.4f}；老词表范数 {old_norm:.4f} 仅作参照）",
          flush=True)
    if missing:
        print(f"[init] 警告：{len(missing)} 个 token 不在 tokenizer.added_tokens 中，"
              f"样例 {missing[:5]}", flush=True)
    print(f"[init] 其余 {len(rest)} 个无码本对应（<d_*> 消歧 token）保持随机方向、尺度已对齐",
          flush=True)

    # --- 承重校验：所有新 token 行的范数都必须落在**初始化基准**附近 ---
    # 容差 ±50%：正常情况下码本行与 <d_*> 行都被显式设为 new_init_norm，偏差只来自 bf16 舍入；
    # 放宽到 1.5×/0.5× 是为了在少数 token 不在 added_tokens（保留 MVN 原值，见 missing 警告）时不误报。
    new_norm = emb.weight.data[original_vocab_size:].float().norm(dim=1)
    if new_norm.max() > 1.5 * new_init_norm or new_norm.min() < 0.5 * new_init_norm:
        raise ValueError(
            f"新 token 行范数异常：[{new_norm.min():.3f}, {new_norm.max():.3f}] vs 初始化基准 "
            f"{new_init_norm:.3f}（老词表 {old_norm:.3f}）——尺度未正确对齐，"
            f"会导致 logits 爆炸或部分 token 不可达")
    print(f"[init] 新 token 行范数：[{new_norm.min():.4f}, {new_norm.max():.4f}] "
          f"（目标 {new_init_norm:.4f}）✓", flush=True)
    return len(written_ids)


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
    # === LoRA 消融（2026-09-13）===
    # 冻结底座，只训「新增 SID token 的 embedding 行 + 低秩适配器」。
    # 与 freeze_LLM 的关系：freeze_LLM 是它的子集（只训新 token 行，0.8M），
    # LoRA 在此之上给 28 层的 7 个投影加低秩增量（~8.7M），让模型有能力重写计算逻辑。
    # 动机：全参 SFT 的 fp32 Adam 状态 4.8G/卡 是显存主压力之一；LoRA 把它压到 ~76MB。
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # 这其实对应了模型中的所有Linear层，除了lm_head
    # 是否把预训练词表的老行梯度清零（只训新增 SID token 行）。
    # 实测不省内存（Adam 按整块 tensor 分配，开不开都是 ~1.27G），是纯行为约束开关：
    #   默认 True  = 保护预训练词嵌入，只让 783 行新 token 移动
    #   设 False   = 全词表可更新，让 SFT 自由重塑整个嵌入空间（零成本消融点）
    lora_freeze_old_emb: bool = True,
    # === 新增 SID token 的 embedding 初始化策略（2026-09-14）===
    #   "resize"（默认）= transformers 默认：老词表均值/协方差随机采样，语义空白。
    #                     与全参 SFT of record 一致，保证 A/B 可比。
    #   "codebook"      = 用 RQ-KMeans 码本向量初始化，SID token 一开始就带语义。
    #     实测依据（2026-09-14）：码本与 LLM embedding 空间语义结构实质对齐——
    #       线性对齐留出 R²=0.40（打乱对照 -0.03）
    #       level-0 簇命中率 47.3%（随机 0.39%，122×）
    #       level-1 严格检验（固定 <a_x> 上下文）82.4%（随机 25%，3.3×）
    #     必须重标定尺度（码本范数 65.9 vs 老词表 0.926，71×），函数内已处理并断言。
    #     ⚠️ 这会改变实验前提（SFT 第 2 个任务本就是"学 SID 语义"）→ 必须作为独立消融，
    #        不可与使用随机初始化的全参 SFT 直接比较。
    init_new_emb: str = "resize",
    codebook_path: str = "",   # 空 = 由 sid_index_path 推导（同目录 *.codebooks_constrained.npz） 
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
    new_tokens = []  # 先绑定，避免 sid_index_path 为空时后续分支 NameError
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

            # === tie 校验（LoRA 方案的承重假设）===
            # Qwen3-0.6B tie_word_embeddings=True：resize 后 lm_head 应与 embed_tokens
            # 共享同一块 tensor。共享时"解冻新 token 行"会同时训到输入侧（embed）和
            # 输出侧（lm_head）——这是 tied 模型给的红利，我们才能只碰一个矩阵。
            # 若 tie 断了：输入侧在学、输出侧仍是随机初值，模型永远预测不出新 SID token，
            # loss 掉到平台不动，而根因藏在序列化层，极难 debug。故在此显式校验并修复。
            # （2026-09-13 澄清：曾据"ckpt_archive 的 RL ckpt 是 311 tensor 含独立
            #  lm_head.weight"推断 tie 会脱开——实测两块张量**逐位相同**，只是保存路径
            #  把同一块 tensor 写了两遍，白费 ~312MB/ckpt。tie 从未断过。）
            if getattr(model.config, "tie_word_embeddings", False):
                inp_w = model.get_input_embeddings().weight
                out_w = model.get_output_embeddings().weight
                if inp_w is out_w:
                    print(f"[tie] 校验通过：lm_head 与 embed_tokens 共享权重 {tuple(inp_w.shape)}，"
                          f"解冻新行将同时覆盖输入/输出两侧", flush=True)
                else:
                    print("[tie] 警告：tie_word_embeddings=True 但 lm_head 未与 embed_tokens 共享，"
                          "正在重新 tie", flush=True)
                    model.tie_weights()
                    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, \
                        "重新 tie 失败，LoRA 方案下新 token 输出侧将无法训练"
                    print("[tie] 修复完成：已恢复共享", flush=True)
            else:
                print("[tie] 注意：cfg.tie_word_embeddings=False，输入/输出两侧需分别解冻", flush=True)

            # === 新增 token 的 embedding 初始化策略（2026-09-14）===
            # 放在 resize + tie 校验之后：初始化的是"值"，与后面的 LoRA/冻结（管 requires_grad）
            # 正交，所以对全参和 LoRA 两条路径都生效。
            if init_new_emb in ("codebook", "codebook_shuffled"):
                # codebook_shuffled = 对照组：同一级内错位重排，破坏语义对应但保留向量分布
                init_new_emb_from_codebook(model, tokenizer, original_vocab_size,
                                           sid_index_path, codebook_path,
                                           shuffle=(init_new_emb == "codebook_shuffled"))
            elif init_new_emb != "resize":
                raise ValueError(
                    f"未知 init_new_emb={init_new_emb!r}，可选：resize | codebook | codebook_shuffled")
            else:
                print("[init] 新 token 使用 resize 默认初始化（老词表均值/协方差随机采样）——"
                      "与全参 SFT of record 保持一致", flush=True)
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

    # === LoRA 消融：冻结底座，只训「新 token embedding 行 + 低秩适配器」 ===
    # 与 freeze_LLM 的区别：freeze_LLM 只训 embedding 新行（801,792 个），模型没有任何能力
    # 调整计算逻辑，只能靠"随机初始化的符号向量"硬扛；LoRA 在此之上给 28 层的 7 个投影
    # 加低秩增量（10,092,544 个），让注意力/FFN 的映射能一起被改写。
    #
    # 参数量（Qwen3-0.6B + 783 SID token，均为 2026-09-13 实测，非估算）：
    #   底座 596,578,304 | LoRA 10,092,544 | embedding 整块 156,110,848
    #   requires_grad=True 的合计 166,203,392 = 27.40% —— 但这个数字有欺骗性：
    #   真正会移动的只有 LoRA 的 10.09M + embedding 的新 783 行 0.80M ≈ 10.89M（1.8%），
    #   因为 requires_grad 是 tensor 级的，掩码只能让老行梯度为零、无法只开 783 行。
    #
    # 显存：fp32 Adam 状态 全参 4.8GB → 本方案 1.27GB（LoRA 77MB + embedding 整块 1.25GB）。
    #   注意 embedding 那 1.25GB 是"按整块分配"的必然开销，开不开掩码都一样（见下方掩码开关）。
    if use_lora:
        assert not freeze_LLM, "--use_lora 与 --freeze_LLM 互斥（前者已包含后者语义）"
        from peft import LoraConfig, get_peft_model

        target_modules = [t.strip() for t in lora_targets.split(",") if t.strip()]

        # --- 第 1 步：先套 LoRA（它自己会冻结所有非 adapter 参数）---
        # ⚠️ 顺序踩过坑（2026-09-13 冒烟实测）：PEFT 的 LoraModel 在构造时会调用
        # _mark_only_adapters_as_trainable()，把所有名字里不含 "lora_" 的参数**一律**
        # 设为 requires_grad=False。因此"先打开 embedding 再 get_peft_model"会被它关掉，
        # 表现为：训练照跑、loss 照降，但 783 个 SID token 的 embedding 全程冻结在
        # resize 时的随机初值上——正是本项目最不能接受的失败模式（SID 符号永远是随机码）。
        # 正确顺序只能是：get_peft_model 之后，再把 embedding 打开。
        model = get_peft_model(model, LoraConfig(
            r=lora_r,                    # 秩：扫描时须同步扫 alpha，保持 alpha/r 不变
            lora_alpha=lora_alpha,       # 增益分子；实际缩放 = alpha/r = 32/16 = 2.0
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ))

        # --- 第 2 步：打开新增 SID token 的 embedding 行 ---
        embedding_layer = model.get_input_embeddings()
        assert embedding_layer.weight.shape[0] > original_vocab_size, (
            f"use_lora 需要新 token：embedding 现有 {embedding_layer.weight.shape[0]} 行，"
            f"未超过原词表 {original_vocab_size}；请检查 --sid_index_path")
        embedding_layer.weight.requires_grad = True  # 整块打开（requires_grad 是 tensor 级的）

        # --- 掩码开关：是否保护预训练词表的老行 ---
        # 关键事实（2026-09-13 实测）：**掩码不省内存**。Adam 在"建状态"那一刻看到的是
        # "这块 tensor 需要梯度"，于是给整块 [vocab, 1024] 分配 fp32 m/v（~1.27G）——
        # 开不开掩码都是这个数。所以掩码不是省显存的开关，而是**行为约束**开关：
        #   开（默认）：前 original_vocab_size 行梯度清零 → 预训练词嵌入不被扰动
        #   关        ：全部行可更新 → 让 SFT 自由重塑整个词嵌入空间
        # 两者内存/速度完全一致 ⇒ 这是一个零成本的消融点，回答"该不该保护预训练词嵌入"。
        if lora_freeze_old_emb:
            def mask_grad(grad):
                grad[:original_vocab_size].zero_()
                return grad

            embedding_layer.weight.register_hook(mask_grad)
            print(f"[lora] 掩码 ON：前 {original_vocab_size} 行（预训练词表）梯度清零，"
                  f"仅索引 {original_vocab_size}~{embedding_layer.weight.shape[0]-1} 的 "
                  f"{embedding_layer.weight.shape[0]-original_vocab_size} 行更新", flush=True)
        else:
            print(f"[lora] 掩码 OFF：全部 {embedding_layer.weight.shape[0]} 行（含预训练词表）"
                  f"均可更新——Adam 状态大小与 ON 时相同（都是整块分配）", flush=True)

        # --- 第 3 步：把上面的教训固化成运行时守卫 ---
        # 不靠"我记得顺序对了"，而是直接断言 embedding 真的可训。
        assert embedding_layer.weight.requires_grad, (
            "embedding 未被打开（被 get_peft_model 或其后的逻辑关掉了）——SID 新 token 将无法学习")
        n_new_rows = embedding_layer.weight.shape[0] - original_vocab_size
        # 不硬编码 hidden_size：换底座模型（如 Qwen3-1.7B 的 2048）时这行必须自动跟随
        n_new_elems = n_new_rows * embedding_layer.weight.shape[1]
        emb_trainable = embedding_layer.weight.numel()
        lora_trainable = sum(p.numel() for n, p in model.named_parameters()
                             if p.requires_grad and "lora_" in n)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable == lora_trainable + emb_trainable, (
            f"可训练参数构成异常：总 {total_trainable:,} ≠ LoRA {lora_trainable:,} + "
            f"embedding {emb_trainable:,}——有非预期的参数被打开/关闭")

        model.print_trainable_parameters()
        print(f"[lora] r={lora_r} alpha={lora_alpha} dropout={lora_dropout} "
              f"freeze_old_emb={lora_freeze_old_emb} targets={target_modules}", flush=True)
        print(f"[lora] 可训练构成（实测，非推算）：LoRA {lora_trainable:,} + "
              f"embedding整块 {emb_trainable:,}（其中新 token {n_new_rows} 行 × "
              f"{embedding_layer.weight.shape[1]} 维 = {n_new_elems:,}）"
              f" = {total_trainable:,}", flush=True)

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

    # 超参哨兵（2026-09-13 扩充）：把**实际生效**的超参打出来。
    # 动机：sft_lora.sh 用 ${VAR:-默认} 让环境变量可覆盖，代价是变量名拼错/漏 export 时
    #   会静默退回默认值（如 `bash sft_lora.sh LR=2e-4` 里的 LR 被当成了 $1，脚本读不到，
    #   bash 不报错）。启动日志是唯一能当场发现这件事的地方。
    # learning_rate 此前完全没打——它只出现在每步训练日志里，且第 1 步恒为 0（warmup 造成），
    #   无法用来核对，等于最该核对的超参反而最不可见。
    # 用 trainer.args.* 而非函数参数：报的是 Trainer 真正拿到的值（经 TrainingArguments 处理后）。
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        # lr_scheduler_type 是 SchedulerType 枚举，直接插值会打成 "SchedulerType.LINEAR"；
        # 取 .value 打成命令行里写的那种形式（getattr 兜底：万一将来变回字符串也不炸）
        _sched = getattr(trainer.args.lr_scheduler_type, "value", trainer.args.lr_scheduler_type)
        print(f"[cfg] lora={use_lora} lr={trainer.args.learning_rate:g} sched={_sched} "
              f"warmup={trainer.args.warmup_steps} epochs={trainer.args.num_train_epochs} "
              f"cutoff={cutoff_len} seed={trainer.args.seed}", flush=True)

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

    # run 目录里先落一份原生产物：LoRA 下这里是 adapter（几 MB，可留作"关掉 adapter 即 ref model"
    # 的复用素材），全参下是完整模型——两条路径都保留"训练产物可回溯"的性质。
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    if use_lora:
        # === merge 回完整模型：下游契约要求 final_checkpoint 是自包含的 HF 模型目录 ===
        # rl.py:160 / evaluate.py:59 都是裸的 AutoModelForCausalLM.from_pretrained(model_path)，
        # 只认 model.safetensors；存 adapter 的话它们拿到的是 adapter_model.safetensors，
        # 整个 RL/评测流水线会断在加载这一步。
        # merge_and_unload() 把 LoRA 增量 (alpha/r)*B@A 并进 base 权重，返回普通 Qwen3ForCausalLM。
        # 我们解冻的 embedding 新行不在 LoRA 里（是 base 参数），merge 不碰它们，天然保留。
        merged_model = trainer.model.merge_and_unload()
        # ⚠️ 必须是 True（2026-09-17 修正）。
        # 训练侧在 643 行把 model.config.use_cache 设成 False 是对的（梯度检查点/省显存），
        # 但 merge 出来的 merged_model 继承了这个 False，会被**写进部署产物的 config.json**——
        # 于是它声明"这个模型不想要 KV cache"。实测 evaluate.py 侥幸没受影响，是因为它显式传了
        # use_model_defaults=False（本意是管采样参数，顺手把 use_cache 也保住了）；见
        # generation/utils.py:1760 的分支：走 else 就完全不碰 use_cache，保持 GenerationConfig 默认的 True。
        # 但这是个"看着像 bug 却不是 bug"的字段，坑后来人。部署产物是推理用的，显式改回 True。
        merged_model.config.use_cache = True
        # 把"我以为"变成"我验证了"：merge 后 embedding 行数必须仍等于 tokenizer 长度，
        # 否则下游的 SID token id 会整体错位（embedding 行数与 tokenizer 对不上）。
        # 注释里刻意不写具体数字（如 152452）——换 SID 变体/底座时它会过期，而断言本身是动态的。
        assert merged_model.config.vocab_size == len(tokenizer), (
            f"merge 后词表不一致：config={merged_model.config.vocab_size} vs tokenizer={len(tokenizer)}")
        n_new_rows = merged_model.get_input_embeddings().weight.shape[0] - original_vocab_size
        print(f"[lora] 已 merge 为完整模型：vocab_size={merged_model.config.vocab_size} "
              f"dtype={merged_model.dtype} 新 token 行={n_new_rows}", flush=True)
        merged_model.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
