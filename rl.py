from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from data import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import re
from minionerec_trainer import ReReTrainer, MemTrackerCallback, ArchiveCheckpointCallback
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score

os.environ['WANDB_MODE'] = 'disabled'
# zero2 配套补丁（engine ckpt 跳过 / 末端 best HF 直载），与 sft.py 共用；仅 ds 启用时生效，非 ds 原样
from ds_zero2_patches import apply_ds_zero2_patches, make_run_dir_name
apply_ds_zero2_patches()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    resume_from_checkpoint: str = "",  # 续训：从 <run目录>/checkpoint-N 恢复（欠费断电恢复用），沿用原 run 目录
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    save_steps: float = 0.1,   # checkpoint 保存频率（分数 = 占总步数比例，compute_steps 时 ceil 成整数步）
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,  # grpo损失函数中 KL 项系数
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    sample: int = -1,   # 每个训练数据集抽样条数（-1 = 全量）。快速验证设小值可大幅缩短每 epoch 步数（如 2048）
    eval_sample: int = -1,  # 评估集抽样条数（-1 = 全量）。eval 每轮也做完整 rollout，冒烟测试建议设小（如 256）
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",   # {id:['<a_1>', '<b_2>', '<c_3>'], ...}
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    # !!!
    token_norm: str = "group",     # token 级 advantage 归一化："group"=组内（原实现）| "column"=跨组列（RL_IDEAS 想法 c）
    all_wrong_penalty: float = 0.0,  # 全错列附加惩罚 λ（想法 b，0=关；如 1.0）
    archive_steps: str = "",       # 逗号分隔的步数（如 "750,1500"）：该步 ckpt 保存后立即归档移出 run 目录
    archive_dir: str = "ckpt_archive",  # 归档根目录（<archive_dir>/<run名>/checkpoint-N）
    deepspeed_config: str = "",    # transformers 格式 ds json（config/ds_zero2.json）；空 = 不启用（单卡/未装 deepspeed 环境）

    reward_weights: list = [1.0, 1.0, 1.0],
):
    # 禁用 PyTorch 的两种加速注意力后端
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)

    # 为每次运行生成独立输出目录（run_时间戳），避免不同 run 的 checkpoint 互相覆盖（与 sft.py 一致）
    if resume_from_checkpoint:
        # 续训：沿用原 run 目录（resume 路径 = <run目录>/checkpoint-N），不新建时间戳目录，保证 ckpt 轮换与 global_step 连续
        assert os.path.isdir(resume_from_checkpoint), f"resume checkpoint not found: {resume_from_checkpoint}"
        output_dir = os.path.dirname(resume_from_checkpoint)
    else:
        output_dir = os.path.join(output_dir, make_run_dir_name())  # rank0 广播时间戳，防跨秒竞态（2026-09-08）

    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)

    # 从 info 文件中提取 SID 字符串 '<a_1><b_2><c_3>'
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}  # SID → 编号

    # === 构建数据集 ===
    # 注意：sample 对每个子任务分别生效（ConcatDataset 总量 ≈ 3 × sample，train_data3 原先写死 10000）
    train_datasets = []
    # 任务1: 历史 sid seq -> target sid
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    # 任务2: title -> sid， description -> sid
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    # 任务3: 历史 title seq -> target sid
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data3)
    # === per-task 双 trie（2026-09-03，见 docs/RL_IDEAS.md）===
    # 对齐任务（title/desc→sid）target 只有前 3 级语义前缀（RLTitle2SidDataset 丢弃无语义的 extra <d_x>）→
    # 其 prompt 解码用"前缀 trie"（3 级即停、<d_x> 不可达）；NTP 任务（target 全长含 <d_x>）用全长 trie。
    # 修复：全长 trie 下碰撞前缀被强制续 <d_x>、EOS 被屏蔽 → 对齐样本 3 级答案永远不在解码支持集，
    # 且 first-diff 把被迫生成的 <d_x> 位误判 -1（支持集铁律，详见 RL_IDEAS 讨论）。
    trie_prefix_prompts = set(train_data2.prompt2history.keys())
    train_data = ConcatDataset(train_datasets)
    # eval：也走完整 rollout+reward，冒烟测试用 eval_sample 单独收紧
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=(eval_sample if eval_sample > 0 else 10000))

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)

    # === 收集所有样本的 prompt2history 和 history2target ===
    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    # === 加载模型 ===
    llm_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    # === 构造 ndcg 式的排序奖励 ===
    # a_i = 1/log2(i+2) 是第 i 位的 NDCG 权重；-a_i 表示"如果第 i 位不是 target，这个位置造成的损失
    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]  # -DCG  
    # 归一化
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]  # 除的是 sum(a) = 所有位置的权重之和

    # === 四种奖励函数 ===

    def ndcg_rule_reward(prompts, completions):
        """
        NDCG 式排名奖励：给每条生成按 '标准答案排在哪一位' 打分
        
        prompts:      [p0, p0, ..., p0, p1, p1, ..., p1, ...]   # 每个 prompt 连续重复 num_generations 次

        completions:  [g0, g1, ..., g15, g16, ...]               # 一一对应，每条是一个 rollout 的生成文本

        GRPO 的约定：同一 prompt 的 num_generations=16 条生成连续排在一起，所以 completions 长度 = 16 × prompt 数。返回等长的奖励列表。

        ``注意``：这里的优势计算为结果监督 per-completion，非 per-token.
        """

        # 反查标准答案 （由于 prompt 重复 16 次，targets 里每个 target 也重复 16 次）
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        # 逐条判定命中 + 按位惩罚
        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):  # 如果该条正确
                flag = True     # 本组至少一条命中
                lis.append(0.0) # 命中的：惩罚为 0
            else:
                lis.append(ndcg_rewards[i%num_generations])  # 未命中的：负惩罚，位置越靠前越重

            # 每满 num_generations 个结算一组
            if (i+1)%num_generations == 0:  
                if flag:
                    rewards.extend(lis)  # 组内至少一条命中 → 用真实惩罚
                else:
                    rewards.extend([0.0] * repeat)  # 16 条全没命中 → 整组全 0
                # 复位
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        """
        硬二值奖励
        
        问题：稀疏，对排序质量的指导作用有限
        """
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):
            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        # 预测的 sid 与真 target sid 的语义向量的相似度
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))  # 生成无效 → 随机塞一个 id
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his)) # 记录真实历史长度
            if len(his) < len_seq:   # len_seq=10，超长截断（CSVBaseDataset 已处理）、不足补位
                his = his + [item_num] * (len_seq - len(his))  # padding 用 item_num（不在 0..item_num-1 里）
            history_ids.append(his)
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    

        # 使用 sasrec 打分
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))  # [batch, item_num] 每序列对每个物品的预测分
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)  # 只取"模型生成的那个物品"的分数
        return scores


    # === first-diff token 级监督奖励 ===
    # 动机：RQ-KMeans 层级残差结构下，只有与 target 残差路径一致的"前缀"才有语义。
    # 对每条生成，逐 SID 位与 target 对齐，只监督【首个分歧位】及之前的正确前缀：
    #   - 正确前缀位：+1（强化正确路由）
    #   - 首个分歧位：-1（负信号）
    #   - 分歧后的后缀位：0（屏蔽——错误路径上的偶然对齐无意义，不给信号）
    #   - 生成提前结束（3级 vs 4级 target）：EOS 停止位 = 分歧位 -1
    #   - 整串命中（含正确停止）：全位 +1
    # 返回 list of dict {"scores": [...], "mask": [...]}，长度 = 生成 SID 数 + 1（末位 = EOS/停止位）。
    # 标记 token_level=True：trainer 走 token 级 advantage 路径（per-token 组内归一化），
    # 可与 per-completion 标量奖励（rule/ndcg）共存（trainer 内合并）。
    def first_diff_reward(prompts, completions):
        sid_pat = re.compile(r"<[abcd]_\d+>")
        results = []
        for i, completion in enumerate(completions):
            target = history2target[prompt2history[prompts[i]]].strip("\n\" ")
            gen = completion.strip("\n\" ")
            g = sid_pat.findall(gen)   # 生成的所有 SID token（按出现顺序）
            t = sid_pat.findall(target)
            L = len(g)
            T = len(t)
            # 逐位找首个分歧位置
            k = 0
            while k < L and k < T and g[k] == t[k]:
                k += 1
            scores = [0.0] * (L + 1)  # 末位 = EOS/停止位
            mask = [0] * (L + 1)  # mask 标记哪些位有监督(1)
            if k == L and k == T:      # 整串命中（含正确停止）
                scores = [1.0] * (L + 1)
                mask = [1] * (L + 1)
            elif k == L:               # 生成是 target 严格前缀：分歧发生在停止决策（EOS 位 -1）
                scores = [1.0] * L + [-1.0]
                mask = [1] * (L + 1)
            else:                      # 分歧在生成第 k 位（k < L）：前缀 +1，分歧位 -1，后缀屏蔽
                scores = [1.0] * k + [-1.0] + [0.0] * (L - k)  # 错误前缀下的后续token不惩罚
                mask = [1] * (k + 1) + [0] * (L - k)   # EOS 位无监督
            results.append({"scores": scores, "mask": mask})
        return results

    # 函数对象也可以挂属性！
    first_diff_reward.token_level = True  # token 级奖励标记（trainer 据此分流）

    if reward_type == "rule":
        reward_fun = rule_reward
        reward_w = reward_weights[:1]
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
        reward_w = reward_weights[:2]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
        reward_w = reward_weights[:1]
    elif reward_type == "semantic":
        reward_fun = semantic_reward
        reward_w = reward_weights[:1]
    elif reward_type == "sasrec":
        reward_fun = cf_reward
        reward_w = reward_weights[:1]
    elif reward_type == "first_diff":
        reward_fun = first_diff_reward
        reward_w = reward_weights[:1]
    elif reward_type == "ranking_firstdiff":   # rule(0/1) + ndcg(排序惩罚) + first-diff(token级) 共存
        reward_fun = [rule_reward, ndcg_rule_reward, first_diff_reward]
        reward_w = reward_weights[:3]
    
    # wandb：传了 --wandb_run_name 才启用（online，默认；可用 WANDB_MODE=offline 覆盖 → 本地记录后 wandb sync）；
    # 不传 = 保持历史行为（disabled，无 key 也不崩）
    if wandb_run_name:
        os.environ['WANDB_PROJECT'] = wandb_project or "MiniOneRec"
        os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "online")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    training_args = GRPOConfig(output_dir=output_dir,
                                deepspeed=(deepspeed_config or None),  # zero2（config/ds_zero2.json）；None = 关闭
                                save_steps=save_steps,  # 每跑完总步数的 10% 保存一次 checkpoint
                                save_total_limit=3,  # 每 checkpoint ~7GB（optimizer.pt fp32），20 个会把 50GB 数据盘写满（2026-09-02 事故）
                                max_completion_length=128,  # 每个 completion 生成的新 token 的最大数量
                                num_generations=num_generations,  # 每个 prompt 生成 num_generations 条候选 
                                temperature=temperature,  # 采样温度
                                sync_ref_model=sync_ref_model,  # 是否每个优化器步把 policy 权重同步到 ref 模型；相当于与上个优化器步的policy滚动对比
                                per_device_eval_batch_size=eval_batch_size,                 # 每卡每步处理的 prompt 条数
                                per_device_train_batch_size=train_batch_size,               # 同上，eval 时（eval 也做完整 rollout + reward）
                                gradient_accumulation_steps=gradient_accumulation_steps,    # 梯度累积步数，梯度累积多少步更新一次权重，等效 batch_size = gradient_accumulation_steps x per_device_eval_batch_size x 卡数
                                eval_steps=eval_step,  # 每总步数的 eval_step 做一次 eval
                                eval_strategy="steps", # eval 按步数触发而不是 epoch
                                logging_steps=1, 
                                learning_rate=learning_rate,  # 这里是 1e-5，sft中是 3e-4
                                beta=beta,  # GRPO 损失里的 KL 惩罚系数
                                warmup_ratio=0.03,  # 前 3% 步数线性升 lr
                                max_grad_norm= 0.3,  # 梯度裁剪
                                num_train_epochs=num_train_epochs, 
                                bf16=True,
                                optim="adamw_torch",  # 原为 paged_adamw_32bit（需 bitsandbytes，未安装）；4卡zero2分片下显存充裕
                                lr_scheduler_type="cosine",  # 学习率调度，余弦退火
                                save_strategy="steps",
                                report_to="wandb",
                                run_name=wandb_run_name,
                                reward_weights=reward_w,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        token_norm=token_norm,
        all_wrong_penalty=all_wrong_penalty,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,  # 是否动态采样
        beam_search=beam_search,  # 是否使用 beam search
        test_beam=test_beam,      # 束搜索宽度
        test_during_training=test_during_training,
        info_file=info_file,
        trie_prefix_prompts=trie_prefix_prompts, # 对齐任务（RLTitle2SidDataset）产生的全部 prompt 字符串的集合，用于区分两套-trie约束的启用
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    # ds/batch 核对（rank0 打一行；zero2 只分片优化器，batch 语义须与预期一致）
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[cfg] ds={'on: ' + str(trainer.args.deepspeed) if trainer.args.deepspeed else 'off'} "
              f"per_device_prompts={trainer.args.per_device_train_batch_size} "
              f"gas={trainer.args.gradient_accumulation_steps} gen={num_generations} "
              f"world={trainer.accelerator.num_processes} "
              f"| prompts/步={trainer.args.per_device_train_batch_size * trainer.accelerator.num_processes}",
              flush=True)

    # OOM 调查（2026-09-02）：每 50 步打点显存并 empty_cache——allocated 涨=真泄漏，仅 reserved 涨=池/碎片
    # zero2 首跑建议保留以验证 OOM 是否解决（预期 allocated 从 ~11.2G 降至 ~4-5G）；稳定后可删
    trainer.add_callback(MemTrackerCallback(log_every=50))

    # 归档回调：把指定步数的 ckpt 刚保存完即整体移出 run 目录（防 save_total_limit 轮换删掉对照锚点）
    archive_steps = [s.strip() for s in str(archive_steps).split(",") if s.strip()]
    if archive_steps:
        trainer.add_callback(ArchiveCheckpointCallback(steps=archive_steps, archive_dir=archive_dir))

    trainer.train(resume_from_checkpoint=(resume_from_checkpoint or None))

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
