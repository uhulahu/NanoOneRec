# MiniOneRec（本项目改造版）

基于 **MiniOneRec** 的生成式推荐实践仓库：语义 ID（SID）+ LLM（Qwen3-0.6B）微调 + 面向推荐的自研 RL（GRPO 变体），主数据集 **Amazon23 `Industrial_and_Scientific`**（13,046 items / 16,163 条 test 样本）。

> 上游框架：MiniOneRec（Apache-2.0，arXiv 2510.24431，[HF](https://huggingface.co/kkknight/MiniOneRec)）；原始安装/全流水线说明见 **`README_OLD.md`**。本文档描述本仓库当前（2026-09）实际状态：含自研改造、实验记录与踩坑。

---

## 方法概览

```
交互序列 + item 文本
   │  rq/rqkmeans_constrained.py  （RQ-KMeans 语义码本 256×256×256，3 级）
   ▼
SID：<a_x><b_y><c_z>（碰撞桶成员追加 <d_x> 身份消歧 token）
   │  convert_dataset.py → sid/*/train|valid|test csv + info + index.json
   ▼
SFT（三任务混合，sft.py）      ← 消融证明：NTP-only 几乎学不会（HR@1 0.09% vs 4.05%）
   │  ① next-item 预测（历史 sid 序列→target sid）
   │  ② sid↔title 互译（metadata 语义注入）
   │  ③ 历史 sid 序列→target title
   ▼
RL（rl.py，ReReTrainer=trl GRPOTrainer 子类）  ← beam 约束解码 + 逐 prompt 双 trie
   │  奖励家族：rule / ranking / ranking_firstdiff（first-diff token 级）
   │  token 级 advantage：组内(group)或列(column) masked z-score + all_wrong_penalty
   ▼
评测：test 集 beam50 约束解码（evaluate.sh → evaluate.py → calc.py）
```

**双 trie 约束解码**（核心改造，docs/RL_IDEAS.md §9）：训练数据含两类任务——NTP 样本 target 是**全长 sid**（可含 `<d_x>`），对齐任务（title/desc→sid）target 只到**前 3 级前缀**。单一全长 trie 会强迫碰撞前缀续 `<d_x>`、屏蔽 EOS，使对齐答案不在解码支持集。故按 prompt 来源路由两套 trie：`hash_dict_full`（NTP）/ `hash_dict_prefix`（对齐，3 级即 EOS、`<d_x>` 不可达）。双 trie 纯效应（同配置 750 步对照，两个奖励家族一致）：seen×碰撞桶（S4）HR@50 **+1.6~1.7pp**、U4 +0.3~0.6pp、总量 ≈0（重分配而非提升）。

**first-diff token 级奖励**（收益性改造，rl.py `--reward_type ranking_firstdiff`）：

- **动机**：beam 生成整条 sid，但序列级奖励（rule 的 0/1、ranking 的秩分）是**稀疏的标量**——模型不知道"错在哪一层"。SID 是层次码（前 3 级 = 语义前缀，第 4 级 `<d_x>` 才消歧），早期路由 token 错了后面全白费；而约束 trie 内大量候选共享前缀，同一前缀列上多组样本的路由难度高度相关。序列级监督把 credit 平均摊掉，学不动"哪一步走错"。
- **做法**：从生成序列与 target 的**第一个不同 token**（first diff）开始做 token 级拆解——此前缀正确的 token 段获得正向梯度，分歧点起的 token 段按组内/列向 masked z-score 给优势；另配 `all_wrong_penalty`（整列全错的组附加惩罚）与 `token_norm`（group=组内 / column=跨组列标准化，抑制列间量纲噪声）。rule/ranking 仍作为标量项保留（`ranking_firstdiff` = rule + NDCG 排序惩罚 + first-diff token 级三分量加权）。
- **收益**（稳定复现于多个 regime）：
  - zero2 + 双 trie regime（完整 1 epoch，8 档 ckpt）：**fd 全档优于纯 ranking**，HR@50 恒 +0.45~0.48pp（fd_3000 10.98% vs bsl_3000 10.53%；NDCG@50 同步 +0.0018）；
  - 旧 regime（单 trie）fd 家族同为最高；SFT→RL 迁移分析显示 fd 的增益集中在 unseen/深层路由桶。
- **机制解读（谨慎表述）**：把序列级稀疏监督分解到"语义树上第一次走错的层级"，token 级优势给的是**位置明确的梯度**；配合列标准化缓解了不同前缀难度不均带来的比较噪声。结论基于单 seed + 双 regime 同向证据，属初步但稳健的工程性改进。

---

## 快速开始

GPU 配置：4× RTX 5090

### 0. 环境
```bash
bash setup_env.sh        # conda env recenv：torch 2.12 / transformers 4.57 / trl 1.12 / accelerate
pip install deepspeed==0.18.0   # zero2（本机实测 0.19.6 亦可）
wandb login              # 可选：wandb 记录
```
数据/模型（git 不入库）：`data/Amazon23/<category>/sid/rqkmeans-td-mean-20260830/{train,valid,test,info}/`、`data/pretrained_model/Qwen3-0.6B`。

### 1. SFT（标准入口）
```bash
bash sft.sh                     # mean 变体 × zero2 × 4 卡；产物 → outputs_ds/final_checkpoint
WANDB_RUN=my_run bash sft.sh    # 可选 wandb
# 消融（仅 next-item 预测）：
bash sft_ntp.sh                 # → outputs_ntp_ds/final_checkpoint
```
要点：`--batch_size` = **全局** batch（内部 gas = batch//micro//nproc）→ 1024 = 16×16×4；10 epoch 上限 + 早停 patience3（实测 best≈828 步）；`save_total_limit=4` 保证 best 不被轮换；每 ckpt 仅 ~1.5G（ds engine 状态已跳过，见下）。

### 2. RL
```bash
bash rl.sh                      # b/c 配置两段（ranking_firstdiff: group/column）
bash rl_2trie.sh                # 双 trie regime：ranking / ranking_firstdiff
bash rl_ds.sh                   # zero2 双段：baseline(ranking) / first_diff，各完整 1 epoch
```
要点：per-device batch 用 **16 + gas 2**（= 历史 32×1 的 128 prompts/步语义）——rollout 尖峰（fp32 SDPA）随行数线性，减半后 0.7 epoch 历史 OOM 不再复现；750 步 ckpt 自动归档 `ckpt_archive/`（对照锚点）。

### 3. 评测（test 集 beam50）
```bash
EXP_NAME=./outputs_ds/final_checkpoint bash evaluate.sh   # 任意模型目录
bash eval_rl_ds_ckpts.sh        # RL 8 档 ckpt 批量评测（已完成的自动跳过需手动按需）
```
结果 json → `results/<模型路径下划线>/`；指标口径 `calc.py`：HR@[1,3,5,10,20,50]、NDCG@[1,3,5,10,20,50]（NDCG 无 IDCG 归一；K=1 时 NDCG=HR@1）。

---

## DeepSpeed-ZeRO2 接入（本仓库自定义）

| 文件 | 作用 |
|---|---|
| `config/ds_zero2.json` | transformers 格式 zero2：stage2 无 offload；batch/gas/clip 全 `"auto"` 由命令行回填 |
| `ds_zero2_patches.py` | 三个配套 monkeypatch + `make_run_dir_name()`（rank0 广播 run 目录时间戳，防多卡跨秒竞态） |
| sft.py / rl.py 参数 | `--deepspeed_config config/ds_zero2.json`（空 = 关闭，纯 DDP） |

补丁做了什么（坑都在 docs/PROGRESS.md 2026-09-07 条目）：
1. **跳过 ds engine 的 optimizer 落盘**：zero2 ckpt 的 `global_step*/` 每档 ~7.8G（fp32 分片优化器状态，仅供引擎级 resume）→ 跳过，ckpt 缩到 ~1.5G（model.safetensors 由 HF 正常保存，eval/部署不受影响；resume 本就只恢复权重）；
2. **末端 best 恢复改 HF 直载**：`load_best_model_at_end` 在 ds 分支强制 `deepspeed_load_checkpoint`（要引擎文件，已被 1 跳过）→ zero2 每 rank 全量权重，各 rank 直接 `load_file(best/model.safetensors)`；
3. `save_total_limit=4`：早停 patience3 ⇒ best 之后至多 3 次保存，limit 4 保证 best 目录存活。

实测结论（SFT 与 RL 双验证）：**zero2 与普通 DDP 训练指标等价**（浮点归约路径差异在噪声内）；显存每卡省 ~6-7G（SFT 19.6-22G used/32G）；无加速（compute-bound，吞吐 ~298 样本/s 与 batch 无关）；RL 段 0.7 epoch 历史 OOM 的根治靠的是 **per-device batch 16 + gas 2**（削减 rollout 尖峰），zero2 只贡献基座余量。zero3 勿用（参数分片破坏逐 rank 全量权重的约束 beam 生成与 `sync_ref_model`）。

---

## 关键结果速览（截至 2026-09-08，详见 docs/PROGRESS.md）

| 实验 | HR@50（test beam50） | 结论 |
|---|---|---|
| SFT 三任务（mean 变体） | 9.72% | 主基线（best=828 步早停） |
| SFT zero2（同配方） | 9.66% | 与 DDP 等价（最大偏差 0.1pp） |
| SFT 消融 NTP-only | 3.56% | **metadata 对齐任务决定性**（HR@1 0.09% vs 4.05%） |
| **RL zero2 fd_3000（first_diff）** | **10.98%** | 双 trie regime 内 fd 全档 > ranking（+0.45~0.48pp） |
| RL zero2 bsl_3000（ranking） | 10.53% | 3000 步见顶，3750 微降 |

分桶分析（temp/bucket_analysis.py 等）：RL 相对 SFT 的净增益几乎全部来自 **unseen**（冷启动）；按训练交互频次分桶后增益随频次单调递减、F25+（中高频）转为显著损失（见 PROGRESS 2026-09-04 条目，含商品级 macro/聚类 bootstrap 口径）。

---

## 离线分析工具（temp/*.py）——零训练成本的分桶与机制验证

所有工具直接吃评测产物 json（`[{input, output, predict(≤50 beams)}]`），不碰训练；结果文本写 `results/*.txt`。它们把"RL 提升来自哪"从整体指标拆到桶/迁移/商品级，是论文与面试分析的主要来源。

| 工具 | 回答的问题 | 用法（示意） |
|---|---|---|
| `bucket_analysis.py` | 主分桶：seen/unseen × 3/4-token 九桶的 HR@K/NDCG@K/前缀存活 + SFT→RL 逐样本迁移 2×2（both/new/lost/neither）+ 配对 bootstrap Δ95%CI | `python temp/bucket_analysis.py --index <index.json> --train-csv "<train/*.csv>" --sft-json <SFT eval json> --model name=<RL eval json> [--model …] --out results/bucket_analysis.txt` |
| `item_level_analysis.py` | 商品级：macro-HR（商品内样本均值→商品等权）+ item-cluster bootstrap；unseen 按 **RL 对齐曝光**分 E1/E2/E3（复刻 RLTitle2SidDataset 采样，判定"前缀是否受对齐监督"） | `python temp/item_level_analysis.py --index … --train-csv … --item-json <item.json> --sft-json … --model name=json --out results/item_level_analysis.txt` |
| `freq_analysis.py` | **训练交互频次分桶**（长尾/冷启动/头部）：按用户重建真实交互序列（target 链不变量校验过，0/109269 违例），cnt_all 分 8 桶 × 行级/商品级 HR + cluster bootstrap Δ | `python temp/freq_analysis.py --index … --train-csv … --sft-json … --model name=json … --out results/freq_analysis.txt` |
| `prefix_survival.py` | first-diff 是否改善**早期路由**：d1-d4 逐深度 top1/top50 前缀存活、stop 正确率（前缀找对后停/续决策） | `python temp/prefix_survival.py <eval json …>` |
| `test_dual_trie.py` | 双 trie 行为单测（碰撞 item 在两种 trie 下的续 `<d_x>`/EOS 行为、`<d_*>` 全量不可达扫描、count-window 走树到 EOS） | `python temp/test_dual_trie.py` |
| `verify_firstdiff.py` / `verify_token_adv.py` | first_diff_reward / `_masked_column_advantages` 的语义单测（group 抵消病理、(b) all_wrong_penalty、(c) column 跨组恢复对比） | `python temp/verify_*.py` |

口径备忘（与 calc.py/评测逐位核对过）：HR@K=target 是否在 beam 前 K；NDCG 无 IDCG 归一（rank j → 1/log2(j+2)）；d_k 前缀存活分母 = T≥k 的样本；`seen` 判定 = target item 出现在 train 交互（history ∪ target）；频次桶的 cnt_all 是**去前缀重复后**的每用户真实交互数（勿用逐行 history 累计——早位置会被后续行重复携带）。

关键结论落点：冷启动/长尾/头部（freq 桶）→ PROGRESS 2026-09-04；unseen 增益通道（E1/E2/E3）→ 2026-09-03；商品级稳健性 → 同条目 item-level 表。

---

## 目录结构

```
sft.py / sft.sh            # SFT 三任务/消融（--train_tasks ntp）
rl.py / rl*.sh             # RL（ReReTrainer，奖励/双 trie/zero2 全参数入口）
minionerec_trainer.py      # ReReTrainer：trl GRPOTrainer 子类 + 双 trie + masked 优势
data.py                    # 数据集（SidSFTDataset、RLTitle2SidDataset…）
LogitProcessor.py          # 约束解码（count-window + trie 路由）
evaluate.py/.sh, calc.py   # beam50 评测与指标
rq/rqkmeans_constrained.py # SID 码本（RQ-KMeans）
ds_zero2_patches.py        # zero2 补丁（见上）
config/ds_zero2.json       # zero2 配置
temp/*.py                  # 离线分析工具（分桶/迁移/商品级/前缀存活，见"离线分析工具"节）
docs/PROGRESS.md           # 全量实验记录（含事故与修复，最重要）
docs/RL_IDEAS.md           # RL 设计讨论（双 trie、reward 家族）
```

---

## 文档导航

- `docs/PROGRESS.md`：实验日志/结论/事故记录（分桶、迁移 bootstrap、双 trie 效应、zero2、消融、8ckpt 评测）
- `docs/RL_IDEAS.md`：reward 变体、token 级优势、双 trie 动机
- `README_OLD.md`：上游框架原始说明（环境/数据准备/论文复现）

## 注意事项（踩过的坑）

- 磁盘空间占用：outputs/ckpt_archive/results 是占用大头，注意清理；zero2 引擎 ckpt 已跳过但仍需 ~7G/run
- resume 语义 = 权重级（优化器/scheduler 有意跳过）；续跑用 `--resume_from_checkpoint`
- `--batch_size`（sft）是全局口径；`--train_batch_size`（rl）是 per-device 口径
- wandb：脚本内 `WANDB_RUN` 非空才启用（online）；sft.py `report_to` 勿设 None（回落 "all" 会要求 wandb 登录）
- 评测口径一致性：结果目录 slug 已全路径化（防撞名覆盖）；不同变体（mean/td-last）需各自数据 glob
