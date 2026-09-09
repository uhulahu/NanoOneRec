# MiniOneRec-Enhanced：基于 LLM 与强化学习的端到端生成式商品推荐

本项目面向电商推荐中的**新商品冷启动**与**低频商品行为监督不足**问题，基于 Qwen3-0.6B 搭建从商品语义编码、Semantic ID（SID）构造、多任务监督微调，到推荐导向 GRPO 强化学习和约束生成的完整链路。

项目在 Amazon Reviews 2023 `Industrial_and_Scientific` 数据集上完成实验，覆盖 **13,046 件商品**与 **16,163 条测试样本**。核心思路是先利用商品标题与描述建立内容侧语义表示，再通过多任务学习连接商品语义与用户行为，最后利用结果级排名奖励和 First-Diff Token 级奖励优化层次 SID 的生成路径。

> 本仓库基于开源项目 [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 进行实现与扩展。本文重点介绍本项目新增的方法、实验和工程改造；上游原始说明见 [`README_OLD.md`](README_OLD.md)。

## 核心结果

| 实验 | 对照 | HR@50 | NDCG@50 | 结论 |
|---|---|---:|---:|---|
| 多任务 SFT | NTP-only SFT | **9.66% vs 3.56%** | 0.0573 vs 0.0102 | 联合移除两类 metadata 辅助任务后，HR@50 下降 6.10pp |
| Ranking GRPO | SFT 初始化模型 | **10.47% vs 9.72%** | **0.0613 vs 0.0569** | 完整 1 epoch 后，结果级推荐奖励带来稳定增益 |
| Ranking + First-Diff | Ranking GRPO | **10.91% vs 10.47%** | **0.0628 vs 0.0613** | First-Diff 在 epoch-end 额外提升 HR@50 0.44pp |
| 完整 RL | SFT 初始化模型 | **10.91% vs 9.72%** | **0.0628 vs 0.0569** | HR@50 / NDCG@50 相对提升 **12.3% / 10.5%** |

> 口径注记（重要）：第 2 行消融对照均为 **ZeRO-2** 复跑（同一训练协议，HR@1 4.12% vs 0.09%）；RL 各行的"SFT 初始化模型"指 RL 实际加载的 `./outputs/final_checkpoint`（mean 变体 SFT，DDP，best=828 步，HR@50 9.72% / NDCG@50 0.0569），ZeRO-2-SFT（9.66% / 0.0573）与其评测差 ≤0.1pp，视为等价。所有数字单 seed、来自同一训练轨迹的 checkpoint；完整 HR@K/NDCG@K 聚合见 [`reports/metrics.csv`](reports/metrics.csv)（可审计）。

在 750、2250、3000、3750 四组相同步数对照中，加入 First-Diff 后的 HR@50 均高于 Ranking baseline，增量分别为 **+0.48、+0.57、+0.45、+0.44pp**。这些 checkpoint 属于同一训练轨迹，不视为独立随机种子复现；当前结论限定为**单 seed 下跨训练阶段方向一致**。

## 项目贡献

### 1. Balanced RQ-KMeans Semantic ID

使用商品标题与描述的文本向量构建三级残差量化码，将每件商品表示为层次化 SID：

```text
item text -> text embedding -> Balanced RQ-KMeans
          -> <a_x><b_y><c_z>[<d_x>]
```

- 前三级 `<a_x><b_y><c_z>` 表示由粗到细的语义路径；
- 对共享三级前缀的碰撞商品追加 `<d_x>`，仅用于商品身份消歧；
- 在每级聚类中加入容量约束，避免码本坍缩并改善码字承载均衡；
- 当前主实验使用 `rqkmeans-td-mean-20260830` 版本，last-token pooling 版本的 SFT 结果与其基本持平（HR@50 9.57% vs 9.72%），因此后续实验保持主线 SID 不变。

实现见 [`rq/rqkmeans_constrained.py`](rq/rqkmeans_constrained.py)。

### 2. 多任务 SFT：连接商品语义与行为转移

SFT 阶段构造约 28.2 万条指令数据，联合训练三类互补任务：

| 任务 | 输入 -> 输出 | 作用 |
|---|---|---|
| Next-Item Prediction | 历史 SID 序列 -> 下一商品 SID | 学习用户行为转移 |
| SID-Text Alignment | SID <-> 商品标题 | 为新增 SID Token 注入商品语义 |
| Semantic Target Prediction | 历史 SID 序列 -> 下一商品标题 | 将行为上下文与自然语言商品空间连接起来 |

仅训练 Next-Item Prediction 时，模型主要依赖稀疏的 SID 共现关系；加入两类 metadata 辅助任务后，目录侧内容能够为低频乃至行为侧零交互商品提供语义锚点。

#### SFT 联合消融

消融实验一次性移除 `SID <-> title` 与 `历史 SID -> 下一 title` 两类辅助任务，其他训练配方保持一致：

| 模型 | HR@1 | HR@50 | NDCG@50 |
|---|---:|---:|---:|
| 三任务 SFT（ZeRO-2） | **4.12%** | **9.66%** | 0.0573 |
| NTP-only SFT（ZeRO-2） | 0.09% | 3.56% | 0.0102 |

HR@1 从 4.12% 降至 0.09%（相对 −97.8%），HR@50 从 9.66% 降至 3.56%，NDCG@50 从 0.0573 降至 0.0102，说明**两类 metadata 辅助任务作为整体，是建立 SID 语义路由的重要组成部分**。该消融没有进一步分离两个辅助任务的单独贡献。

实现见 [`sft.py`](sft.py)；标准训练与消融入口分别为 [`sft.sh`](sft.sh) 和 [`sft_ntp.sh`](sft_ntp.sh)。

### 3. 推荐导向 GRPO 与 First-Diff 信用分配

RL 阶段围绕 Next-Item Prediction 主任务，混合三类 prompt：

1. 历史 SID 序列 -> 下一商品完整 SID；
2. 商品标题或描述 -> 商品三级 SID 语义前缀；
3. 历史商品标题序列 -> 下一商品完整 SID。

三个子任务各采样约 10,000 条训练样本，使 RL 同时保留行为建模、内容对齐和自然语言历史建模能力。

#### Ranking reward

基础 GRPO 使用二值命中奖励。为区分同组候选的排序质量，本项目加入基于 Beam 位置折扣的排名惩罚：命中候选获得正向结果信号，高排名错误候选受到更强惩罚。

Ranking reward 能在"组内至少存在一个正确候选"时提供比纯 0/1 奖励更细的结果级对比；如果同组 16 个候选全部错误，该项仍为全零，因此它只能**缓解**而不能消除 exact-match 奖励稀疏。

#### First-Diff Token reward

SID 是层次生成路径，序列级标量奖励会广播到整条 completion，无法直接指出模型从哪一级开始走错。First-Diff 将监督细化到首次分歧位置：

- 首次分歧前的正确 SID Token：`+1`；
- 首个错误 SID Token：`-1`；
- 进入错误分支后的后续 Token：mask，不再归因；
- 完整 SID 及停止位置均正确：所有有效位置为 `+1`；
- 提前停止或多生成一级：在对应停止/分歧位置给出负反馈。

训练时，`ranking_firstdiff` 在相同的二值命中与排名奖励上增加 First-Diff Token advantage，从而构成干净的方法消融：

| Step | Ranking HR@50 | Ranking + First-Diff HR@50 | First-Diff 增量 |
|---:|---:|---:|---:|
| 750 | 10.06% | 10.54% | +0.48pp |
| 2250 | 10.26% | 10.83% | +0.57pp |
| 3000 | 10.53% | 10.98% | +0.45pp |
| 3750 | 10.47% | 10.91% | +0.44pp |

README 采用完整一轮结束时的 `checkpoint-3750` 作为主结果；`checkpoint-3000` 仅用于展示训练轨迹，不依据测试集将其选择为最终模型。

实现见 [`rl.py`](rl.py) 和 [`minionerec_trainer.py`](minionerec_trainer.py)。

### 4. 双 Trie 约束生成

不同任务的合法输出空间并不相同：

- Next-Item 与历史标题任务需要生成**完整 SID**，碰撞商品可能包含 `<d_x>`；
- 标题/描述对齐任务只预测**三级语义前缀**，不包含无语义的身份消歧 Token。

如果所有任务共用完整 SID Trie，碰撞前缀到达第三级后会被强制继续生成 `<d_x>`，导致对齐任务的正确答案无法合法结束。为此，本项目构建两棵前缀树并按 prompt 类型路由：

```text
Next-Item / History-Title -> Full-SID Trie
Title-or-Description Alignment -> 3-Level Prefix Trie
```

双 Trie 保证两类目标都处于解码支持集内。在 750 步同配置对照中（ranking 与 first-diff 两个奖励家族方向一致），seen 碰撞桶 S4 的 HR@50 提升约 **+1.2~1.7pp**、unseen 碰撞桶 U4 约 +0.3~0.7pp，同时 seen 唯一桶小幅让渡；其总体作用表现为**不同商品桶之间的收益重分配，而非整体 HR 的显著提升**（总量 ≤ +0.2pp）。

### 5. DeepSpeed ZeRO-2 多卡训练

SFT 与 RL 均接入 DeepSpeed ZeRO Stage 2，采用 BF16 四卡全参数训练（4× RTX 5090 / 32GB）：

- 模型参数仍在每张 GPU 保留完整副本；
- 梯度与 AdamW 优化器状态在多卡间分片；
- 不使用 CPU/NVMe offload；
- SFT 使用全局 batch 1024；
- RL 使用 per-device batch 16、gradient accumulation 2、16 beams。

RL 的采样器会将每个独立 prompt 重复 16 次形成一个 GRPO group。因此每个 optimizer step 对应：

```text
8 个独立 prompt groups x 16 条 beam completions = 128 条生成序列
```

将 per-device batch 从 32 降至 16，并把 gradient accumulation 从 1 调整为 2，在保持每次参数更新语义不变（步数/样本集/优化器更新时机与历史 32×1 完全一致）的同时降低了单次 rollout 的注意力峰值。ZeRO-2 提供基座显存余量，micro-batch 调整直接削减 rollout 峰值——历史 0.7 epoch 必现的 OOM（batch32 在 step~2661 崩溃）在 batch16×gas2 下不再复现，两组 RL 实验均跑满完整 1 epoch。

**资源实测**（训练日志 / GPU 采样 CSV 见 `logs/`，实测环境版本见 [`docs/environment.txt`](docs/environment.txt)）：

| 实验 | 显存（nvidia-smi 30s 采样，32G/卡） | 时长 / 吞吐 |
|---|---|---|
| SFT ZeRO-2（global batch 1024，早停于 1242 步） | 每卡 used 中位 20.2–21.0G、峰值 21.7–22.1G；GPU util 均值 92% | 73 min（4410s）；~3.5 s/步 |
| RL 每段（ranking / ranking_firstdiff，各 3750 步 = 1 epoch） | 训练典型 13–17G/卡、采样瞬时峰值 24.5G（单卡） | 每段 ≈ 8400s ≈ 2h20m（2.24 s/步，30k prompts/段） |

补充背景：zero2 之前同一 RL 配方的 DDP 瞬时峰值约 29.6G/32G（batch32，0.6-0.8 epoch OOM 的根因记录，见 PROGRESS.md 2026-09-03）；ZeRO-2 每卡约省 6–7G 的优化器基座（fp32 AdamW 分片），为 rollout 尖峰留出余量。

#### Checkpoint 工程适配

针对本地磁盘与 Transformers/DeepSpeed 保存流程，实现了以下工程改造：

1. 跳过体积较大的 DeepSpeed optimizer engine checkpoint，仅保留可用于评测和部署的模型权重（每 ckpt ~9.2G → ~1.5G）；
2. `load_best_model_at_end` 通过 `model.safetensors` 在各 rank 直接恢复最佳权重；
3. 由 rank 0 生成并广播输出目录时间戳，避免多进程跨秒启动造成路径不一致；
4. 保留足够数量的模型 checkpoint，防止最佳 checkpoint 在早停前被轮换删除。

该方案的 resume 语义是**权重级恢复**，不会恢复 optimizer、scheduler 和完整 DeepSpeed engine 状态。

实现见 [`config/ds_zero2.json`](config/ds_zero2.json) 和 [`ds_zero2_patches.py`](ds_zero2_patches.py)。

## 冷启动与流行度分析

本文将冷启动商品定义为：**测试目标商品在训练交互中出现次数为 0，但其目录侧标题/描述可用于 SFT 与内容对齐任务**。这对应"行为侧没有监督、内容侧 metadata 可用"的商品冷启动场景。

2026-09-09 已用最终 `fd_3750`、其 ranking 对照 `bsl_3750` 与 RL 的 SFT 初始化模型重新完成分桶（工具 [`temp/freq_analysis.py`](temp/freq_analysis.py) / [`temp/item_level_analysis.py`](temp/item_level_analysis.py)，test 集 16,163 样本，商品级 macro-HR 口径 + item-cluster bootstrap 95% 置信区间；产物见 `results/freq_analysis_3750.txt`、`results/item_level_analysis_3750.txt`）：

| 商品桶 | 商品数（样本数） | Δmacro HR@50（fd_3750 vs SFT） | 说明 |
|---|---:|---:|---|
| F0：训练交互 0 次（冷启动） | 1,065（8,494） | **+1.43pp [+1.05, +1.88]** | 稳定增益；新增命中商品 48 个、**丢失 0 个** |
| F1–F4：训练交互 1–4 次 | 623（1,541） | 弱正（F1 +1.28 / F2 +1.73 / F3-4 **+1.29pp [+0.35, +2.36]**） | 方向一致为正；单桶商品数少、置信区间较宽（幅度估计不精确） |
| F5–F24 | 1,933（3,725） | ≈0（+0.28 / +0.35pp，均不显著） | 中等频次基本持平 |
| F25–F99 | 519（1,718） | **−2.23pp [−4.36, −0.10]** | 一致转负 |
| F100+：真头部 | 88（685） | **−6.24pp [−12.54, −0.12]** | 头部商品让渡最重 |

要点：

1. **收益与商品流行度单调负相关**：冷启动（F0）显著正、长尾弱正、中频持平、F25+ 显著负——行级与商品级口径同向；fd_3750 在冷启动桶 48 个新增、0 丢失。
2. **与 ranking baseline 的差异**：ranking_3750 同样呈现单调形态（F0 +0.79pp [+0.53, +1.11]、F25-99 −2.48pp、F100+ −0.52pp 不显著），但幅度更温和——First-Diff 的 token 级监督把"容量向冷启动转移"做得更彻底，代价集中在 88 个真头部商品上。
3. **机制通道（unseen 按 RL 对齐前缀监督分 E1/E2/E3）**：直接内容对齐 +1.43pp、前缀级迁移 **+2.94pp**、无 RL 对齐监督的泛化 +0.96pp，三通道在最终模型上均统计显著（95% 置信区间整体不含 0），与"内容语义锚点经对齐任务注入、跨前缀泛化"的机制解释一致。
4. 早期配置（fd750/fd1500 锚点）的分桶形态与此一致，趋势非最终模型独有；完整历史见 [`docs/PROGRESS.md`](docs/PROGRESS.md)。

## 实验设置

| 项目 | 配置 |
|---|---|
| 数据集 | Amazon Reviews 2023, `Industrial_and_Scientific` |
| 商品数 | 13,046 |
| 测试样本数 | 16,163 |
| 硬件 | 4× RTX 5090 (32GB) |
| Backbone | Qwen3-0.6B Base |
| SID | Balanced RQ-KMeans, 3-level semantic prefix + optional identity token |
| SFT | BF16, 4 GPUs, global batch 1024, early stopping（~10 epoch 上限） |
| RL | GRPO, 3 × 10,000 prompts, 16 beams, 1 epoch（3750 步） |
| 分布式训练 | DeepSpeed ZeRO-2, no offload |
| 推理 | Trie-constrained Beam Search, beam size 50 |
| 指标 | HR@[1,3,5,10,20,50], NDCG@[1,3,5,10,20,50] |

本项目每条测试样本只有一个目标商品，因此 HR@K 与单目标场景下的 Recall@K 等价。NDCG@K 按目标商品首次出现的 Beam 排名计算折损（无 IDCG 归一化，故 NDCG@1 = HR@1）。

## 快速开始

### 1. 环境

依赖以 [`setup_env.sh`](setup_env.sh) 为准（`requirements.txt` 已与其 pin 对齐），实测版本记录在 [`docs/environment.txt`](docs/environment.txt)。核心：Python 3.12.3、torch 2.12.1+cu130、transformers 4.57.3（勿升 5.x：依赖 `use_model_defaults` 行为）、trl 1.12.0、deepspeed 0.19.6。

```bash
bash setup_env.sh        # 安装/核对核心依赖；或 pip install -r requirements.txt（含 torch，走 cu130 索引）
wandb login              # 可选：wandb 记录（脚本内 WANDB_RUN=xxx 启用）
```

### 2. 数据与模型

训练数据、预训练模型、checkpoint 和逐样本评测结果不进入 Git 仓库。运行脚本前需准备：

```text
data/
├── pretrained_model/Qwen3-0.6B/          # 也可用 Qwen3-Embedding-0.6B 生成向量
└── Amazon23/Industrial_and_Scientific/
    ├── raw/                              # 原始 .inter（已按时间切 train/valid/test）+ item.json
    ├── emb/                              # 商品文本向量 .npy
    └── sid/rqkmeans-td-mean-20260830/    # 主线 SID 变体（自包含）
        ├── *.codes_constrained.npy / *.codebooks_constrained.npz / *.index.json
        ├── train/ valid/ test/           # 交互 CSV（内嵌该变体 SID 字符串）
        ├── info/                         # 约束解码前缀表（sid \t title \t item_id）
        └── sid_eval.txt
```

从原始数据到 CSV 的四步流水线（本仓库相对路径；Amazon 原始数据下载与过滤参数见 `data/amazon23_data_process.sh`）：

```bash
# ① 原始下载 → 过滤切分（产出 raw/ 的 .inter 与 .item.json）
bash data/amazon23_data_process.sh        # 时间窗/交互数过滤参数见脚本头

# ② 商品标题+描述 → embedding（Qwen3-Embedding-0.6B）
bash rq/text2emb/amazon_text2emb.sh       # 现产 last-token+L2（-td-last.npy）；
                                          # ⚠️ 主线 -td.npy（mean pooling、无归一化）由早期
                                          #    GPR 分支脚本生成，npy 已保留于 emb/，无需重生成

# ③ embedding → Balanced RQ-KMeans 码（编辑 EMB_PATH/SID_VARIANT 指向目标变体）
bash rq/rqkmeans_constrained.sh           # K=256, L=3, 容量约束 min/max=n/K±1；n_jobs 必须=1

# ④ SID 码 + raw 交互 → train/valid/test CSV + info（变体目录自包含）
bash convert_dataset.sh                   # INDEX_DIR/OUTPUT_DIR 指向目标变体目录
```

变体命名与切换规则见 `data/Amazon23/Industrial_and_Scientific/README.md`：换 embedding/pooling 后按 ②→④ 重跑并全仓库替换变体目录名，CSV↔index 一致性用 `temp/` 下脚本抽样核对。

### 3. SFT 与消融

```bash
# 三任务 SFT（mean 变体 × ZeRO-2 × 4 卡，产物 outputs_ds/final_checkpoint）
bash sft.sh

# NTP-only 消融
bash sft_ntp.sh
```

### 4. 完整 RL 对照

```bash
# 依次训练 Ranking baseline 与 Ranking + First-Diff，均为完整 1 epoch（3750 步）
bash rl_ds.sh
```

### 5. 评测

```bash
# 评测任意 checkpoint
EXP_NAME=./outputs_ds/final_checkpoint bash evaluate.sh

# 批量评测完整 RL 对照的 checkpoint
bash eval_rl_ds_ckpts.sh
```

评测采用合法 SID 约束下的 Beam Search（beam 50），输出 HR@K、NDCG@K。逐样本 JSON 位于 `results/`（不入 Git）；聚合审计结果见 [`reports/metrics.csv`](reports/metrics.csv)。

## 目录结构

```text
rq/rqkmeans_constrained.py  # Balanced RQ-KMeans SID 构造
data/amazon23_data_process.sh / rq/text2emb/   # 原始数据过滤 / 商品文本 embedding
convert_dataset.py(.sh)     # 交互数据 → SID CSV + info
sft.py / sft*.sh            # 多任务 SFT 与 NTP-only 消融
rl.py / rl*.sh              # GRPO、Ranking、First-Diff 与完整训练入口
minionerec_trainer.py       # 推荐约束生成与 Token advantage（ReReTrainer）
LogitProcessor.py           # Trie 约束解码
ds_zero2_patches.py         # ZeRO-2 checkpoint 与多 rank 路径适配
evaluate.py(.sh) / calc.py  # Beam Search 评测与指标计算
temp/                       # 分桶、商品级、前缀存活率、双 trie 与奖励单测
reports/metrics.csv         # 主实验聚合指标（README 数字可审计）
docs/PROGRESS.md            # 完整实验记录与证据边界
docs/RL_IDEAS.md            # RL 设计讨论与消融动机
docs/environment.txt        # 实测环境版本
```

## 局限性

- 受限于时间和算力成本，当前主要结果来自 Amazon23 单一品类，尚未验证跨品类泛化；Ranking 与 First-Diff 的完整对照为单随机种子；
- SFT 消融同时移除两个 metadata 辅助任务，未对二者的独立贡献进行区分；
- RL 对冷启动与低频商品更有利，但**中高频与真头部商品命中率下降**（训练交互 ≥100 次的 88 个头部商品损失最重）：F25+ 各桶的下降在统计上显著（95% 置信区间整体为负，并非抽样噪声）；F100+ 桶因只有 88 件商品、置信区间很宽，下降幅度难以精确估计。该"偏向长尾"的副作用可通过**去偏**（训练中按商品频次加权、抵消对头部的系统性牺牲）或**头部保护**（为高频商品设置保底机制）等方向缓解，本项目尚未实现，可作为后续工作。

## 上游项目与许可证

本项目复用并修改了 MiniOneRec、TRL 等开源项目中的部分实现：

- MiniOneRec: <https://github.com/AkaliKong/MiniOneRec>
- MiniOneRec technical report: <https://arxiv.org/abs/2510.24431>
- MiniOneRec Hugging Face: <https://huggingface.co/kkknight/MiniOneRec>

本项目在 **Apache-2.0** 协议下发布（见 [`LICENSE`](LICENSE)）。代码基于 MiniOneRec 与 TRL（同为 Apache-2.0）修改而来，上游版权与修改声明见 [`NOTICE`](NOTICE)。
