# MiniOneRec 项目进展记录

> 记录时间：2026-08-31。本文档汇总 SID 构造 → SFT → 评估的完整结果、踩坑与代码改动，供后续回顾。
> 训练过程的逐次崩溃/修复细节见 [MONITORING_LOG.md](MONITORING_LOG.md)。

---

## 一、环境与配置

| 项 | 值 |
|---|---|
| GPU | 4×4090（24GB），torchrun 4 卡；曾用单卡 4090 |
| 环境 | conda env `recenv`，torch 2.3.0+cu121，transformers 4.57.3 |
| base model | Qwen3-0.6B（**base，非 Instruct**——README 公告提示 Instruct 在当前依赖下约束解码有 CC 风险，base 是官方 workaround） |
| 数据 | **Amazon23**（`data/Amazon23/`，Industrial_and_Scientific，**13,046 items**，时间窗 2018-10~2023-9）；论文同口径的 **Amazon18**（`data/Amazon/`，**3,685 items**，2016-10~2018-11，与论文 Table 4 完全一致）在仓库中可用 |
| 关键环境变量 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（解决显存池不收缩导致的反复 OOM） |

## 二、完成的工作

### 1. SID 构造（rqkmeans_constrained.py）

- 输入：Qwen3-E-0.6B 文本 embedding（`*.emb-qwen3-E-0.6B-td.npy`，last-token pooling + L2 归一化，13046×1024）
- 配置：**K=256，L=3**，约束簇大小（min=n//K-1, max=n//K+1），n_jobs 必须 =1（见踩坑 ②）
- 产物（`data/Amazon23/Industrial_and_Scientific/`）：
  - `*.codes_constrained.npy`（N×L 码）
  - `*.codebooks_constrained.npz`（每层 K×1024 质心）
  - `*.index.json`（item_id → `['<a_5>','<b_23>','<c_55>']`，碰撞 item 追加去重 token）
- 已知质量问题：对齐任务中 `combined_sid = sids[0]+sids[1]+sids[2]` 会静默丢弃碰撞 item（sid2title 仅覆盖 10,862/13,046）

### 2. SFT 训练（sft.py）——最终成功 run

- 配置：batch 1024 / micro 16（gas=16）/ 4 卡 / cosine lr 3e-4（warmup 20 步）/ 10 epoch 上限 / cutoff 512 / `EarlyStoppingCallback(patience=3)`（3 次 eval ≈ 1.5 epoch，论文为 patience=1 epoch，我们更宽松）/ freeze_LLM=False
- 数据：ConcatDataset 三任务（SidSFTDataset 129,296 + SidItemFeatDataset 23,865 + FusionSeqRecDataset 129,296 = 282,457 条）
- **训练结果**：从零训练，**4.5 epoch 早停**（step 1242/2760，train_runtime 66 min，train_loss 0.616）
  - eval_loss 曲线：138:2.89 → 276:2.39 → 414:2.24 → 552:2.16 → 690:2.15 → 828:**2.094**(best) → 之后连续 3 次未改善 → 早停
  - **best 模型 = checkpoint-828 权重**，已保存到 `outputs/model.safetensors` 和 `outputs/final_checkpoint/`
  - 全程无 OOM（expandable_segments 生效），花费 ≈ 8.7 元（4×4090 × 66min）

### 3. 评估（evaluate.py）——两轮修复后的最终指标

- 流程：split.py 切分 → 4 卡并行**约束解码 beam search**（`ConstrainedLogitsProcessor` + info 前缀表，只允许生成合法 item SID）→ merge.py → calc.py 算 HR@K/NDCG@K
- **修复 1（重要）**：transformers 4.50+ 的 `use_model_defaults` 行为会用**模型 config 默认**（`do_sample=True, temperature=0.6`）覆盖显式设置的默认值（`do_sample=False` 也判为"未设置"）→ 评估实际是 **beam 采样**而非纯 beam search。修复：`generate(use_model_defaults=False)` + 显式 `do_sample=False`
- **修复 2**：评估 prompt 与训练对齐（EvalSidDataset 的 `get_history` 改回训练版措辞，旧版注释保留）
- **最终指标**（Amazon23，纯 beam search 50 候选，16,163 测试样本全量）：

| K | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| HR@K | 0.0407 | 0.0470 | 0.0521 | 0.0605 | 0.0733 | **0.0978** |
| NDCG@K | 0.0407 | 0.0442 | 0.0463 | 0.0490 | 0.0523 | 0.0571 |

- 修复前（beam 采样）对比：HR@1 0.0334→0.0407（**+21.8%**），HR@50 0.0939→0.0978（+4.1%）——纯 beam 的增益集中在 top-1
- 参考：随机基线 HR@50 ≈ 50/13046 = 0.38%，当前 ≈ 25×；论文（Amazon18 小目录 3,685 items + Instruct + RL）HR@5 在 0.1~0.3 量级，**直接对比不公平**

## 三、关键经验与踩坑

1. **显存构成**（Qwen3-0.6B + 152k 词表，24GB 卡）：权重 1.1 + 梯度 1.1 + Adam fp32 m/v 4.8 + **logits ~7.5GB**（bf16 2.5GB + `.float()` 转换 5GB，micro=16）+ 激活 ~5GB ≈ **22-23GB**。词表是显存大头，micro 降到 4 也只能省 1-2GB（被 caching allocator 池子行为掩盖）
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**：解决"池子只增不减 + 碎片化"导致的反复 OOM（micro16 原跑在 step 138 必崩，开启后全程稳定）。零性能代价，**强烈建议所有 run 都加**
3. **resume 三连坑**（sft.py 顶部已有 2 个 monkeypatch 绕过）：
   - torch 2.3 < 2.6 → transformers 4.57 的 CVE-2025-32434 检查拒绝 `torch.load`（monkeypatch `check_torch_load_is_safe`）
   - DDP 下 Trainer 把 optimizer.pt（fp32 ~2.4GB/卡）加载到 GPU → 挤爆显存（monkeypatch 跳过 `_load_optimizer_and_scheduler`，代价：丢 optimizer 动量，仅恢复模型权重）
   - resume 后 `WORLD_SIZE` 可能未读到 → ddp 分支失效 → 有效 batch 变化（曾变 4096，无害但需注意）
4. **transformers 4.50+ 的 `use_model_defaults`**：显式设置等于默认值的参数会被模型 config 静默覆盖（`do_sample=False` → `True`）。评估/生成务必显式 `use_model_defaults=False`
5. **约束解码的前缀表**依赖 `### Response:\n` 恰好 = 3 个 token（`prefix_index=3`）的假设，换 base model 需重新验证
6. **sft.py 已加 `run_{时间戳}` 输出子目录**——每次运行自动隔离，避免跨 run checkpoint 覆盖
7. **checkpoint 保留机制**：`save_total_limit=1` + `load_best_model_at_end=True` → 磁盘保留"best + 最新"两个；旧 run 残留会被新 run 保存时自动清理

## 四、代码改动清单（相对原始仓库，均未提交）

| 文件 | 改动 | 位置 |
|---|---|---|
| sft.py | ① monkeypatch 绕过 CVE 检查 ② 跳过 optimizer/scheduler 加载（均带 `MONITORING` 注释）| 顶部 25-35 行 |
| sft.py | `import time` + `run_{时间戳}` 输出子目录 | 14 行、133-137 行 |
| evaluate.py | GenerationConfig 显式 `do_sample=False, temperature=1.0`；generate 加 `use_model_defaults=False` | 180 行、191-198 行 |
| data.py | EvalSidDataset prompt 改为与训练一致（旧版注释保留）| 623-624 行 |
| evaluate.sh | cuda_list 8→4 卡 | 30、38 行 |
| rqkmeans_constrained.py | `n_jobs=16→1`（joblib memmap 只读 bug）| 50 行 |

## 五、待办 / 下一步

1. **论文同口径复现**：sft.sh / evaluate.sh 数据路径切回 `data/Amazon/`（Amazon18，3,685 items）——item 数与论文完全一致
2. **RL 阶段**：rl.py（GRPO 2 epoch + constrained beam search）——论文指标的重要组成，当前是 SFT-only
3. **早停**：保持当前 patience=3（比论文 1 epoch 宽松），无需调整
4. **backbone**：维持 base（README 官方 workaround）；若换 Instruct 需先验证 CC 指标为 0
5. 训练质量观察：eval_loss 2.09 vs train_loss 0.62 存在过拟合 gap，SFT 阶段可接受（论文同样早停）

## 2026-09-03 RL（ranking_firstdiff, GRPO）评估归档 — 此表为权威记录

**run**: outputs_rl/run_20260902_211136（sample 10000 ×3 任务，ranking_firstdiff，epoch 0.6 = 2250 步即中断）
**口径**: test 集 + beam50 约束解码（evaluate_rl.sh 流水线）；原始 json 在 results/run_20260902_211136_checkpoint-{750,1500,2250}/final_result_Industrial_and_Scientific.json
**注**: calc.py 的 K 轴为 [1, 3, 5, 10, 20, 50]（v0.1 归档表曾错位一列：K=3 被标成 HR@5、K=50 被标成 HR@100，已修正）

**HR@K**

| K | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| SFT baseline | 0.0407 | 0.0470 | 0.0521 | 0.0605 | 0.0733 | 0.0978 |
| ckpt-750 | 0.0432 | 0.0506 | 0.0554 | 0.0640 | 0.0776 | 0.1031 |
| ckpt-1500 | 0.0426 | 0.0499 | 0.0561 | 0.0648 | 0.0784 | 0.1070 |
| ckpt-2250 | 0.0433 | 0.0505 | 0.0561 | 0.0658 | 0.0793 | 0.1057 |

**NDCG@K**

| K | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| SFT baseline | 0.0407 | 0.0442 | 0.0463 | 0.0490 | 0.0523 | 0.0571 |
| ckpt-750 | 0.0432 | 0.0475 | 0.0495 | 0.0522 | 0.0556 | 0.0606 |
| ckpt-1500 | 0.0426 | 0.0467 | 0.0492 | 0.0520 | 0.0554 | 0.0611 |
| ckpt-2250 | 0.0433 | 0.0475 | 0.0498 | 0.0529 | 0.0563 | 0.0615 |

**结论**: RL 相对 SFT 全 K 一致 +6~9%（如 HR@1 0.0407→0.0433，HR@50 0.0978→0.1070）；750 步后基本平台（三个 ckpt 互差 <1%）。中断（OOM）损失不大。
**精度提示**: 训中 eval（valid rollout 0/1 命中率）≈ 0.00406~0.00417 的差在 ±3e-4 噪声带内，不可用于 run 对比；本表 test 口径才可信。
**待续**: outputs_rl_baseline/run_20260903_000015（ranking 无 firstdiff，全量 sample 10000）跑完后用 evaluate_rl.sh 补同表对比 → 判定 firstdiff 是否有增益。

## 2026-09-03 RL baseline（ranking 无 firstdiff）评估归档 + 逐 token 前缀存活率分析

### 1. baseline run 测试集评估（outputs_rl_baseline/run_20260903_000015，ranking，sample 10000）

**口径**: test 集 16,163 样本 + beam50 约束解码（evaluate_rl.sh）；原始 json 在 results/outputs_rl_baseline_run_20260903_000015_checkpoint-{750,1125}/

**HR@K**

| K | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| SFT baseline | 0.0407 | 0.0470 | 0.0521 | 0.0605 | 0.0733 | 0.0978 |
| baseline-750 | 0.0431 | 0.0492 | 0.0547 | 0.0632 | 0.0759 | 0.1004 |
| baseline-1125 | 0.0416 | 0.0488 | 0.0544 | 0.0621 | 0.0747 | 0.0984 |
| firstdiff-750 | 0.0432 | 0.0506 | 0.0554 | 0.0640 | 0.0776 | 0.1031 |
| firstdiff-1500 | 0.0426 | 0.0499 | 0.0561 | 0.0648 | 0.0784 | 0.1070 |

**NDCG@K**

| K | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| SFT baseline | 0.0407 | 0.0442 | 0.0463 | 0.0490 | 0.0523 | 0.0571 |
| baseline-750 | 0.0431 | 0.0466 | 0.0489 | 0.0516 | 0.0548 | 0.0596 |
| baseline-1125 | 0.0416 | 0.0458 | 0.0480 | 0.0505 | 0.0536 | 0.0584 |
| firstdiff-750 | 0.0432 | 0.0475 | 0.0495 | 0.0522 | 0.0556 | 0.0606 |
| firstdiff-1500 | 0.0426 | 0.0467 | 0.0492 | 0.0520 | 0.0554 | 0.0611 |

**要点**:
- baseline-750 vs firstdiff-750（唯一干净对照：同数据/seed/步数，仅奖励函数不同）：HR+NDCG 共 12 个点 firstdiff 全 K 同向领先 1~3%（单点在 ~1σ 噪声内，方向一致为真实弱信号）
- baseline 750→1125 全 K 同向小幅回退（~1σ，6/6 同向）：ranking 单独在 750 步附近已到平台/微过峰；firstdiff 曲线 750→1500 仍爬升 → firstdiff 额外带来更新轨迹稳定性
- 两 run 相对 SFT 均 +6% 上下（result 级监督贡献主体增益；firstdiff 为增量）

### 2. 逐 token 前缀存活率分析（temp/prefix_survival.py，离线重分析 eval json，零额外推理）

**定义**: 目标 SID t 长 T∈{3,4}（test 集 11904 条 T=3 + 4259 条 T=4）；d 级存活 = 50 beams 中某条前 d token == t[:d]；条件存活 = alive(d)|alive(d-1)；停止正确率 = exact hit / 完整前缀存活（约束解码下恒 100%——前缀全对必整串命中）。

**逐层存活/命中（5 个模型列 = SFT | baseline-750 | fd-750 | fd-1500 | fd-2250）**

| 指标 | SFT | baseline-750 | fd-750 | fd-1500 | fd-2250 |
|---|---|---|---|---|---|
| d1 top1 | 11.0% | 11.1% | 11.3% | 11.7% | 11.8% |
| d1 top50 存活 | 43.4% | 40.9% | 39.6% | 39.1% | 38.9% |
| d2 top50 | 15.1% | 15.4% | 15.8% | 15.9% | 15.5% |
| d3 top50 | 10.8% | 10.9% | 11.3% | 11.6% | 11.6% |
| d4 top5 | 5.8% | 6.4% | 6.9% | 7.5% | 7.6% |
| d4 top50 | 12.7% | 13.1% | 13.2% | 13.9% | 14.0% |
| exact hit | 9.8% | 10.0% | 10.3% | 10.7% | 10.6% |
| T3 d2\|d1 条件 | 30.6% | 33.3% | 35.7% | 36.3% | 35.3% |
| T3 d3\|d2 条件 | 65.3% | 65.5% | 65.5% | 66.8% | 67.3% |
| T4 d2\|d1 条件 | 46.4% | 50.0% | 51.4% | 52.5% | 52.9% |
| T4 d3\|d2 条件 | 83.0% | 80.4% | 83.1% | 86.3% | 88.5% |
| T4 d4\|d3 条件 | 77.1% | 79.9% | 77.9% | 79.5% | 78.8% |

**结论**:
1. **firstdiff 未改善早期（d1）路由**：d1 top50 存活随 RL 单调降（43.4→38.9%，概率质量集中、beam 覆盖收窄），top1 仅微升（11.0→11.8%）
2. **firstdiff 增益在中深层条件路由**：T4 d3|d2 条件存活 80.4→88.5%（+8.1pp 最大单点）、d2|d1 +2.9pp；d3/d4 各档全线升。机制：+1 前缀奖励在"组内多条 beam 共享正确前缀"处产生强对比（中深层情形），a 层 90% 组首 token 即分歧、信号弱
3. **瓶颈层级 = d2（`<b_y>`）**：条件漏斗最陡在 d2|d1（T3 ~35%、T4 ~50%），其后每层 65-88% 走高；exact hit ≈ 0.43×0.33×0.65 乘性吻合。杠杆最大处是压 d2|d1
4. 停止决策非瓶颈（恒 100%）；提升 a/b 层路由需更强早期监督（如逐层分歧全罚）或查 SFT teacher-forcing 分层准确率判断可学性

**待续**: baseline resume（1125→3750）后续 ckpt-1500/1875/… 评估后补曲线；1125 为 resume 起点（见 rl.py --resume_from_checkpoint）。

> **方法学想法归档**：2026-09-03 基于本页 §2 存活率分析触发的机制讨论与四个改进方向（全错列惩罚 / 跨组标准化 / 离线 rankCE / 零梯度组剪枝），含核心论断"答案必须在支持集里"，见 [RL_IDEAS.md](RL_IDEAS.md)。

## 2026-09-03 想法 (b)(c) 实验归档 — 双 run OOM 中断于 ~2250-3000，≤2250 数据完整

**run**: b = outputs_rl/run_20260903_023633（token_norm group + all_wrong_penalty 1.0）；c = outputs_rl/run_20260903_041610（token_norm column）。均 sample 10000、ranking_firstdiff、save/eval @750。**两 run 均在 step ~2250-3000 区间 OOM 崩溃**（rank0，瞬时峰值 ~29.6G：batch 内长 prompt(~560 token) + fp32 math SDPA 注意力矩阵；[mem] 打点显示 allocated 恒 11.17G 无泄漏，为确定性数据序下的单步尖峰，同 seed 必复现）。750 均已归档 ckpt_archive/。

**test 集 HR@K/NDCG@K（beam50，K=1,3,5,10,20,50）**

| 模型 | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@50 |
|---|---|---|---|---|---|---|---|
| SFT | 0.0407 | 0.0470 | 0.0521 | 0.0605 | 0.0733 | 0.0978 | 0.0571 |
| baseline(ranking)-750 | 0.0431 | 0.0492 | 0.0547 | 0.0632 | 0.0759 | 0.1004 | 0.0596 |
| fd-750 | 0.0432 | 0.0506 | 0.0554 | 0.0640 | 0.0776 | 0.1031 | 0.0606 |
| **b-750** | **0.0443** | 0.0501 | 0.0540 | 0.0627 | 0.0741 | 0.0983 | 0.0597 |
| c-750 | 0.0433 | 0.0497 | 0.0538 | 0.0624 | 0.0748 | 0.0946 | 0.0585 |
| fd-1500 | 0.0426 | 0.0499 | 0.0561 | 0.0648 | 0.0784 | **0.1070** | 0.0611 |
| b-1500 | 0.0439 | 0.0509 | 0.0559 | **0.0648** | **0.0791** | 0.1018 | 0.0609 |
| c-1500 | 0.0430 | 0.0498 | 0.0549 | 0.0642 | 0.0755 | 0.0968 | 0.0590 |
| fd-2250 | 0.0433 | 0.0505 | 0.0561 | 0.0658 | 0.0793 | 0.1057 | 0.0615 |
| b-2250 | 0.0438 | **0.0511** | 0.0556 | 0.0639 | 0.0755 | 0.0984 | 0.0599 |
| c-2250 | 0.0435 | 0.0510 | 0.0553 | 0.0649 | 0.0752 | 0.0963 | 0.0593 |

**逐层前缀存活率（temp/prefix_survival.py；750 锚点：fd/baseline/b/c）**

| 指标 | SFT | baseline-750 | fd-750 | b-750 | c-750 | b-1500 | b-2250 | c-2250 | fd-2250 |
|---|---|---|---|---|---|---|---|---|---|
| d1 top1 | 11.0 | 11.1 | 11.3 | 10.6 | 11.1 | 10.8 | 10.3 | 11.7 | 11.8 |
| d1 top50 存活 | 43.4 | 40.9 | 39.6 | 38.7 | **44.7** | 36.0 | **33.9** | 43.3 | 38.9 |
| T3 d2\|d1 条件 | 30.6 | 33.3 | 35.7 | 34.1 | 28.4 | 36.6 | 37.4 | 28.9 | 35.3 |
| T4 d2\|d1 条件 | 46.4 | 50.0 | 51.4 | 51.2 | 47.8 | **54.6** | 52.5 | 48.4 | 52.9 |
| T4 d3\|d2 条件 | 83.0 | 80.4 | 83.1 | **86.4** | 83.5 | 87.1 | **87.5** | 86.3 | 88.5 |
| exact hit | 9.8 | 10.0 | 10.3 | 9.8 | 9.5 | 10.2 | 9.8 | 9.6 | 10.6 |

**结论（机制验证 + 反直觉发现）**:
1. **(b) 假设证伪一半**：全错列惩罚非但没止住浅层收缩，反而加速（d1 top50 38.7→33.9，全程最低）——−λ 把"磨尖分布"的引擎开到最大。但它把**条件深层推到全场最强**（T4 d3|d2 86.4-87.5 超 fd；T3 d2|d1 34-37.4 高）且 **HR@1 全场最佳（0.0443）**。定性：b = "更激进的 firstdiff"。
2. **(c) 假设证实**：跨组列标准化**确实保住浅层覆盖**（d1 top50 44.7 > SFT 43.4，全程平坦）——机制点成立；但代价是条件深层学习被稀释（T3 d2|d1 仅 28-29，全方法最低）→ exact hit 垫底 9.5-9.7。定性：c = 保 recall 但丢了组内对比锐度。
3. **fd 是均衡者**：浅层收缩温和（39.6→38.9）+ 条件深度第二 → HR@50 上限最高（0.1070）。
4. 三者构成明确 trade-off 面：浅层覆盖 (c) ⟷ 条件深度 (b) ⟷ 平衡 (fd)。**未探索组合空间**：(b)+(c) 叠加、λ 调小（0.5）、(c) 只对"全错列"做跨组等，接口已留（--token_norm/--all_wrong_penalty）。
5. 训练曲线截断于 2250（OOM），b 的中段最优点是 b-1500（HR@10/20 追平 fd-2250）。

## 2026-09-03 数据目录重组：emb/raw/sid 三分 + 变体命名规范（含 embedding off-spec 发现）

**动机**：发现实际使用的向量 `emb-qwen3-E-0.6B-td.npy` 为 **masked mean pooling 且无 L2 归一化**（npy 实测 L2 范数 52.7-87.8，0% 归一化；生成脚本 amazon_text2emb_gpr.py:126 直接 sum/mask，无 normalize），而规范做法是 `amazon_text2emb.py --pooling last`（last-token + L2，输出 -td-last.npy，代码注释明言 decoder-only 下 mean pooling 稀释信息）。归一化前后 K=256 划分 ARI=0.646 → 码本实质不同 → 全部 SFT/RL 实验基于 off-spec 码。方法间对比仍有效；绝对水平 vs 论文的差距可能部分源于此。
**重组**（规则详见 data/Amazon23/Industrial_and_Scientific/README.md）：
- `raw/` = 原始 .inter/.item.json（不可变）
- `emb/` = 向量（文件名后缀区分 pooling/归一化）
- `sid/<rqkmeans-<emb后缀>-<日期>>/` = 码 + index + **CSV + info 自包含**（用户拍板：CSV 进变体目录）
- 现有变体改名 `rqkmeans-td-mean-20260830`（全部实验的激活码，CSV↔index 抽样 2000 行 0 mismatch 验证无损）
- 五个脚本（sft/rl/rl_smoke/evaluate/evaluate_rl/convert_dataset .sh）路径全部参数化到变体目录，bash -n + glob 实解析验证通过
**遗留**：① data.py SidSFTDataset_GPR（死代码，硬编码旧路径，无 trainer 引用）② 探针（-td-last 变体 + SFT 66min 对比 HR）待做 ③ rl.sh 末尾 shutdown 在容器内无效（无 systemd），需改 poweroff 或控制台关机

## 2026-09-03 发现对齐任务 trie/target 错配 → 实现 per-task 双 trie

**用户发现**：trie 基于**全长 sid** 构建（碰撞 item 含 `<d_x>`），而 RLTitle2SidDataset 对齐 target 只到前 3 级；碰撞前缀在 3 级后 trie 只允许 `<d_x>`、EOS 被屏蔽。分析确认（量化见 docs/RL_IDEAS.md §9）：27.9% item 的对齐样本"满分不可达"（支持集铁律 violation），路由正确时被迫生成的 `<d_x>` 位被 first-diff 误判 -1；(b) 变体下全对组整组吃 -λ。
**实现**（用户拍板方向 A：per-task 双 trie）：
- minionerec_trainer.py：trie 构建抽成模块级 `build_sid_hash_tries(info_file, base_model)` → `hash_dict_full` + `hash_dict_prefix`（sid 截到 3 级再建树，`<d_x>` 不可达、任意 3 级前缀后 `\n→EOS`）；新构造参数 `trie_prefix_prompts`；`prefix_allowed_tokens_fn` 按行 kind 选表；`_prepare_inputs` 算 `_row_trie_kinds`（采样路径按行；**beam 路径按 dedup prompt 序号 `prompts[::G]`**——实际 RL run 全是 beam_search=True；dynamic 分支 1.5×G 扩展）。
- rl.py：`trie_prefix_prompts = set(train_data2.prompt2history.keys())` 传入 trainer。rl_gpr.py 未动（默认全长 = 旧行为）。
- 验证：temp/test_dual_trie.py PASS（400 抽样走树 + 全量扫描 `<d_>` 泄漏 0 + 5 桶 d 层完整性）。py_compile 通过。
- 未改 reward/rule/ndcg——对齐样本在 prefix trie 下天然全对可达，无需 reward 分支；k==T<L 情形随 prefix trie 消失。
**待做**：下一轮 RL run 验证——alignment completion 应停在 3 级、对齐组命中率回升（注：2026-09-03 晚用户纠正——实际 RL run 全是 beam_search=True（rl.sh:33/77），首版实现误将 beam 路径回退全长 trie 导致修复失效，已改为按 dedup prompt 序号存 kinds）。

## 2026-09-03 td-last 探针结论：mean ≈ last（平手），RL 主线维持 mean 变体

**探针**（用户跑）：td-last（last-token+L2，官方口径）SFT early-stop 846 步（outputs_lastpooling/run_20260903_191633/final_checkpoint）test 集 beam50 评测：
- HR [0.04065, 0.04721, 0.05191, 0.06212, 0.07319, 0.09565]；NDCG [0.04065, 0.04444, 0.04638, 0.04966, 0.05244, 0.05690]
**对比** mean SFT（828 步，PROGRESS 上方 SFT baseline 行）：HR [0.0407, 0.0470, 0.0521, 0.0605, 0.0733, 0.0978]，NDCG@50 0.0571。
**判定：统计平局**——12 点方向混合（td-last 在 HR@3/+0.4%、HR@10/+2.7%、NDCG@10/+1.3% 反超），仅 HR@50 −2.2%；全部在 ~1-3% 单点噪声判据内，n=1 且早停步数不同。
**码本结构**（eval_sid 两变体对比）：碰撞 16.7%/26.6%（mean）vs 17.9%/27.9%（last）——td-last 测试 exact-match 尾部更难（+1.3% item 需猜对 `<d_x>`）；brand purity、L1 均衡、桶分布基本一致。
**解读**：① 两版同一模型，官方 last+L2 优化的是检索排序，对"产品文本聚类→可预测码本"无单调保证；② HR 评测混入了码本差异，不是纯 embedding 质量；③ **off-spec 假设降级**——embedding pooling 不是与论文差距的来源（或至少方向不对），此前"全部实验基于 off-spec 码"的顾虑关闭。
**决策**：RL 主线维持 mean 变体（全部锚点 baseline/b/c/fd + ckpt_archive 可比）；td-last emb/codes/SFT ckpt 保留备用。遗留②关闭。

## 2026-09-03 数据事故：mean-SFT 逐样本 eval json 被 td-last 评测覆盖

**事故**：evaluate.sh 结果目录用 `basename $exp_name` 命名 → `./outputs/final_checkpoint`（mean SFT）与 `./outputs_lastpooling/run_20260903_191633/final_checkpoint`（td-last SFT）slug 同为 `final_checkpoint` → 跑 td-last SFT 评测时直接覆盖 `results/final_checkpoint/final_result_*.json`。检测：分桶分析时发现该 json 对 mean index 解析率 0%、对 td-last 100%，且 HR 复算与 td-last 报数逐位一致。
**影响**：11 个 RL eval json（fd/base/b/c）全是 mean 变体、100% 可解析、不受影响；mean-SFT（outputs/final_checkpoint，828 步）的逐样本 json 丢失 → SFT→RL 迁移/配对分析被阻塞，需重跑 evaluate.sh（结果将进 `results/outputs_final_checkpoint/`）。
**修复**：evaluate.sh slug 改为全路径下划线化（`sed 's|^\./||; s|/|_|g'`），杜绝再覆。
**新增分析工具** temp/bucket_analysis.py：seen/unseen × 3/4-token 九桶 × {HR@1/5/10/20/50, NDCG@10/20/50, d1-d4 top1/top50 存活, stop}，+ SFT→RL 迁移 2×2 与配对 bootstrap CI（待 mean-SFT 重评后启用）。口径与 calc.py/prefix_survival.py 逐位核对一致（fd750 全表吻合）。产物 results/bucket_analysis_11models.txt。

## 2026-09-03 分桶分析：unseen/seen × 3/4-token（11 模型，mean 变体）— 主要结果与结论

**定义**：seen = target item 出现在 train 交互（train csv history∪target，共 11902 item）；unseen = 训练交互零出现（但可能出现在对齐 title→sid 任务里——"内容语义"即指对齐学到的编码语义，非真正零样本）；tok3/tok4 = SID 长度 3/4（4 = 碰撞桶成员、含 `<d_x>`）。
**数据**：test 集 beam50 eval json（11 个 RL 模型：fd/base/b/c × ckpt-750/1500/2250，base 只到 1125）。

| 桶 | n 样本 | n 商品 | 样本/商品 |
|---|---|---|---|
| all | 16163 | 4228 | 3.8 |
| seen / unseen | 7669 / 8494 | 3163 / 1065 | 2.4 / 8.0 |
| S3（seen×unique） | 5748 | 2343 | 2.45 |
| S4（seen×碰撞） | 1921 | 820 | 2.34 |
| U3（unseen×unique） | 6156 | 762 | 8.1 |
| U4（unseen×碰撞） | 2338 | 303 | 7.7 |

**四象限 HR@1 / HR@10 / HR@50**（fd750 / base750 / b750 / c750；全 11 模型矩阵见 results/bucket_analysis_11models.txt）：

| 桶 | HR@1 | HR@10 | HR@50 |
|---|---|---|---|
| S3 | 5.65 / 5.51 / 5.60 / 5.53 | 8.59 / 8.61 / 8.42 / 8.12 | 15.05 / 14.60 / 13.73 / 12.84 |
| S4 | 7.96 / 8.07 / 8.22 / 8.33 | 13.79 / 13.69 / 13.22 / 14.47 | 22.80 / 23.58 / 23.48 / 24.62 |
| U3 | 2.88 / 2.94 / 3.01 / 2.88 | 3.18 / 3.18 / 3.25 / 3.18 | 3.87 / 3.64 / 3.77 / 3.57 |
| U4 | 1.88 / 1.84 / 2.18 / 1.92 | 3.38 / 2.87 / 3.25 / 2.87 | 5.35 / 4.58 / 5.00 / 4.19 |

**结论 Q1（内容语义能否召回零交互商品）——成立但弱，且 first_diff 后期收益集中在 unseen**：
- unseen 商品可被内容语义召回到 HR@50 3.6-5.4%（≈10× 全目录随机 0.38%），但只有 seen 的 ~1/4；HR@1 仅 1.7-3.0%。
- **first_diff 家族（fd/b/c）750→2250 的 HR@50 增量几乎全部来自 unseen**（fd：unseen 4.27→4.83；U3 3.87→4.66；同时 S3 15.05→14.37 反而下降，S4 22.80→24.57 上升）→ 后期训练继续改善语义/冷门路由、牺牲 seen unique 长尾。
- **ranking baseline 750→1125 与 first_diff 走向相反**：全线下行（unseen 3.90→3.61、S3 14.60→14.30）。两类奖励的后期行为差异是重要观察。
- 局限：单 run 跨步长趋势（非独立样本）、幅度 0.2-0.8pp；严谨判定待迁移配对 bootstrap（见 Q3）。
**结论 Q2（extra token 是否构成身份瓶颈）——不是瓶颈**：
- S4 在全部模型/步长上高于 S3（HR@50 ~23-25% vs ~13-15%），U4 ≥ U3（@50）。桶内频次无混淆（S3/S4 每商品 ~2.4 样本，U3/U4 ~8）。解读：碰撞桶=语义近邻、常共现的高可预测商品，`<d_x>` 在富交互上下文下完全可学；真正难的是 unique 长尾身份（S3）与 unseen 内容路由（U）。
- 与既有观察自洽：约束解码下 stop-correct 恒 100%（完整前缀存活 ⟹ exact），瓶颈从未在"停止决策"。
**结论 Q3（SFT→RL 迁移 + 配对检验）——被阻塞**：mean-SFT 逐样本 json 被 td-last 评测覆盖（见上节事故），须待当前双 trie baseline（tmux rl_2trie）跑完后重评 `./outputs/final_checkpoint`（evaluate.sh 的 exp_name；slug 已修，落 results/outputs_final_checkpoint/），再加 `--sft-json` 重跑 bucket_analysis 出迁移 2×2 与 ΔHR/ΔNDCG 配对 bootstrap CI。
**遗留**：① 按商品宏平均（每商品一票，消除 seen/unseen 每商品样本数 2.4 vs 8 的频次效应）暂不做（用户搁置）；② mean-SFT 逐样本 json 重评（evaluate.sh exp_name → ./outputs/final_checkpoint）→ 迁移配对分析。

## 2026-09-03 双 trie 纯效应（(b) 配置 750 步同配置对照）— 结论：S4/U4 系统性受益、S3 让渡、总量 ~0

**run**：ckpt_archive/run_20260903_232843（tmux rl_2trie，23:28 启动，750 步归档即停）。测试集评测 logs/eval_rl_2trie_baseline_ckpt750.log：HR [0.04405, 0.05086, 0.05562, 0.06410, 0.07560, 0.09918]，NDCG@50 0.06019；json 在 results/ckpt_archive_run_20260903_232843_checkpoint-750/。
**配置澄清（重要）**：用户以为跑的是 ranking，实际是 **(b) 配置**（ranking_firstdiff + token_norm=group + all_wrong_penalty=1.0，即 rl.sh 第一段）。判据：归档 trainer_state.json 750 步每步含 rewards/first_diff_reward —— 该指标仅 ranking_firstdiff 会实例化。故正确对照 = **旧 b-750**（run_20260903_023633，同配置/同 seed/同 SFT 起点/同步数，唯一差异 = 双 trie 代码）。
**分桶效应**（results/bucket_analysis_trie_vs_b750.txt，全 K 同向）：

| 桶 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---|---|---|---|
| all | +0.16 | +0.14 | +0.15 | +0.09 pp |
| S3（seen×unique） | −0.19 | −0.17 | −0.49 | −0.59 pp |
| S4（seen×碰撞） | +1.20 | +0.94 | +1.41 | +1.19 pp |
| U3 | +0.06 | +0.10 | +0.21 | +0.16 pp |
| U4（unseen×碰撞） | +0.43 | +0.34 | +0.51 | +0.65 pp |

**结论**：双 trie 修复激活的正是碰撞商品的对齐信号（3 级 target 从不可达变可学），碰撞商品（S4/U4）语义路由系统性提升（S4 @20 +1.41pp、U4 @50 +0.65pp），且 (b) 的 −λ 全错列惩罚在碰撞对齐组上基本失效（对组转成全对满分）；代价是 seen unique 长尾（S3）让渡 ~0.5pp；总量 ~0。单 run 同向趋势 → 方向可信、量级待配对 bootstrap 定量。

## 2026-09-03 SFT→RL 逐样本迁移 + 配对 bootstrap（Q3 闭环）— 核心结论：RL 净增益全部来自 unseen，seen 是零和/负和

**前置**：mean-SFT 重评完成（evaluate.sh 曾因 variant glob 仍指 td-last 而错评一次——mean 模型打 td-last 测试集 HR@10 0.044 vs 真值 0.060，教训：改 exp_name 时 glob 也要改；修正后重评 json 对 mean index 100% 可解析）。重评值 HR [0.04046, 0.04659, 0.05203, 0.06045, 0.07307, 0.09720] vs 记录 SFT 行（0.0407/…/0.0978）差 ≤0.0006 ——同模型复现，逐位差异来自早期评测协议版本差，量级在噪声内，记录行仍有效。
**分析**：results/bucket_analysis_sft_migration.txt（5 模型 × SFT，行序校验通过，bootstrap B=1999）。

**总体（all，n=16163，K=50 口径）**：

| 模型 | both | RLnew | RLlost | neither | ΔHR@50 (95%CI) | ΔNDCG@50 (95%CI) |
|---|---|---|---|---|---|---|
| fd750 | 1355 | 311 | 216 | 14281 | +0.59pp [+0.31,+0.86] | +0.0037 [+0.0026,+0.0049] |
| base750 | 1358 | 265 | 213 | 14327 | +0.32pp [+0.06,+0.59] | +0.0028 [+0.0015,+0.0039] |
| trieB750 | 1254 | 349 | 317 | 14243 | +0.20pp [−0.11,+0.50] | +0.0033 [+0.0020,+0.0046] |

（RL 保留 SFT 命中 ~86%（fd 1355/1571），新增/丢失比 ≈ 1.4-1.6:1——不是剧烈洗牌。）

**分桶迁移（K=50，RLnew / RLlost，ΔHR@50 95%CI）**：

| 桶 | fd750 | base750 | trieB750 |
|---|---|---|---|
| S3 | +152 / −135，+0.30 [−0.30,+0.87] | +130 / −139，−0.16 [−0.68,+0.40] | +137 / **−230**，**−1.62 [−2.26,−0.97]** |
| S4 | +42 / **−80**，**−1.98 [−3.12,−0.88]** | +48 / **−71**，**−1.20 [−2.34,−0.15]** | +82 / −84，−0.10 [−1.51,+1.20] |
| U3 | +56 / **0**，**+0.91 [+0.68,+1.15]** | +43 / 1，**+0.68 [+0.49,+0.91]** | +62 / 2，**+0.97 [+0.73,+1.23]** |
| U4 | +61 / **1**，**+2.57 [+1.92,+3.21]** | +44 / 2，**+1.80 [+1.24,+2.40]** | +68 / 1，**+2.87 [+2.18,+3.51]** |

**Q3 结论**：
1. **RL 相对 SFT 的净增益几乎全部来自 unseen（U3+U4），且 unseen 是"纯增"**——三个奖励家族（ranking / first_diff fd / (b)）RLlost ≤ 2 条，新增全被保留；bootstrap CI 全部显著为正、跨模型一致 → 单 seed 下结论稳定。U4（unseen×碰撞）增益最大（fd +2.57pp、trieB +2.87pp @50）：内容语义 + first-diff 后期持续投入 unseen 的路由/身份都学到了。
2. **seen 是零和/负和**：S3 fd 净 ≈0（有换手：new 152/lost 135），trieB 显著净负（−1.62pp）；**S4 在 fd/base 显著丢失尾部**（−1.98 / −1.20pp @50，CI 不含 0）——RL 把 seen 碰撞桶的 deep/identity 命中让渡给了 unseen。唯一例外：trieB 的 S4 相对 SFT ≈0（双 trie 修复把 S4 损失补回来了，与其机制一致）。
3. 与跨步长观察互相印证：之前"fd 后期增量在 unseen"→ 现在配对层面证实提升从 SFT 起就在 unseen 侧、seen 不增反减。

## 2026-09-04 双 trie 纯效应——ranking 家族同配置对照（run_20260904_005316，新 regime ranking baseline 补全）

**run**：ckpt_archive/run_20260904_005316（00:53 启动，750 步归档即停）。trainer_state 确认 **reward_type=ranking**（仅 rule/ndcg，无 first_diff_reward——上一轮 232843 曾误以为是 ranking 实为 (b)）。评测 logs/eval_rl_2trie_ckpt750.log：HR [0.04232, 0.04925, 0.05506, 0.06404, 0.07536, 0.10196]，NDCG@50 0.0596；json 在 results/ckpt_archive_run_20260904_005316_checkpoint-750/。
**对照**（旧 regime ranking baseline-750，同 seed/数据/步数，仅代码差异 = 双 trie）：

| 桶 | HR@1 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---|---|---|---|---|
| all | −0.08 | +0.04 | +0.08 | −0.05 | +0.16 pp |
| seen | −0.09 | +0.12 | +0.25 | −0.11 | +0.21 pp |
| unseen | −0.06 | −0.04 | −0.06 | −0.02 | +0.10 pp |
| S3 | −0.13 | +0.05 | +0.09 | −0.71 | −0.26 pp |
| **S4** | +0.05 | +0.31 | +0.73 | **+1.71** | **+1.62 pp** |
| U3 | −0.05 | −0.03 | −0.08 | −0.11 | −0.07 pp |
| U4 | −0.09 | −0.05 | +0.00 | +0.25 | +0.55 pp |

**结论**：与 (b) 家族同配置对照**形态一致、幅度相近**——双 trie 的效应 = 碰撞桶商品（S4 强 +1.6~1.7pp、U4 中 +0.3~0.6pp）系统性受益、S3/U3 小幅让渡、总体 ≈0。两个独立奖励家族（ranking 与 (b)/firstdiff）给出同方向证据 → 双 trie 修复"激活碰撞商品对齐监督"的机制在奖励类型间稳定；ranking 下 S4 @20/50 单点 ~1.6-1.7pp（n=1921，约 1.7-1.9σ），同形态跨家族复现使其可信。
**新 regime 对照现状**：ranking baseline 已补（本 run）；(c)/fd 的新 regime 对照仍缺（可选，若做方法学比较）。
**遗留**：① 若需方法学结论，同 regime 下重跑 (c)/fd 各一段（750 步即可）补全矩阵（ranking 已补）；② 双 trie 修复合入后，旧 regime 的 b/c 假设结论（(b) 磨尖浅层等）在新 regime 需重估（首段证据：trieB 与旧 b 同向但 S4 损失回补；ranking 对组同形态）。

## 2026-09-03 商品级 cluster bootstrap + unseen 对齐曝光分桶（第二次分析轮，零训练成本）

**工具** temp/item_level_analysis.py（复刻 RLTitle2SidDataset 构造 + random.seed(0)+sample(10000) 求曝光集）→ results/item_level_analysis.txt。

### A. 商品级（macro HR = 商品内样本命中率均值 → 商品间等权；Δ 用 item 级 cluster bootstrap，B=1999）
关键行（rowHR50 参照 | Δmac50 @K=50，95%CI）：

| 桶 | fd750 | base750 | b750 | trieB750 | n_newIt / n_lostIt（fd） |
|---|---|---|---|---|---|
| all | +0.02pp [−0.39,+0.47] | −0.14 | −0.09 | −0.05 | 108 / 80 |
| seen | −0.23 [−0.79,+0.34] | −0.41 | −0.40 | −0.39 | 80 / 79 |
| **unseen** | **+0.78 [+0.52,+1.11]** | +0.64 | +0.84 | +0.97 | 28 / 1 |
| U3 | +0.48 [+0.28,+0.70] | +0.36 | +0.51 | +0.42 | 15 / 0 |
| U4 | +1.54 [+0.81,+2.55] | +1.35 | +1.68 | +2.36 | 13 / 1 |
| S3 | +0.17 ns | −0.14 | −0.63 | **−0.95 [−1.65,−0.20]** | 64 / 49 |
| S4 | **−1.37 [−2.58,−0.16]** | **−1.17 [−2.35,−0.03]** | +0.24 | +1.22 [−0.21,+2.69] | 16 / 30 |

**A 结论**：
1. 总体宏观层面 Δ≈0（fd all +0.02pp），而行级 +0.59pp 显著 → **行级增益确有频次放大**（高频 unseen 商品贡献多行）；商品级视角总体打平。
2. **unseen 提升在 item-cluster bootstrap 下依然显著**（fd +0.78pp CI 不含 0；丢失 ≤2）→ 排除逐行频次造成的纯统计假象，但新增覆盖集中在 **23-39 个 unseen 商品**（base 23 / fd 28 / b 29 / trieB 39）；U3/U4 各自独立显著。
3. seen 损失同样在商品级显著（fd/base 的 S4 −1.2~−1.4pp；b/trieB 的 S3 −0.6~−1.0pp，trieB CI 不含 0）→ 非行级伪影；trieB 的 S4 +1.22pp（ns 正）与双 trie 修复方向一致。
→ 行级与商品级互证：RL = unseen 逐商品净增（~30 商品）+ seen 逐商品让渡（~80-100 商品有增有损），商品级净 ≈0。

### B. unseen 按 RL 对齐曝光分 E1/E2/E3（2026-09-03 修正版——用户发现旧版两个 bug 后重算）
**旧版 bug**：① 只存采样文本、不存 (text→target 3 级前缀) 配对 → 空 desc（804/1065 个 unseen 商品）恰好被采到 `""` 行 → 全被误判 E1；② E2 的"同前缀成员"只在 unseen 测试商品里找，漏掉 seen/全目录其他商品提供的前缀监督。修正：保存采样三元组 (task, text, target_3prefix)；E1=自身记录（文本↔前缀配对）被采；E2=前缀 ∈ sampled_prefixes（全目录）；E3=前缀未受 RL 对齐监督（≠"RL 没看过内容"：SFT 全量曝光过、U4 前缀可能经其他商品出现在 NTP）。
**修正后规模**（与用户独立重算逐位一致）：

| E组 | items | samples | U3 / U4 | 定义 |
|---|---|---|---|---|
| E1 | 617 | 4866 | 447 / 170 | 自身 (title/desc→自身前缀) 记录被采样 |
| E2 | 104 | 741 | 0 / 104 | 自身未采样，前缀被其他对齐样本监督（含 seen） |
| E3 | 344 | 2887 | 315 / 29 | 前缀未获 RL 对齐任务监督 |

**修正后结果**（rowHR50 SFT→RL；Δmac50 = 商品级 ΔHR@50，item-cluster 95%CI）：

| E组 | fd750 | base750 | b750 | trieB750 |
|---|---|---|---|---|
| E1 | +0.69pp [+0.39,+1.02] | +0.47 [+0.24,+0.75] | +0.82 [+0.40,+1.34] | +0.91 [+0.52,+1.42] |
| E2 | +2.02pp [+0.44,+4.37] | +2.11 [+0.53,+4.46] | +1.96 [+0.43,+4.22] | +2.35 [+0.71,+4.58] |
| E3 | +0.56pp [+0.29,+0.87] | +0.51 [+0.26,+0.82] | +0.54 [+0.29,+0.84] | +0.67 [+0.33,+1.11] |

**修正后结论**（推翻旧版"直接内容适配是 unseen 增益主体"——该命题未被可靠证明）：
1. **三组都有显著增益**：E1 直接内容适配（+0.5~0.9pp）、**E2 前缀级迁移幅度最大**（+2.0~2.4pp，全为 U4=碰撞桶商品：监督同桶任意成员 → 桶路由提升）、**E3 无 RL 前缀监督仍 +0.5~0.7pp 显著**（344 商品，功效充足）——存在超出 RL 对齐曝光的跨商品泛化（注意 SFT 全量曝光过，故为"RL 阶段强化信号向未再曝光前缀的迁移"）。
2. 旧版 E1 数字（+0.8~1.0pp）被空 desc 误判稀释/混杂，不可再用；新版 E1 仍显著但幅度小于 E2。
3. 无需再跑 sample=2000-3000 的判别实验（E3 已 344 商品、功效充足）。

## 2026-09-04 训练交互频次分桶（修正版：按用户重建真实序列，temp/freq_analysis.py → results/freq_analysis.txt）

**口径修正（重要，初版作废）**：初版逐行累计 history 有前缀污染——train csv 每行 = 同一用户序列的
前缀展开快照（第 k 行 history=前 k 个交互、target=第 k+1 个），早交互被该用户后续每行重复携带；
且 history 列只保留最近 10 个（长于 10 是滑动窗口，2205/20027 个用户的最长行只是窗口、不能当完整
序列取）。修正：全量验证不变量 last(H_{k+1})==T_k 0/109269 违例 → 行序即时序、target 链即真实
序列；每用户 S = 首行 history + 各行 target，每位置计一次（16 行完全重复快照去重；连续重复购买=
独立位置）。seen 商品总数不变（11902），但商品在桶间大洗牌：旧口径 F100+ 729 商品（虚胖）→ 真头部
（≥100 次真实交互）仅 **88 商品/685 样本**；F5-24 桶扩到 1933 商品。

分桶（n样本/n商品）：F0 8494/1065（52.6%）、F1 361/84、F2 339/108、F3-4 841/431、F5-9 1565/920、
F10-24 2160/1013、F25-99 1718/519、F100+ 685/88。tok4 占比桶间 15-27% 近似均匀（F1 最低 15%）。

**核心结果（ΔmacHR@50 SFT→RL，item-cluster 95%CI；9 个 RL 变体）**：
- **F0 冷启动：全变体显著正** +0.42~+1.18（fd1500 最高；lostIt≤5）——增益主体。
- **F1-F4 长尾 seen（1541 样本/623 商品）：方向一致正但多数不显著**（fd 家族 +0.4~+2.1，
  F3-4 fd750/fd1500/fd2250 显著 +1.13/+1.25/+1.94；rankT750 F1 +1.56 [0,+3.72]）。
- **F5-24（3725 样本/1933 商品）：≈0**（9 变体 ±0.8 内无一显著）。
- **F25-99（1718/519）：一致转负，6/9 显著**（fd2250 −3.56 [−5.75,−1.52] 最重；fd750 −1.57 ns、
  rank750 −0.97 ns、c750 −2.50*、b750 −2.81*、bT750 −2.79*、rankT750 −2.23*）。
- **F100+ 真头部（685/88）：全负、损失最重，6/9 显著**：fd750 −5.64 [−10.86,−0.76]、b750 −7.47、
  c750 −7.71、rank750 −7.31、bT750 −7.78、rankT750 −3.65（ns）、fd2250 −2.02（ns）；行级 @50
  −2.9~−9.5pp、HR@1 同步 −2~−4pp → 头部整体召回下滑，不是单纯被挤深。
- 行级 @50 损失（b750 F100+ −9.49pp）比旧口径更大；但 F25+ 合计只占 test 样本 14.9%，所以总量仍 ≈0。

**结论（修正后强化）**：RL 相对 SFT 的净增益是"训练交互频次"的**单调递减函数**——冷启动（F0）强
显著、长尾（F1-4）弱正、F5-24 平台、F25-99 起一致转负、真头部（88 商品）损失最重，全奖励家族
（fd/ranking/b/c 及双 trie 修复版）无一幸免。此前"净增益几乎全部来自 unseen、seen 零和/负和"细化
为：seen 内部 = 长尾弱正 + F5-24 零 + F25+ 显著负。"保护中高频/头部"是与双 trie（碰撞对齐修复）
相互独立的开放问题，缓解手段需朝去偏/头部保护方向设计。

SFT 绝对水平（HR@50/macHR@50）：F0 2.9/3.6% → F1 13.0/12.9 → F2 14.5/12.8 → F3-4 10.2/7.0 →
F5-9 7.5/5.1 → F10-24 11.6/9.2 → F25-99 22.7/18.9 → F100+ 56.1/41.8。F1-2 高于 F3-9 的凹坑为
test 难度特性（所有模型同构），疑与低频商品 test 样本密度高（F1 4.3 样本/商品 vs seen 均 2.4）及
临边界新品在 test 窗口扩散有关，机制未查。

## 2026-09-07 DeepSpeed-ZeRO2 接入 SFT + 质量对照闭环（全部完成）

**背景/动机**：4×5090 全参微调 Qwen3-0.6B，引入 transformers 原生 zero2（config/ds_zero2.json，
stage2 无 offload、batch 全 auto 由 trainer 回填）。配置接线：sft.py/rl.py 新增 deepspeed_config 参数
（默认 ""=关）→ TrainingArguments/GRPOConfig(deepspeed=...)；sft.sh/rl.sh/rl_2trie.sh 已带参；
rl.py 的 GRPO 侧（minionerec_trainer 继承 trl）钩子原生就绪（is_deepspeed_enabled→ref_model
prepare_deepspeed；ds3_gather_for_generation 仅 zero3 用）。evaluate.sh 增 EXP_NAME 覆盖。

**两个崩溃与修复**（都有日志佐证）：
1. 磁盘写满崩（step~414，torch.save unexpected pos）：zero2 engine ckpt 每 9.2G（model.safetensors
   1.5G + global_step*/ 优化器分片状态 7.8G），load_best_model_at_end 下保留 best+newest 两目录 +
   写入瞬态需 ~28G > 21G 空闲。修复：monkeypatch _save_optimizer_and_scheduler 在 ds 下跳过 engine
   落盘（ckpt →1.5G；resume 本就只恢复权重，与既有跳过 optimizer 加载补丁一致）。
2. 早停后收尾崩（rank1-3 ValueError Can't find valid checkpoint at checkpoint-828）：transformers
   load_best_model_at_end 的 ds 分支强制 deepspeed_load_checkpoint（要 engine 文件，已被 1 跳过）+ 
   best-828 目录被轮换删（limit=1 只保护 newest）。修复：save_total_limit=4（patience=3 ⇒ best 后
   至多 3 次保存）+ monkeypatch _load_best_model：zero2 每卡全量权重 → 各 rank 直接 HF 加载 best 目录
   model.safetensors（瞬态 +1.2G/卡），其余情形回退原版。

**zero2-SFT 训练实况**（tmux sft_ds，log sft_ds_zero2_20260907_162106.log）：
- 配置核对 [cfg] per_device16 gas16 world4 全局 batch=1024（sft.py 语义：--batch_size=全局，
  gas=batch//micro//world，曾误传 256 致 gas4/11040 步，已纠）＝ of-record 配方
- 早停同构：best=828（3.0ep, eval 2.0900）→ 3.5/4.0/4.5ep 未改善 → 停 1242/2760（4.5ep）
- 显存 4 卡 19.6–22.1G used / 32G（vs 非 ds 理论 ~27-29G，省 ~6-7G/卡 ≈25%）；util 88-99%；
  吞吐 ~298 样本/s（与 batch 无关 → compute-bound，zero2 通信开销可忽略，无加速）
- 每步 3.4s → 全程 ~1h16m；GPU csv：logs/gpu_sft_ds_zero2_20260907_162106.csv（30s 采样）
- 部署 outputs_ds/final_checkpoint（best-828 权重 + tokenizer；run 内保留 828/966/1104/1242 四 ckpt）

**质量对照闭环**（logs/eval_sft_ds_zero2_20260907_204604.log，beam50 约束解码，结果 json 在
results/outputs_ds_final_checkpoint/）：HR@[1,3,5,10,20,50] = [4.121, 4.721, 5.172, 6.107, 7.412,
9.658]% vs of-record mean-SFT [4.046, 4.659, 5.203, 6.045, 7.307, 9.720]% → 最大偏差 0.1pp
（<1.5% 相对），zero2 与 DDP 训练指标一致（浮点归约路径差异在噪声内）。train loss 尺度与记录行
(0.616) 不符疑为记录口径问题，评测数字为判据。

**结论**：zero2 在本规模无指标/速度收益，收益=每卡 +7G 显存余量 + 全链路模板（RL 侧随时可复用，
含三处补丁的教训：engine ckpt 跳过 / best HF 直载 / limit4 保护）。面试素材点：显存账（fp32
AdamW 7.2G→1.8G/卡）、zero2 vs zero3 选择理由（per-rank 全量权重约束生成）、transformers ds 集成
路径与三处源码行为。wandb 已接线（sft.py/rl.py：--wandb_run_name 非空启用 online；脚本 WANDB_RUN
变量；项目 MiniOneRec），未实跑验证。

## 2026-09-08 SFT 三任务消融（NTP-only）——metadata 对齐任务是决定性组件

**问题**：面试问"SFT 三个任务有无消融"→ 补最小消融：只保留任务1 next-item 预测（SidSFTDataset），
去掉任务2 sid-title 互译（SidItemFeatDataset）与任务3 历史sid→target title（FusionSeqRecDataset）。
配方与主 SFT 完全一致（全局 batch1024=micro16×gas16×4卡、lr3e-4 linear、10 epoch 上限+早停
patience3、seed42、zero2、cutoff512）。实现：sft.py --train_tasks ntp；脚本 sft_ntp.sh（输出
outputs_ntp_ds/）。

**事故与修复（记录在案）**：首跑 1270 步训练完成后末端 best 恢复崩溃——根因是 run 目录名
（time.strftime 秒级）在 4 rank 启动跨秒时不一致（双 run 目录 034543/034544，rank 间 trainer_state
错位）。修复：make_run_dir_name()（ds_zero2_patches.py）由 rank0 生成时间戳并
broadcast_object_list 广播，sft.py/rl.py 均已替换。成功半（034543）best-1216 权重完好 → 直接部署
outputs_ntp_ds/final_checkpoint 完成评测（无重训）。注：wrapper 用 ";" 曾致失败也写 done 标记，
已改为人工驱动。

**test 集 beam50 结果**（results/outputs_ntp_ds_final_checkpoint/，log eval_sft_ntp_rescue.log）：
HR@[1,3,5,10,20,50] = [0.09%, 0.40%, 0.73%, 1.06%, 1.54%, 3.56%]，NDCG@50 0.0102
对照：三任务 mean-SFT [4.05, 4.66, 5.20, 6.05, 7.31, 9.72]%；zero2-SFT [4.12, 4.72, 5.17,
6.11, 7.41, 9.66]%。
- **HR@1 相对下降 97.8%（0.09 vs 4.05pp 绝对值）**；HR@50 下降 ~63%（3.56 vs 9.7pp）
- 训练 loss 佐证：NTP-only 至 epoch 8+ 仍 ~2.8（三任务同期 ~0.3-0.5）

**结论**：metadata 对齐任务（sid-title 互译 + 序列 title 预测）是 SFT 的决定性组件——纯符号共现
（NTP）无法建立 sid token 的语义路由（top1 基本失效，仅长尾 HR@50 保留少量共现记忆）。该结论与
RL 阶段"unseen 增益依赖 title/desc 前缀监督（E1/E2）"的分析互相印证：内容的语义锚点从 SFT 的
metadata 任务注入。面试口径：消融下限 = NTP-only（HR@50 3.56%），三任务必要性成立；
任务2/3 单独贡献的进一步拆分（两两消融）未做（成本）。

## 2026-09-08 zero2-RL 双段评测（完整版：8/8 ckpt，含 06:30 关机后补跑的 3750 档）

背景：段1 baseline(ranking)→outputs_rl_baseline_ds/run_20260907_224824、段2 first_diff
(ranking_firstdiff+group+penalty0)→outputs_rl_firstdiff_ds/run_20260908_010923，各完整 3750 步
（zero2 + per-device16×gas2=128 prompts/步，与历史 32×1 语义一致）。全程无 OOM——历史 0.7 epoch
（step~2661）OOM 确认解决：v1(batch32) 在 2661 崩（进程 26.0G，尖峰=rollout fp32 SDPA 随行数线性），
v2 减半 rollout 行数后峰值 ~16.8-24.5G、两段各跑满 3750。评测=test 集 beam50 约束解码，json 在
results/<ckpt路径下划线>/，日志 eval_rl_ds_ckpts_20260908_052636.log + eval_fd_3750_rescue.log。

> ⚠️ **资产依赖：下表 `bsl_750` / `fd_750` 的权重只存在于 `ckpt_archive/`，不可删除。**
> `ArchiveCheckpointCallback` 用的是 `shutil.move`（搬运，不是复制；`minionerec_trainer.py:185`），
> 所以 `outputs_rl_baseline_ds/run_20260907_224824/` 和 `outputs_rl_firstdiff_ds/run_20260908_010923/`
> 里只有 2250/3000/3750 三档，**没有 750**。删掉 `ckpt_archive/` 就再也复现不出这两行：
> `ckpt_archive/run_20260907_224824/checkpoint-750` → bsl_750（4.21 / 10.06 / 0.0593）；
> `ckpt_archive/run_20260908_010923/checkpoint-750` → fd_750（4.31 / 10.54 / 0.0611）。
> 该目录另含 `run_20260904_005316/checkpoint-750`（2026-09-03 那节点名引用，见上）。

HR@[1,3,5,10,20,50]（%）；SFT 基线：mean [4.05,4.66,5.20,6.05,7.31,9.72] / zero2 [4.12,4.72,5.17,6.11,7.41,9.66]
| ckpt | HR@1 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@50 |
|---|---|---|---|---|---|---|
| bsl_750 | 4.21 | 5.50 | 6.37 | 7.59 | 10.06 | 0.0593 |
| bsl_2250 | 4.18 | 5.52 | 6.43 | 7.71 | 10.26 | 0.0598 |
| bsl_3000 | 4.29 | 5.63 | 6.63 | 7.92 | 10.53 | 0.0612 |
| bsl_3750 | 4.32 | 5.65 | 6.63 | 7.91 | 10.47 | 0.0613 |
| fd_750 | 4.31 | 5.63 | 6.43 | 7.73 | 10.54 | 0.0611 |
| fd_2250 | 4.39 | 5.78 | 6.63 | 7.94 | 10.83 | 0.0625 |
| fd_3000 | 4.42 | 5.80 | 6.68 | 8.15 | 10.98 | 0.0630 |
| fd_3750 | 4.42 | 5.73 | 6.67 | 8.12 | 10.91 | 0.0628 |

NDCG@[1,3,5,10,20,50]（同口径 calc.py，K=1 时 NDCG=HR@1）：
| ckpt | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|---|---|---|---|---|---|---|
| bsl_750 | 0.04207 | 0.04632 | 0.04859 | 0.05141 | 0.05445 | 0.05933 |
| bsl_2250 | 0.04182 | 0.04615 | 0.04858 | 0.05157 | 0.05477 | 0.05979 |
| bsl_3000 | 0.04288 | 0.04728 | 0.04961 | 0.05280 | 0.05605 | 0.06121 |
| bsl_3750 | 0.04325 | 0.04744 | 0.04984 | 0.05298 | 0.05623 | 0.06128 |
| fd_750 | 0.04306 | 0.04733 | 0.04968 | 0.05227 | 0.05553 | 0.06105 |
| fd_2250 | 0.04387 | 0.04827 | 0.05082 | 0.05353 | 0.05684 | 0.06254 |
| fd_3000 | 0.04424 | 0.04826 | 0.05096 | 0.05376 | 0.05747 | 0.06303 |
| fd_3750 | 0.04424 | 0.04815 | 0.05065 | 0.05367 | 0.05731 | 0.06283 |

**结论**：
1. **first_diff 全档优于 ranking**：HR@50 差 +0.45~0.48pp、NDCG@50 +0.002~0.003，750→3750 单调成立
   ——token 级 first-diff 奖励的正贡献在 zero2+双trie+完整 1 epoch 下复现（旧 regime fd>baseline 一致）。
2. **段内轨迹：两段都在 3000 步见顶、3750 微降**（bsl −0.06pp、fd −0.07pp @50）——cosine lr 收尾段的
   轻微过优化/饱和，量级在噪声内（NDCG@50 bsl 仍微升）；与旧 regime"ranking baseline 后期(750→1125)
   明显下行、fd 后期继续升"的形态不同——新 regime 两家族后期都趋平，可能受 3000 步后 eval 采样噪声
   或双trie 的影响，待更多档位验证。
3. **RL 相对 SFT 增益**：@3000 步 bsl +0.87pp / fd +1.32pp（HR@50），750 步即 +0.4~0.9pp——与旧
   regime"增益主要来自 unseen"的结论一致（分桶复核为可选后续）。
4. **zero2 vs 非 zero2（同 regime 锚点对照）**：与双trie 非-zero2 锚点 rankT750（ranking，HR@50
   10.196）相比，zero2 bsl_750 HR@50 10.06（−0.14pp）、HR@10 6.37 vs 6.40——噪声内等价，结合
   zero2-SFT 评测等价结论：**zero2 与 DDP 训练在指标层面等价**（RL 750 档亦成立）。
5. 完整度说明：06:30 定时关机截断了 eval8 后台任务（fd_3750 于 06:30 被杀在 30%），bsl_3750 实际已
   merge 落盘；fd_3750 于 11:35 补跑完成——8/8 全部闭环。教训：后台评测要按 ckpt 逐档独立落盘（脚本
   已如此），截断只损失进行中的单档。

## 2026-09-09 最终 checkpoint-3750 冷启动/流行度分桶复核 + 审计表与环境对齐

背景：README 主结果改用 fd_3750（epoch 末），需把此前基于早期配置（fd750/fd1500 锚点）的分桶结论
换到最终 ckpt 并核对 ranking 差异。工具 temp/freq_analysis.py + temp/item_level_analysis.py；输入 =
SFT=outputs_final_checkpoint（即 RL 初始化 mean-SFT，rowHR50 9.72%）+ bsl3750/fd3750 评测 json +
mean 变体 index/train；test 16163 样本三模型 0 不可解析；seed42、B=1999 item-cluster bootstrap
（log logs/bucket_final_run.log；产物 results/freq_analysis_3750.txt、results/item_level_analysis_3750.txt）。

**频次桶 ΔmacHR@50（SFT→fd3750，商品级 item-cluster 95%CI）**：

| 桶 | n样本/n商品 | fd3750 | bsl3750 | fd newIt/lostIt |
|---|---|---|---|---|
| F0（冷启动，交互0） | 8494/1065 | **+1.43pp [+1.05, +1.88]** | **+0.79pp [+0.53, +1.11]** | **48 / 0** |
| F1 | 361/84 | +1.28 [−0.07, +3.23] | +0.67 [−0.60, +2.23] | 3 / 1 |
| F2 | 339/108 | +1.73 [−0.32, +4.25] | +1.20 [−0.42, +3.56] | 5 / 1 |
| F3-4 | 841/431 | **+1.29 [+0.35, +2.36]** | +0.71 [−0.22, +1.73] | 7 / 1 |
| F5-9 | 1565/920 | +0.28 [−0.81, +1.36] | +0.23 [−0.75, +1.21] | 19 / 13 |
| F10-24 | 2160/1013 | +0.35 [−0.80, +1.57] | −0.49 [−1.64, +0.70] | 45 / 31 |
| F25-99 | 1718/519 | **−2.23pp [−4.36, −0.10]** | **−2.48pp [−4.57, −0.61]** | 32 / 38 |
| F100+（真头部） | 685/88 | **−6.24pp [−12.54, −0.12]** | −0.52 [−5.32, +3.84] | 4 / 9 |

行级 ΔHR@50 参照：F0 fd +2.63pp（2.91→5.53）、F100+ fd −5.84pp（56.06→50.22）；fd3750 全桶行级
F10-24 +1.39pp（11.62→13.01）而 mac +0.35ns —— 行级增益有高频行放大成分，商品级为准。

**结论**：
1. **单调形态在最终模型上成立且极端更强**：F0 显著正（fd 冷启动丢失商品 = 0，48 个新增）→ F1-4 弱正
   （1/3 显著）→ F5-24 ≈0 → F25-99/F100+ 显著负；与 2026-09-04 早期配置结论同形态。
2. **fd3750 vs bsl3750 的差异集中在真头部（88 商品）与冷启动**：F0 fd +1.43 vs bsl +0.79（fd 更强）、
   F100+ fd −6.24* vs bsl −0.52ns（fd 让渡远重）——first-diff 的 token 级监督把"容量向冷启动转移"
   做得比结果级 ranking 更彻底；中高频两家族都在让渡（F25-99 fd −2.23 / bsl −2.48 量级相近）。
3. **unseen/E-group（fd3750）**：unseen 整体 +1.43pp [1.04,1.85]、lost=0；E1（直接内容对齐）+1.43*
   [0.94,2.03]、E2（前缀级迁移）+2.94* [1.05,5.59]、E3（无 RL 对齐监督）+0.96* [0.54,1.43] ——
   三通道在 epoch 末全部显著（750 锚点时代 E2 +2.0~2.4 最强，此处仍最强）；seen 整体 fd −0.08ns
   （S3 −0.04、S4 −0.18），bsl seen −0.36ns（S4 −1.19ns）。
4. 商品级整体：fd3750 ΔmacHR@50 +0.30ns / bsl3750 −0.07ns（行级 +1.19/+0.75pp）——与 2026-09-03
   "行级增益为频次放大、商品级净≈0"一致；README 分桶节数字自此表。
5. 证据边界不变：单 seed、同轨迹 ckpt 非独立样本、F100+ CI 宽（88 商品），量级估计粗糙。

**顺带完成（README 定稿配套）**：
- 主实验聚合审计表 → **reports/metrics.csv**（11 行：SFT×3 + RL 8 ckpt，HR@K/NDCG@K 六档小数；
  RL 与 zero2/ntp 来自评测 log 逐档解析，mean-SFT 由 outputs_final_checkpoint 逐样本 json 按 calc 口径
  复算并校验 HR@50=9.72%）。
- **口径修正（重要）**：mean-SFT 当前评测协议 NDCG@50 = **0.05686**（早期记录 0.0571 为旧协议，
  差 ≤0.0006）。RL 相对 SFT 提升按同 json 口径 = HR +12.3%（0.10914/0.09720−1）/ NDCG +10.5%
  （0.06283/0.05686−1）；README 主表以此为准。
- 环境事实：docs/environment.txt（python 3.12.3 / torch 2.12.1+cu130 / transformers 4.57.3 /
  trl 1.12.0 / accelerate 1.14.0 / deepspeed 0.19.6，均与 setup_env.sh pin 一致）；requirements.txt
  精简对齐 setup_env.sh（删除上游 torchrec/fbgemm 等无关锁）；setup_env.sh 补 deepspeed pin 并修正
  smoke 指引为 run_smoke.sh。GPU 注记：训练/评测实验在 4×RTX 5090（logs/gpu_*.csv index 0-3），
  2026-09-09 复核实例仅暴露 1×5090。
- RL 段实测（供 README）：两段各 3750 步 ≈ 8401s ≈ 2h20m（2.24 s/步、30k prompts/段）；
  nvidia-smi 30s 采样训练典型 13-17G/卡、瞬时峰值 24.5G/32G（4 卡中单卡）；SFT zero2 ≈73min、
  每卡 used 中位 20.2-21.0G / 峰值 21.7-22.1G、util 均值 92%。

## 2026-09-16 LoRA 与码本初始化实验（E1–E5）—— 码本初始化把 LoRA 拉平到全参水平

**问题**：引入 LoRA 做效率消融时暴露了两件事——(a) 新增 783 个 SID token 让 LoRA 几乎学不动
（HR@50 仅全参的 86.1%）；(b) 由此提出假说「瓶颈是**容量不足**，不是初始化不好」。假说可检验：
把「学 SID 符号」这件事用 RQ-KMeans 码本向量**外包**出去，直接给新增 token 一个好初值。

**五路对比**（test 集 16163 条，beam50 约束解码）：

| | HR@1 | HR@50 | NDCG@50 | 占全参 HR@50 |
|---|---|---|---|---|
| 全参基线（E3） | 0.041205 | 0.096579 | 0.057291 | 100% |
| 全参+码本（E4） | 0.040958 | 0.094846 | 0.056461 | 98.2% |
| **LoRA+码本（E2）** | 0.037617 | **0.095774** | **0.055299** | **99.2%** |
| LoRA+打乱（E5） | 0.036503 | 0.093609 | 0.053856 | 96.9% |
| LoRA 基线（E1） | 0.015406 | 0.083153 | 0.036109 | 86.1% |

**结论**：
1. **码本初始化对 LoRA 是决定性的，对全参几乎无影响**（LoRA：HR@1 +144.2% / HR@50 +15.2%；
   全参：HR@50 −1.79%）。假说两个方向都验证了 —— **问题是容量不足，码本只对"学不动"的那一方有效**。
2. **它把 LoRA 从「明显不如全参」拉到「基本持平全参」**：HR@50 86.1% → 99.2%、HR@1 37.4% → 91.3%；
   且 LoRA+码本（0.0958）与全参+码本（0.0948）几乎重合 —— **两条线被码本拉到了同一水平**。
3. **机制归因（E5 打乱对照）**：打乱后仍保住 E2 提升的 HR@1 95.0% / HR@50 82.8%，残余 2–3%
   在六个 K 上方向一致 → **「放进 SID 该在的向量区域」是主效应，精确的 token↔向量对应叠加一个
   小而稳定的增量**。码本的价值是两部分叠加，不是单一机制。

**限定**：单 seed、单类目（Industrial_and_Scientific），无置信区间；「LoRA 追平全参」是跨配置比较
（LoRA 单卡 vs 全参 4 卡），严格性弱于 E1/E2/E5 那组干净 A/B；码本初始化还会让 LoRA 早停
（epoch 8.5 vs 基线 10.0），比较时注意训练量不同。

> **完整记录见 [LoRA与码本初始化实验.md](LoRA与码本初始化实验.md)** —— 实验设计、LoRA 配置、
> 实现要点（码本尺度重标定的坑、embedding 梯度掩码、resume 有损的证据链），以及一批可复用的
> 实测数据（显存公式 4.8 MiB/token、KV cache 112 KiB/token、LoRA 在单卡为何省不动显存）。

---

## 索引：专题文档

本文档是**主线时间线**记录。以下专题独立成文（避免主文档过长），需要细节时直接跳转：

| 文档 | 内容 | 最后更新 |
|---|---|---|
| **[LoRA与码本初始化实验.md](LoRA与码本初始化实验.md)** | 效率消融（LoRA vs 全参）+ 用 RQ-KMeans 码本初始化新增 token 的 embedding，5 组实验（E1–E5）含机制归因 | 2026-09-16 |
| [SFT_IDEAS.md](SFT_IDEAS.md) | SFT 阶段的改进想法与验证记录（含码本 E1–E5 结论、待跑消融清单） | 2026-09-18 |
| [RL_IDEAS.md](RL_IDEAS.md) | RL 阶段的想法与实验设计（含 logits 切片想法 §10） | 2026-09-18 |
| [RL黑话与GRPO实现详解.md](RL黑话与GRPO实现详解.md) | GRPO 实现细节与术语详解、代码位置速查 | 2026-09-18 |
| [ReReTrainer修改分析.md](ReReTrainer修改分析.md) | ReReTrainer 相对上游 TRL 的逐段改动分析 | 2026-09-18 |
| [MONITORING_LOG.md](MONITORING_LOG.md) | **历史流水**（2026-08-30 ~ 09-03）：训练过程逐次崩溃/修复细节 | 2026-09-03（另有历史声明） |

> `docs/environment.txt` 为环境事实快照（python / torch / transformers / trl / deepspeed 版本），2026-09-09 实测与现环境一致。
>
> ⚠️ **不在此表内的 `docs/*.md` 一律为个人材料，已被 `.gitignore` 排除、不在公共仓库中**
> （如 `面试追问预演.md`、`磁盘清理记录.md`）—— 因此本表**不列**它们，以免在 clone 下来的仓库里指向不存在的文件。

