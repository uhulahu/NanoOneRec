# RL 改进方向备忘录（2026-09-03 讨论归档）

> 来源：逐 token 前缀存活率分析（temp/prefix_survival.py，见 PROGRESS.md §2）触发的机制讨论。
> 本文档收录"想清楚了机制、尚未实验"的方法学想法，供后续实验设计；验证数据见 PROGRESS.md。

## 0. 诊断结论（事实基础）

test 集 beam50 逐层存活漏斗（SFT | baseline-750 | fd-750/1500/2250）：

- d1 top50 存活：43.4% → 40.9% → 39.6/39.1/38.9%（**RL 单调收缩**）
- d1 top1：11.0% → 11.8%（微升）
- 条件存活：T4 d3|d2 80.4→88.5%、d2|d1 46.4→52.9%（**RL 显著改善**）
- exact hit 9.8→10.6%（深层增益 > 浅层收缩损失，净为正）
- 停止正确率恒 100%（前缀全对必整串命中，停止非瓶颈）

即：RL 改善**条件深层路由**（前缀已对时的续写），损害**无条件浅层覆盖**（beam 支持集的宽度）。

## 1. 核心论断：政策梯度的"支持集铁律"

> **本质是把"答案必须在支持集里"这条政策梯度铁律打破——让 target 直接出现在 loss 里，浅层才真正可学。**

机制链（三层）：

1. **PG 梯度只作用于已生成 token**：`∇L = adv · ∇log π(已生成路径)`，未被 beam 生成的 token 梯度恒为 0 → 答案不在 16 条 beam 里 = 该步无法学习"把答案换进来"。
2. **组内 z-score 只做支持集内对比，且整组全错时整体抵消**：全组在 position 0 分歧（无一条 beam 的 a 对）→ firstdiff column 0 全 -1 → masked z-score `(v-v̄)/(0+ε)=0` → 连"我的支持集在这一层全错"这个 anti-信号都被当 baseline 抹掉（这是 GRPO 的 baseline 设计，但此情形下它抹掉了该保留的信号）。
3. **softmax 重归一化漂移**：每步抬升已生成路径 logp = 隐式压低一切未生成候选 → 确定性 beam 反复强化自己的模式 → 浅层支持集越磨越尖（d1 top50 43.4→38.9 的来源）。SFT 无此效应：CE 无条件直接抬升 target token，与 target 在不在模型当时分布里无关 → SFT 浅层覆盖最高。

推论：

- **RL 教的不是"哪个答案对"，而是"你生成的这些里哪个更好"**——深层错误时答案在支持集内（前缀已对），学得动；浅层错误时答案在支持集外，学不到，只剩重归一化磨尖。
- firstdiff 把"哪个更好"归因得更细 → 学得动处学得更好（深层条件 +8pp）、学不动处收缩更快（浅层覆盖 -4.5pp）——同一机制的两个镜像。
- **任何"答案进不了支持集"的错误，policy-gradient 族结构性不可学**；修复必须让 target（或其序列级 logp）直接进入 loss（CE / rank / contrastive），或把答案带进支持集（采样拓宽）。

## 2. 想法 (a)：去掉标准化用全 raw（讨论中否决的基线变体）

- 机制：adv = 原始 ±1/0，不做组内 z-score → 全错列不再抵消。
- 后果：等于 REINFORCE + 稠密 shaped reward；全错列变成"主动 anti-支持压力"（压低已生成错误 a → 重归一化把质量推给未生成候选，对冲机制 3 的被动漂移）。
- 问题：失去尺度校准（难组易组同幅度）、列间语义尺度混杂；±1 有界故方差可控，确定性 rollout 下 PG 方差担忧弱。
- 地位：被 (b)/(c) 取代，仅作思路记录。

## 3. 想法 (b)：z-score 保留 + 组外"全错列惩罚"（推荐首试）

- 机制：在 `_compute_token_advantages` 的归一化之外，对"该列所有 mask 成员同值（全 -1，即整组该层全错）"的列附加 `-λ`（不进 z-score）。
- 语义：精准补上唯一被 z-score 抹掉的病理信息（整组在 position j 全错），其余列保持现有组内对比语义不变。
- 性质：**anti-支持压力**——压低 16 个已生成候选 → 重归一化抬高未生成候选（含潜在正确 a）。它不能告诉模型哪个 a 对（正确答案不在支持集，任何组内方法都做不到），只对抗漂移、保住浅层覆盖。
- 成本：~10 行内，全部改动在 minionerec_trainer.py `_compute_token_advantages`；多一个超参 λ。
- 验证：小规模 run（sample 2000）A/B vs 现版 ranking_firstdiff → prefix_survival.py 看 d1 top50 存活率是否停止收缩。
- **状态（2026-09-03）：已实现**。接口：rl.py `--all_wrong_penalty <λ>`（默认 0 = 关），实现为 `_masked_column_advantages` 纯函数（minionerec_trainer.py），单测 temp/verify_token_adv.py。

## 4. 想法 (c)：列级 z-score 改跨组（batch/全局）标准化

- 机制：同列不同组的 ±1 一起做 z-score → "a 全错组" vs "含 a 对组"之间重新有对比 → 全错是全局性的坏重新可学。
- 性质：打破组内相对语义（GRPO baseline 从"同 prompt 16 条"变成"同列全体"），需要跨进程 gather；全局列均值漂移需跟踪。
- 成本：中（gather 已有基础，语义改动大）。
- 与 (b) 的关系：二选一即可，先试 (b)（改动小、语义清晰）。
- **状态（2026-09-03）：已实现**。接口：rl.py `--token_norm column`（默认 "group" = 原实现），实现同 (b) 的 `_masked_column_advantages` 纯函数。与 (b) 正交可叠加（`--token_norm column --all_wrong_penalty λ`）。

## 5. 想法：离线对比 / rank loss（target 进 loss，治浅层的正解）

- 形态 1（expert iteration 朴素版）：pi_old 生成 beams → 打标 → 离线训练。**陷阱：loss 若仍只作用于生成 token，target 不在生成里照样学不到浅层**——与 on-policy 同病。
- 形态 2（正确版）：候选集 = {target 全序列, pi_old 的 16 条 beams}，按前缀对齐深度排序，训 listwise/rank 损失：
  ```
  L = -Σ_t log p(target_t | ctx)                # 直接监督 = SFT 机制（修浅层）
      + λ · RankLoss(target > beam_1 > beam_2 > ...)   # 按前缀深度排序的负例压制
  ```
  **关键：rank loss 需要计算 log p(target)，梯度直接走 target 序列——不要求 target 出现在任何生成里。** 这就是"打破支持集铁律"的具体形态：SFT 的直接监督（浅层覆盖最高）保留，同时补上 SFT 没有的负例压制。
- 为什么值得做：
  1. 采样侧自由：pi_old 生成时 temp>1 / 更多 beams / 多次 seed → 支持集远宽于确定性 beam16
  2. 老师选择自由：用最好的 ckpt（如 fd-2250）生成 → 命中多、near-miss 硬（信息量大的负例）
  3. 成本解耦：30000 prompts × 16 一次离线生成约 1-2 GPU-时，缓存 corpus 后所有 loss 消融都是 SFT 量级 run（~66 min），而非 4-5 小时 RL run
  4. 统计上干净：rank/contrast 分类无重要性比问题，pi_old 与 pi_θ 漂移不破坏目标函数
- 风险：负例的浅层上限 = pi_old 的浅层覆盖（治标需周期刷新 corpus，如每 N 轮用当前策略重生成）；前缀深度排序假设在 RQ 层级语义下成立但非完美；λ 需控制。
- 备注：firstdiff 的信息 ⊂ CE（正方向）+ 负例（CE 没有）；作为"纯 SFT 监督"没有增量，增量只在带负例的 rank/对比形态中。

## 6. 想法：剪掉保证零梯度的组（算力剪枝，非方法改进）

- 事实：a 层 miss 组（16 条 beam 无一条 a == target.a）→ rule/ndcg 恒零 + firstdiff column0 全 -1 被 z-score 抵消 → **整组对损失贡献精确为零**（仅剩可忽略的 KL 项）。
- 剪枝 = 精确无偏（零梯度组的期望贡献为 0；替换 prompt 迟早也会被采样）。不是近似。
- 收益：~65-70% 组在 a 层死亡 → 省其 ~4/5 生成步 → step 时间 −20~30%（生成占 step 40-60% 的估计下）。
- 实现：约束 beam 第一步 = 同前缀 top-16 next-token → 只需一次 `logits_to_keep=1` 的 forward 检查 target.a ∈ top16（trie 过滤后），幸存者才做完整生成（第二段需重新 prefill，KV 不便跨调用续）。也可做第二层再筛，边际收益递减。
- 陷阱：
  1. 幸存者偏差不破坏梯度，但**固化浅层盲区**——剪枝后模型永远看不到"自己 a 层错"的样本；若未来上 (b)/(c)（全错列不再是零梯度），剪枝必须同步关掉，否则变有偏。
  2. batch 组成失真 → categorical_diversity 等监控指标不可再解读为真实分布。
  3. 纯速度优化：不产生新信号；省下的预算用于更大 sample / 更多 epoch。

## 7. 想法间关系与建议实验顺序

```
(现版 ranking_firstdiff)  ← 基准，已归档
   │
   ├─ 加 (b) 全错列惩罚 ──────────────── 10 行改动，先试
   ├─ 或 (c) 跨组列标准化 ────────────── (b) 无效再试
   │
   ├─ 剪枝零梯度组（若嫌慢）────────── 纯提速，独立可叠加，注意与 (b)/(c) 互斥
   │
   └─ 离线 rankCE（若浅层要真正回升）── 1-2 GPU-时 建 corpus + SFT 量级消融
```

统一验证协议：小规模 run（sample 2000/epoch 1）→ evaluate_rl.sh test 评估 → prefix_survival.py 查 (i) d1 top50 存活率（浅层收缩是否止住/回升）、(ii) exact hit、(iii) 条件深层存活；训中 eval 噪声大不可用于比较。

## 8. 一句话备忘

- 支持集铁律：PG 只能学"支持集内的重排"；答案在支持集外时，唯一机制是重归一化漂移（有害）或 anti-支持压力（(b)/(c)，保住覆盖但不指向答案）。
- 浅层真正可学的唯一路径：target 直接出现在 loss（CE/rank/contrastive），即离线形态 2。
- 全错列惩罚 / 跨组标准化 / 剪枝的语义边界：前两者让"整组全错"重新可学（保覆盖），剪枝的前提恰恰是它不可学（零梯度）——三者在未来方法空间里互斥，实验时注意同步开关。

## 9. per-task 双 trie：对齐任务的"支持集铁律"修复（2026-09-03 已实现）

**发现的错配**：trie（minionerec_trainer 构建）用**全长 sid**（碰撞 item 含 `<d_x>`），EOS 只挂在完整路径末端；
而 RLTitle2SidDataset 的对齐 target 只到**前 3 级**（data.py:830 显式丢弃 extra token："不承载任何语义"）。
→ 对碰撞前缀（td-last 变体 3643/13046 = 27.9% item、1304 桶），约束解码在 3 级后**只允许 `<d_x>`、EOS 被 -inf 屏蔽**：

- 目标序列 `<a><b><c>EOS` 永不在解码支持集 → 满分不可能（支持集铁律 violation）；
- first_diff_reward 落入 else 分支（rl.py:323，k==T<L 只可能是"对齐+碰撞+路由正确"）：前三 +1、**被迫生成的 `<d_x>` 位 -1**——模型被逼输出 reward 认为错的动作，且永远没机会输出对的动作；
- rule/ndcg 精确串匹配永不成立 → 该组恒 flag=False 整组 0（死组，非惩罚）；
- **(b) 变体最坏情形**：16 beam 全部正确路由到桶时，d 位列全 -1 命中 all-wrong 惩罚 → 整组唯一信号 = -λ（可能是 (b) run 浅层崩溃推手）；
- SFT 已教"3 级即停"（SFT 对齐同样截断 target），RL 在系统性摧毁它。

**修法（已实现）**：per-task 双 trie。
- `build_sid_hash_tries()`（模块级函数）：全长 trie（NTP/eval）+ 前缀 trie（每条 sid 截到 3 级再建树 → 任意 3 级前缀后 `\n→EOS`，`<d_x>` 不可达）。两树在 3 级前（含 unique 前缀 3 级节点）结构完全相同，仅碰撞前缀的 3 级节点分叉。
- 行级选择：rl.py 把 `train_data2.prompt2history.keys()`（对齐 prompt 集合）作为 `trie_prefix_prompts` 传入 trainer；`_prepare_inputs` 算 kinds——采样路径按行（行序与 batch 对齐）、**beam 路径按 dedup 后的 prompt 序号存（`prompts[::G]`）**（实际 RL run 都是 beam_search=True：rollout 先 dedup 每 prompt 取 1 行、generate 内部按 num_beams=G 展开且每 prompt 的 G 条 beam 连续 → processor 的 batch_id = prompt 序号，两路径均等长对齐）；dynamic_sampling 分支（非 beam）按 1.5×G 扩展；`prefix_allowed_tokens_fn(batch_id, …)` 按行选表，越界行（仅 eval test-beam 20 束 ≠ 16 布局时出现，eval 全 NTP）默认全长。
- **效果**：对齐样本答案回到支持集 → 满分/全对可达、标量奖励可命中、与 SFT 行为一致；d 的学习完全留给 NTP（其 target 全长、有交互上下文、reward 匹配）。
- **验证**：temp/test_dual_trie.py（400 抽样 + 全量扫描）：碰撞前缀 prefix trie 只允许 `\n→EOS`、`<d_>` 泄漏 = 0；unique 前缀两树一致；count-window 走树均可达 EOS；full trie d 层 ⊇ 桶内各 item 4th token。PASS。
- **预期观测**：新 run 的 alignment completion 应普遍停在 3 级（vs 旧 run 被迫 4 级）；对齐组 exact/全对率跳升；rule/ndcg 标量在该子集重新有信号。
- **局限**：eval 的 test-beam（num_beams=test_beam=20）行布局与 rollout 的 16 束不对齐——但 eval 数据集只有 NTP（kind 恒 0），回退全长 trie 无害；rl_gpr.py 未接入（默认全长不变）。
