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

## 10. 想法：只在前缀约束的候选集上算 logits（2026-09-14 讨论归档）

> **状态：未实现，待实验。** 优先级取决于 §10.4 的 `log m` 诊断结果。
> 数据来源：本轮 5090 单卡实测（`temp/probe_*`），不是估算。

### 10.1 事实基础（实测）

**logits 链路的精确定义**：从最后一层 hidden state 到 loss 之间的 `[B, L, V]` 张量群。

```
hidden_states [B,L,1024]  --lm_head-->  logits [B,L,152452]      ← 维度放大 149 倍
                              ↓ .float()  (loss_utils.py:55)
                              logits_fp32 [B,L,V]
                              ↓ cross_entropy
                              loss (标量)
```

`[B,L,V]` 逐项实测（N=1024 隔离测量，字节/元素）：

| 项 | dtype | 字节/元素 | 出处 |
|---|---|---|---|
| lm_head 输出 | bf16 | 2 | `modeling_qwen3.py` |
| `.float()` 上采样副本 | fp32 | 4 | `loss_utils.py:55` |
| cross_entropy 前向内部 | fp32 | 4 | `log_softmax` 输出 |
| 反向梯度 ×2 | fp32 | 8 | autograd |
| **合计** | | **18** | |

**换算**：`18 × 152452 = 2.617 MiB/token`（与 V 无关的常数）。

配合 28 层激活值 **1.96 MiB/token**（实测：同 N 下 L 取 128/256/512 激活显存完全相同 → 注意力 L² 项在 cutoff=512 量级可忽略）：

$$\text{峰值} \approx \underbrace{1.9\text{ GiB}}_{\text{参数+优化器}} + N \times 4.8\text{ MiB}, \quad N = \text{micro} \times \text{批内实际 seq}$$

（四条预测与历史探针实测全部吻合，含"micro=32 在 seq=256 OOM"。）

**关键事实：约束是在 logits 算完之后才施加的。**

`LogitProcessor.py:73`：`mask[...] = 0; scores = scores + mask` —— **全词表 logits 已经算完，mask 只改变"选谁"，不省"算谁"。**

序列维度**已经省了**：`minionerec_trainer.py:1316` `logits_to_keep = completion_ids.size(1)`，`modeling_qwen3.py:493` `slice(-logits_to_keep, None)` —— lm_head 只在 completion 位置上跑。

**词表维度全算** —— 这就是本想法针对的。

### 10.2 想法

RL 阶段输出空间被 trie 限制在 **|S| = 783**（256 `<a_*>` + 256 `<b_*>` + 256 `<c_*>` + 15 `<d_*>`），`V/|S| = 194.7`。

把 lm_head 换成 S 上的切片：

```python
logits = F.linear(hidden[:, slice_indices, :], lm_head.weight[sid_ids])   # [B, L, 783]
```

### 10.3 收益

| | 现在 | 切片后 |
|---|---|---|
| 训练 logits 显存 | 2.617 MiB/token | **0.013 MiB/token**（−99.5%） |
| 生成 logits（`num_beams=16`） | `16 × 152452` | `16 × 783` |
| GRPO 分布一致性 | 近似（见 10.4） | **恒等式** |

### 10.4 分布一致性 —— **不是"ratio 问题"**（重要澄清）

⚠️ **本 trainer 没有 importance ratio。** `minionerec_trainer.py:1330`：

```python
# surrogate；on-policy 下生成和更新之间 θ 不变（不需要修正系数） → ratio 数值恒为 1；不加 clip
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * per_token_adv
```

`detach()` 后因子恒为 1，梯度即 `∇log π_θ(a)` —— **纯 REINFORCE 形态**（GRPO 原始论文形态，非 TRL 加 clip 的版本）。`grep old_per_token_logps` 零命中，无从构造 ratio。

**但"没有 ratio"≠"采样分布无关"**：REINFORCE 估计量 `∇log π(a)·A(a)` 的无偏性**前提就是 `a ~ π_θ`**。采样分布不是通过显式 ratio 进入的，而是通过**"哪些 action 进了 batch"**。

实际情况：采样自 `π_S`（trie 受限），估计的是 `E_{a~π_V}[·]`。两者只差归一化：

$$\pi_S(a) = \frac{\pi_V(a)}{m}, \qquad m(c) = \sum_{v \in S(c)} \pi_V(v \mid c)$$

$$\nabla\log\pi_S(a) = \nabla\log\pi_V(a) - \nabla\log m$$

**`m` = 模型在"该位置合法 token"上的概率质量。** SFT 之后应接近 1 → 偏差可忽略；但**策略漂移时 `m` 会掉，偏差随之放大**——恰是 KL 快兜不住的时候（本实现无 clip，`beta=0.04` 的 KL 是唯一护栏）。

> 代码注释已自知此事（`minionerec_trainer.py:1315`："原始分布，无 trie 约束和 mask 后的归一化"）。

**可直接监控**（`logp_V(a) − logp_S(a) = LSE_S − LSE_V = log m`）：

```python
log_m = (lse_allowed - lse_full).mean()      # 两次 logsumexp，成本可忽略
self._metrics["log_m"].append(log_m.item())
```

| `log m` | 含义 | 行动 |
|---|---|---|
| > −0.01（m>0.99） | 近似无害 | 切片只为显存/算力 |
| −0.05 ~ −0.01 | 轻微偏离 | 观察趋势 |
| < −0.05（m<0.95） | **策略在往非法 token 分质量** | reward hacking 早期信号；切片升级为正确性修复 |

**这个指标比 KL 更直接**：KL 衡量"离 ref 多远"，`log m` 衡量"离合法输出空间多远"。

### 10.5 实施要点与三个坑

**① 允许集合逐位置变化 → 切超集即可**
`prefix_allowed_tokens_fn` 按 trie 节点返回当前位置的合法 token（第 0 位只有 `<a_*>`，第 1 位只有所选 `<a_x>` 的孩子）。固定切 783 的超集已吃掉 99.5% 收益，无需按层细分。

**② token id 下标错位 —— 唯一的静默失败点** ⚠️
切片后 logits 下标是 `0..782`，但 `_get_per_token_logps` 直接拿真实 token id gather：

```python
return selective_log_softmax(logits, input_ids)     # minionerec_trainer.py:750
```

**必须**先建映射 `id2pos = {tid: i for i, tid in enumerate(sid_ids)}` 并加断言。**映射错不报错，只算出错的 logp、训出错策略。**

**③ trie token 覆盖校验（把静默错误变响亮）**
启动时断言 trie 中可能出现的**所有** token ⊆ `sid_ids`：

```python
allowed = set().union(*self.hash_dict_full.values(), *self.hash_dict_prefix.values())
missing = allowed - set(sid_ids)
assert not missing, f"切片漏了 {len(missing)} 个 trie token：{sorted(missing)[:5]}"
```

**④ lm_head 梯度只流向 S 的行**
`W[sid_ids]` 之外的行（含 tie 的 embedding 对应行）在 RL 阶段完全冻结。可接受（RL 只该调 SID 行为），但要意识到 RL 之后模型对非 SID 的输出分布不再变化。

### 10.6 建议顺序

1. **先加 `log m` 指标**（~10 行，不动核心逻辑），跑一段 RL 看数值
2. `log m ≈ 0` → 切片定位为**显存/算力优化**，按显存压力排优先级
3. `log m` 明显非 0 → 切片升级为**正确性修复**，优先级拉满

### 10.7 相关方向（记录备查）

- **SFT 侧不能照搬**：NTP 任务输出全是 SID（可切），但 sid↔title 互译输出任意文本（不可切）；且 SFT 一次 forward 对整条序列算 loss、三任务混批，无法按位置区分词表。要做需拆任务分别 forward，是另一个量级的改动。
- **同族技术**：Liger-Kernel `FusedLinearCrossEntropy` / `CutCrossEntropy` —— 都是"不物化 `[B,L,V]`"。对 CE（每位置单 target）有效；本项目的 RL 也是每位置单 target（gathered logp），故同样适用，但**它们不解决"受限候选集"这一层**，只省物化。
