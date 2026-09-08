# RL 黑话与 GRPO 实现详解

> 整理日期：2026-09-02 ｜ 对应代码：minionerec_trainer.py / rl.py
> 目的：RL 强化学习术语速查 + 本项目 GRPO 实现的关键澄清（与代码逐行对应）

---

## 1. 主线：从 REINFORCE 到 GRPO

```
REINFORCE（1992）→ Actor-Critic → PPO（2017）→ GRPO（DeepSeekMath 2024）→ 本项目
     ↑                ↑              ↑              ↑
  最朴素的         引入价值函数   引入 clip      去 critic、组内归一化
  策略梯度
```

---

## 2. 核心二分法：on-policy / off-policy

| 术语 | 含义 | 直觉 |
|---|---|---|
| **on-policy** | 训练数据**只能由当前策略产生**，用完即弃（一个样本只用于一次更新） | "我现在怎么走，就用现在走的这条路学" |
| **off-policy** | 数据可来自**旧策略/任意策略**，存入缓冲区反复使用 | "以前走过的路也能学"，翻旧账反复练 |

### 2.1 "数据"是什么？——RL 数据是策略**产生**的，不是输入的

- 监督学习（SFT）：数据 = 固定的 (输入 x, 标签 y) 对，**数据集给定**，可反复用
- RL：数据 = (上下文, 动作, 奖励) 三元组——**动作由策略自己生成，奖励依赖动作**

本项目每步的数据流：
```python
prompts = [x["prompt"] for x in inputs]           # ① 只有这个来自数据集
completion_ids = model.generate(...)              # ② 模型自己"产生"动作（rollout）
rewards = reward_func(prompts, completions)       # ③ 奖励依赖生成结果
```
θ 变 → 数据分布变 → 旧数据"过期"——这就是 on-policy 必须每步重新生成的原因（也是 RL 比 SFT 贵一个量级的原因）。

### 2.2 off-policy 如何"省生成开销"？——数据复用

- 生成（rollout）：自回归 **L 步**顺序前向（贵）
- 评估（rerun）：teacher forcing **1 次**并行前向（便宜）

PPO 的做法：1 次生成 → 存 buffer → 用同批数据练 **3~4 个 inner epoch**（每 epoch 只付 1 次评估前向）。

| | 生成次数 | 更新次数 | 总成本 |
|---|---|---|---|
| on-policy（GRPO） | 4 次 | 4 | 4L 份前向 |
| off-policy（PPO） | 1 次 | 4 | L + 3 份前向 |

**代价**：数据过期（distribution shift）→ importance ratio 偏离 1 → 方差爆炸 → **clip 兜底**（PPO 的 clip 就是为 off-policy 复用设计的）。GRPO 选单步 on-policy，ratio 恒=1，clip 结构性无效。

### 2.3 importance ratio 的 π_θ 怎么得到？——评估前向（也要前向，但便宜）

```python
ratio_t = π_θ(y_t) / π_θ_old(y_t) = exp(logp_new − logp_old)
```
分子 π_θ 每 epoch 重新前向算（θ 在变），但**这是"评估"不是"生成"**：

| | 生成（rollout） | 评估（rerun） |
|---|---|---|
| 输入 | prompt（部分序列） | **整条轨迹**（prompt+completion） |
| 方式 | 自回归 L 步（顺序依赖） | teacher forcing 1 次并行 |
| 成本 | L 次前向 | 1 次前向 |

### 2.4 teacher forcing 的 teacher 是谁？——旧策略的生成，不是真实 target

- SFT：teacher = 真实标签（数据集 y_true）
- RL 评估前向：teacher = **rollout 生成的序列**（y ~ π_θ_old）

原因：importance sampling 要求"评估分布 = 采样分布"：
```python
input_ids = torch.cat([prompt_ids, completion_ids], dim=1)   # 拼的是生成的，不是 targets
```
target 只在 reward 函数里当"判卷标准"（`completion == targets[i] → 1`），从不进入 logp 计算路径。

### 2.5 on-policy 下为什么还要评估前向？——生成"拿不到"干净的 logp

生成时每步确实算了 logits，但 `generate()` 不给你：
1. **beam 内部状态不可还原**：beam 每步的 logits 是"候选扩展"分数，经过排序/剪枝/合并后，最终序列对应的 per-token logp 需要 `output_scores + beam_indices` 复杂重组（EOS 截断、padding、beam 重排等边界情况）
2. **约束解码 mask 干扰**：生成时用的是 mask 后分布（非法 token = -inf），评估前向是原始分布

评估前向 = 拿最终序列重跑 1 次并行前向，`selective_log_softmax` 只取 completion 位置的 token logp——成本 ≈ 生成的 1/L，换来接口干净、口径统一、实现简单。

### 2.6 严格性问题：ratio 该用 mask 后分布，为什么不用？

| 量 | 严格该用 | 现状 |
|---|---|---|
| ratio（采样修正） | mask 后（采样时真实行为分布） | 原始 logp（不严格） |
| KL（分布距离） | 原始分布（策略真实差距） | 原始 logp（严格 ✓） |

- 就算收集了生成时的 mask 后 logp，**ref logps 的评估前向省不掉**（ref 没参与生成），且 KL/ratio 口径会分家
- mask 是**采样过程的工程干预**，不是策略本身的属性；KL 衡量模型学到的分布差距，本就该用原始分布
- 误差影响：合法 SID token 原始概率不低 + **advantage 组内归一化缓冲** → 影响可忽略
- 结论：评估前向是"1 次并行前向换接口干净 + 口径统一"的最优工程选择

---

## 3. 策略梯度内部构造

| 术语 | 含义 | 本项目 |
|---|---|---|
| **REINFORCE** | 最朴素策略梯度：∇logπ × R，用总回报当权重 | GRPO 的退化形式（ratio=1 时） |
| **baseline（基线）** | 回报减去常数，不改变期望梯度、只降方差 | 组内 16 条的均值 |
| **advantage（优势）** | A = R − baseline，比平均好多少 | `(rewards − mean)/(std + 1e-4)`，minionerec_trainer.py:972 |
| **critic（评论家）** | 学状态价值的网络，提供 baseline（Actor-Critic） | **无**——GRPO 用组统计代替，这是与 PPO 的最大区别 |
| **credit assignment（信用分配）** | 序列里每个 token 为结果负多少责 | 同一序列所有 token 共享同一个 A（粗糙近似） |
| **GAE（广义优势估计）** | PPO 的多步折现优势 | 无（GRPO 无时间折扣） |

---

## 4. 损失函数两部分（公式化）

```python
# minionerec_trainer.py:1060-1068
per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
ref_per_token_logps = inputs["ref_per_token_logps"]
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
advantages = inputs["advantages"]
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
per_token_loss = -(per_token_loss - self.beta * per_token_kl)
```

### 4.1 surrogate 项（策略梯度目标）

$$r_t = \frac{\pi_\theta(y_t)}{\pi_{\theta_{old}}(y_t)} = e^{\log\pi_\theta - \log\pi_{\theta_{old}}}, \qquad \text{surrogate}_t = r_t \cdot A$$

- **importance sampling**：序列是旧策略生成的，ratio 把期望修正到新策略分布
- A > 0 提升该 token 概率，A < 0 压低
- `detach()` 把分母视为常数（只对 π_θ 求导）

### 4.2 exp(logp − logp.detach()) 的梯度技巧（数值恒为 1）

- 数值上：logp − logp.detach() = 0 → exp(0) = **1**（on-policy 单步下 ratio 数值恒 1）
- 梯度上：链式法则逐层本地梯度 `A × e⁰ × 1 = A` → 对 logp 的梯度 = A → 最终 `A × ∇logπ_θ`
- 这就是 **REINFORCE 策略梯度**（A×∇logπ）——"数值 1、梯度 logp"的 autograd 包装

| 写法 | 数值 | 梯度 | 问题 |
|---|---|---|---|
| `1 * A` | A | 0 | 无学习信号 |
| `logp * A` | logp·A | A·∇logp | 数值量级被 logp（≈−10~−20）带偏 |
| `exp(logp−detach) * A` | 1·A | A·∇logp | ✅ 干净 |

### 4.3 KL 惩罚项（逐点代理形式）

$$\text{kl}_t = e^{d_t} - d_t - 1, \qquad d_t = \log\pi_{ref} - \log\pi_\theta$$

| 情形 | d | kl_t ≈ | 惩罚 |
|---|---|---|---|
| 模型概率 ≈ ref | 0 | d²/2 | 二次（小） |
| 模型概率 << ref（压没 token） | 大正 | **e^d 指数爆炸** | 极重——防概率归零 |
| 模型概率 >> ref（过度自信） | 大负 | 线性 | 轻 |

**不对称设计**：模型把 token 概率压到 0 的代价指数级（无 clip 时防坍缩的刹车）。

### 4.4 完整损失

$$\mathcal{L}(\theta) = -\frac{1}{B}\sum_i \frac{1}{N_i}\sum_{t \in y_i} \left[ r_{i,t}\, A_i - \beta \left(e^{d_{i,t}} - d_{i,t} - 1\right) \right]$$

- 负号：最大化奖励、最小化 KL
- mask：只统计 completion 部分
- A 是序列级标量，广播到序列内所有 token
- 与论文公式的差异：**无 clip**（min(r·A, clip(r)·A) 被省略）——单步 on-policy 下 ratio=1，clip 结构性无效

### 4.5 梯度推导（为什么 ∇ = A × ∇logπ）

计算图：`z=logπ → u=z−c → g=exp(u) → f=g×A`，链式法则：

| 算子 | 本地梯度 | 数值 |
|---|---|---|
| f = g × A | ∂f/∂g = A | A |
| g = exp(u) | ∂g/∂u = exp(u) | e⁰ = 1 |
| u = z − c | ∂u/∂z = 1，∂u/∂c = −1 | c 是 detach，∂c/∂θ = 0 |

相乘：∂f/∂z = A × 1 × 1 = A → 再经网络 ∂z/∂θ = ∇logπ → **∇f = A × ∇logπ**。

---

## 5. 奖励归一化（Group Relative 的核心）

**位置**：minionerec_trainer.py **966-972 行**（_prepare_inputs 内）

```python
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)   # 每组16条均值
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)     # 每组16条标准差
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)  # 组内 z-score
```

- 组 = 同一 prompt 的 16 条生成（RepeatRandomSampler 保证相邻）
- 1e-4 防除零（16 条 reward 全相同）
- 归一化的是**多 reward 加权求和后**的总奖励（963 行），不是每个 reward 单独归一化
- 归一化后每组 advantage 均值≈0、标准差≈1 → reward 函数绝对数值不影响梯度，只有组内相对高低起作用
- 同一序列所有 token 共享同一个 A

---

## 6. 奖励与约束（第四组黑话）

| 术语 | 含义 | 本项目 |
|---|---|---|
| **sparse reward** | 大多数动作无反馈，只有命中才给 | rule_reward（1/0） |
| **dense reward** | 每个动作都有连续反馈 | cf_reward（SASRec 打分） |
| **reward hacking** | 模型钻奖励漏洞，指标虚高真实质量差 | 论文：CF 奖励单独用就 hack 了 |
| **KL 惩罚 / ref model** | 约束策略别偏离参考模型 | per_token_kl + beta（KL 爆炸事故 = β 失灵） |
| **entropy bonus** | 鼓励策略保持随机（防坍缩） | 无（KL 项部分承担） |
| **reward model（RM）** | RLHF 经典 PPO 用的学偏好打分模型 | **不需要**——规则/排序/CF 直接算（RLVR） |

### 6.1 sync_ref_model（GRPOConfig 参数）

- **False**（TRL 默认）：ref = 训练开始的 SFT checkpoint（**固定**）→ KL 衡量**累计漂移**，随训练单调增长
- **True**（本项目）：ref = 上一个 optimizer step 的 policy（**滚动**）→ KL 衡量**步间更新量**，恒定很小（实测 0.1~1.6）
- 论文公式中 π_ref 指 SFT 起点（固定参照）——sync=True 改变了语义，发论文需注明
- 显存代价：ref 模型拷贝 +1.2GB/卡（两种都要）

---

## 7. 生成侧黑话

| 术语 | 含义 | 本项目 |
|---|---|---|
| **rollout** | 一次从 prompt 到生成的完整采样 | 每步 128 prompt × 16 beam |
| **trajectory / episode** | 一条完整动作序列 | 一条 beam 生成 |
| **exploration vs exploitation** | 探索新路径 vs 利用已知好的 | beam = exploitation；dynamic_sampling = exploration（论文对比过） |
| **temperature / top-k / top-p** | 采样随机性旋钮 | beam 路径 do_sample=False 后被忽略 |

### 7.1 四种解码组合

| 组合 | 解码方式 | 特点 |
|---|---|---|
| do_sample=False + num_beams=1 | 贪心 | 确定性，1 条 |
| do_sample=False + num_beams=16 | **纯束搜索** | 确定性，每次相同 16 条 |
| do_sample=True + num_beams=1 | 随机采样 | 每次不同 |
| do_sample=True + num_beams=16 | **采样束搜索** | 每 beam 按概率采样扩展，有随机性 |

**本项目**：训练 beam 路径已改 `do_sample=False`（2026-09-02 修改，与论文 3.4.1 确定性束一致）；采样束的机制 = 每 beam `torch.multinomial` 采样 2 个 token → 32 候选按累计 logprob 排序留 top-16（不是全局 top16）。

### 7.2 beam search without length normalization

- beam 评分 = 累计 log 概率（每 token logp ≤ 0，长序列天然分低）
- length normalization 是补偿手段（GNMT length penalty 等）
- "without" = **length_penalty=0.0**（本项目配置）→ 纯累计 logprob，不偏袒任何长度
- 推荐场景的意义：SID 只有 3-4 token，长度归一化无意义；让模型自主决定"继续 d 级还是 EOS"；训练测试口径一致

---

## 8. 关键代码位置速查

| 内容 | 位置 |
|---|---|
| reward 打分 + 加权求和 | minionerec_trainer.py 935-963 |
| **组内归一化 advantage** | minionerec_trainer.py 966-972 |
| ref logps 计算 | minionerec_trainer.py 897-906 |
| surrogate + KL 损失 | minionerec_trainer.py 1060-1068 |
| compute_loss 默认/dapo/gspo 分支 | minionerec_trainer.py 1069-1077 |
| ref 模型创建 + sync 回调 | minionerec_trainer.py 514-522 |
| 训练 beam generation_config | minionerec_trainer.py 489-505 |
| 测试 beam generation_config | minionerec_trainer.py 574-582 |
| 约束解码（hash 前缀表） | minionerec_trainer.py 529-592 |

---

## 9. 一句话总结

**本项目 = 单步 on-policy 的 GRPO**：每步生成 16 条 beam（确定性束）→ 组内 z-score 归一化得 advantage → `1×A` 的 surrogate（ratio 数值恒 1，梯度 = A·∇logπ）→ KL 拉回 ref（β=0.04）→ 无 critic、无 clip、无 RM——所有"黑话"都在这一条链路上有明确落点。
