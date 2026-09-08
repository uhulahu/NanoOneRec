# minionerec_trainer.py 修改分析（ReReTrainer vs TRL GRPOTrainer）

> 整理日期：2026-09-01 ｜ 对照版本：TRL 1.12.0（环境已装）/ 旧版 TRL 0.10-0.11（代码来源）
> 结论先行：**本文件是 TRL 旧版 `grpo_trainer.py` 的深度修改副本**——GRPO 算法骨架（rollout → reward → 组内归一化 → KL 约束）原样保留，作者在此基础上加入了推荐场景专属的：SID 约束解码、beam search、训练中评估、dynamic sampling、DAPO/GSPO 变体。

---

## 1. 文件结构总览（行号 → 内容 → 来源）

| 行号 | 内容 | 来源 |
|---|---|---|
| 15-49 | import（TRL 内部工具、transformers、accelerate、vLLM） | TRL 原样（连带 import 未清理） |
| 50-97 | `SyncRefModelCallback`：ref 模型权重同步回调 | TRL 原样 |
| 100-118 | `RepeatRandomSampler`：每组重复 G 次的采样器 | TRL 的 `RepeatSampler` **简化版**（去掉 mini_repeat_count/batch_size，只留 repeat_count+seed） |
| 121-209 | `ReReTrainer(Trainer)` docstring + `_tag_names = ["trl", "grpo"]` | TRL docstring **逐字复制**（Example 里还残留 `GRPOTrainer(...)` 示例） |
| 211-368 | `__init__`：模型/ref/优化器/回调初始化 + **新增推荐参数** | TRL 骨架 + 修改 |
| 369-412 | 梯度累积整除检查（batch size 必须是 G 的倍数） | TRL 原样 |
| 413-477 | 进程种子设置、vLLM 初始化 | TRL 原样（本项目不用 vLLM，属残留） |
| 478-505 | **`generation_config` 构建：beam search 分支** | **修改**（TRL 只有采样配置） |
| 506-522 | loss 缩放设置、ref model 分发、sync_ref_model 回调注册 | TRL 原样 |
| 524-527 | reward 模型分发（本项目 reward 全为函数，此分支闲置） | TRL 原样 |
| 529-582 | **SID hash_dict 前缀表构建 + `test_generation_config`（beam 评估用）** | **新增** |
| 584-592 | **`get_hash` / `prefix_allowed_tokens_fn`：SID 约束解码** | **新增** |
| 594-608 | `_set_signature_columns_if_needed` | TRL 原样 |
| 610-624 | `_get_train_sampler` / `_get_eval_sampler`（RepeatRandomSampler） | TRL 骨架（换采样器类） |
| 627-636 | `_get_per_token_logps`（importance ratio 基础） | TRL 原样（含 TRL 的注释） |
| 638-664 | `_move_model_to_vllm` | TRL 原样（残留） |
| 665-682 | `_prepare_inputs` 开头：prompt 取数、tokenize | TRL 骨架 |
| 669-675 | **`prompt2history` / `history2target` 反查 target** | **新增** |
| 689-700 | **`ConstrainedLogitsProcessor` 约束解码器 + logits_processor 组装** | **新增** |
| 702-881 | **rollout 生成：vLLM / beam / dynamic sampling / add_gt 四条路径** | **修改+新增** |
| 884-921 | EOS mask、ref logps 计算、decode | TRL 原样 |
| 922-933 | **cate_diversity / token_diversity 多样性指标** | **新增** |
| 935-981 | reward 函数调用 + gather + **组内归一化 advantage** | TRL 原样（关键注释也在） |
| 986-998 | reward 指标日志 | TRL 骨架 |
| 1000-1003 | **test_during_training：训练中 HR/NDCG 进日志** | **新增** |
| 1024-1032 | `_prepare_inputs` 返回（含 advantages / sliced_rewards） | TRL 骨架 |
| 1035-1073 | **`compute_loss`：GRPO 目标 + dapo/gspo 分支（无 clip）** | **修改** |
| 1075-1096 | `prediction_step` / `log`（_metrics 聚合） | TRL 原样 |
| 1098+ | `create_model_card` | TRL 原样（残留） |

---

## 2. 新增部分详解（推荐场景专属）

### 2.1 SID 约束解码（529-592、689-700 行）

**作用**：RL 的 action space 是 13,046 个 SID（如 `<a_5><b_23><c_55>`），生成时若不加约束，模型会吐出无意义的 token 序列，reward 全部失效。这套机制保证**每个生成的 token 都必须是某个合法 SID 的前缀**。

**机制**（与 evaluate.py 同一套，RL 训练中复用了）：

```
① __init__ 529-572：读 info_file（semantic_id \t title \t item_id）
   → 每个 SID 文本拼成 "### Response:\n<a_5><b_23><c_55>\n"
   → tokenize 得到 ID 序列，末尾补 EOS
   → 按 token 位置构建 hash_dict：
       hash_key = ID[:3]            （第 3 个 token 起，prefix_index=3）
       hash_key = ID[3:i]           （后续每级前缀）
       hash_dict[key] = 下一合法 token 的集合
② 689-698：每次 _prepare_inputs 新建 ConstrainedLogitsProcessor(ccc)
   → 每个生成步：查当前生成前缀的 hash_key → 只允许 hash_dict 里的 token
   → 前缀非法时强制 EOS（LogitProcessor.py 中的 fallback）
```

**注意点**：
- `prefix_index=3` 是硬编码假设（`### Response:\n` 模板下第 3 个 token 开始是 SID 内容），依赖模板不变；
- `ccc` 每次 `_prepare_inputs` 重建，`count` 从 0 开始，与 beam 步数对齐；
- 训练 rollout 用 `TemperatureLogitsWarper + ccc`（699 行），测试用纯 `ccc`（700 行）——测试不做温度采样。

### 2.2 beam search rollout（478-495、574-582、779-792 行）

**作用**：论文 3.4.1 的核心决策——用约束 beam search 替代重采样，保证每组 16 条生成**互不重复且全部合法**（采样 16 次大概率重复，浪费算力且负样本多样性差）。

```
self.generation_config = GenerationConfig(
    num_beams=num_generations,            # 16
    num_return_sequences=num_generations, # 16
    do_sample=True,                        # ← 见"潜在问题 #1"
    temperature=self.temperature,          # 默认 1.0
    ...)
```

- beam 宽度 = `num_generations`，一组 16 条 = 一个 prompt 的 16 个 beam；
- `test_generation_config` 额外带 `num_beams=test_beam`（默认 20），用于训练中评估。

### 2.3 test_during_training（736-776、1000-1003 行）

**作用**：GRPO 训练中穿插 beam 评估，日志里实时输出 HR@k / NDCG@k。

```
每步：取每组第一条 prompt（i % num_generations == 0）→ 用 test_generation_config（beam=test_beam）生成
   → 按命中位置累计 HR/NDCG（hr[index] += 1 / log2(j+2)）
   → 全部组完成后除以组数 → 写进 _metrics["HR@k"] / ["NDCG@k"]
```

- 注意：此处 `dedup_prompt` 的命名是**组去重**（每组取 1 条代表），与 data.py 里 `dedup` 字段（平凡样本标记）无关；
- 与独立评估脚本 evaluate.py 的差异：这里是简化版（无 hash_dict 的完整约束以外仍有约束处理器），指标只能看趋势，最终指标仍以 evaluate.py 为准。

### 2.4 dynamic_sampling（796-851 行）

**作用**：论文 3.4.1 对比过的另一种采样策略（最终未采用，但代码保留）。

```
① 每个 prompt 过采样 1.5×G = 24 条
② select_completion(completions, target)：
    - target 出现多少次就保留多少条（最多 G 条）
    - 其余槽位按出现频次降序填非 target 候选
   → 保证"组内必有 target + 多样性最大化"
```

与论文描述一致："over-sample, then pick a subset that (i) must include the ground-truth item and (ii) maximizes internal diversity"。**缺点**（论文也指出）：多 50% 的前向开销，且随训练推进多样性退化——所以默认 `dynamic_sampling=False`。

### 2.5 add_gt（858-875 行）

**作用**：把 ground-truth SID 直接**混入**某组的生成序列中。

```
repeat = len(prompts) // num_categories   # 按 target 类别数分组
每组最后一个位置（(i+1)%repeat==0）：用 target_ids 替换掉该位置的生成
```

保证每组至少出现一次正例（配合 `num_categories` 的设计：组内 target 各不相同）。默认 `add_gt=False`。

### 2.6 dapo / gspo 变体（1056-1062 行）

`compute_loss` 的三种计算路径：

| 模式 | 公式 | 语义 |
|---|---|---|
| 默认（GRPO） | `(per_token_loss * mask).sum(1)/mask.sum(1)` 再 mean | per-token ratio × advantage，按样本平均 |
| `dapo=True` | 全局 `sum/sum` | DAPO 风格：所有 token 一起平均（长回答 token 权重更高） |
| `gspo=True` | `s_score = exp(mean(ratio)); loss = -(s_score·adv - β·seq_kl).mean()` | GSPO：先按序列聚合 ratio 再算 importance，KL 也按序列平均 |

### 2.7 多样性指标（922-933 行）

```
cate_diversity = 每组内唯一生成数 / G 的平均     → 论文 Div 指标的直接实现
token_diversity = 唯一 token 数 / 总 token 数     → 生成 token 级多样性
```

写进 `_metrics`，可在 wandb 里监控（对应论文 "diversity" 讨论）。

### 2.8 prompt2history 反查（669-675 行）

```
targets = [self.history2target[self.prompt2history[x["prompt"]]] for x in inputs]
```

reward 函数拿不到 ground truth，靠 rl.py 里构建的两级查找表反查。**注意**：只有 `add_gt` / `test_during_training` / `dynamic_sampling` 任一开启时才计算（669 行），纯采样路径不反查。

---

## 3. 修改部分详解

### 3.1 类与 `__init__` 参数（121、211-368）

- 类名 `GRPOTrainer` → `ReReTrainer`，继承保持 `transformers.Trainer`（TRL 0.10-0.11 的继承方式；TRL 1.12 已改为 `_BaseTrainer`，API 不兼容，这也是作者不能用现装 TRL 的原因）；
- 新增参数（rl.py 侧对应）：`base_model`（约束解码的 tokenizer/模板来源）、`prompt2history`/`history2target`（reward 反查表）、`info_file`（SID 前缀表数据源）、`beam_search`、`test_during_training`、`test_beam`、`dynamic_sampling`、`add_gt`、`dapo`、`gspo`、`length_penalty`。

### 3.2 generation_config：TRL 只有采样，这里加了 beam 分支（478-505）

```python
if self.beam_search:
    GenerationConfig(num_beams=G, num_return_sequences=G, do_sample=True, temperature=...)
else:
    GenerationConfig(do_sample=True, temperature=...)
```

注意**两个分支都保留了 `do_sample=True`**（TRL 原版只有采样分支，beam 分支是新增的，但 `do_sample=True` 被原样带了过来——见潜在问题 #1）。

### 3.3 compute_loss：无 clip 的 GRPO 目标（1049-1064）

对照论文公式：
```
论文：min( w·A, clip(w, 1-ε, 1+ε)·A ) - β·KL      ← 有 PPO 式 clip
代码：exp(logp - logp.detach()) * A - β·KL         ← 无 clip（DPO 式简化）
```
`per_token_kl = exp(ref-new) - (ref-new) - 1` 与 TRL 逐字一致（TRL 的 KL 估计式）。

### 3.4 RepeatRandomSampler（100-118）

TRL 原版支持 `mini_repeat_count`/`batch_size`/`shuffle` 精细控制；本项目简化为 `randperm + repeat_count`：每 epoch 洗牌一次，每个索引连续重复 G 次。效果：同一 prompt 的 G 条生成**在同一进程内相邻**，便于组内归一化（多卡时跨进程的组靠 `gather` 汇总——958-960 行）。

---

## 4. 保留原样部分（可直接对照 TRL 文档）

- `_get_per_token_logps`（627）：per-token logp 提取（`logits_to_keep` 技巧）
- advantage 组内归一化（966-972）：`(r - mean) / (std + 1e-4)`，每组 G 条
- EOS 截断 mask（884-889）
- ref logps 计算（897-906，`sync_ref_model` 时用 ref 模型，否则 disable_adapter 后算）
- `_set_signature_columns_if_needed` / `log` / `prediction_step` / `_move_model_to_vllm`
- 梯度累积整除校验（393-411）：`global_batch_size % G == 0` 必须成立

---

## 5. 潜在问题清单（跑 RL 前建议核对）

1. **`use_model_defaults` bug（与 evaluate.py 同款）——已于 2026-09-01 修复**：transformers 4.57 默认 `use_model_defaults=True`，模型 config 的默认值（Qwen3 的 `do_sample=True, temperature=0.6`）会覆盖显式 GenerationConfig——`test_generation_config` 里 `do_sample=False` 会失效变成采样，训练中评估失真（evaluate.py 之前踩过这个坑，HR@1 差 21.8%）。已给 753/789/808/853 行四处 `generate()` 调用补 `use_model_defaults=False`（详见 MONITORING_LOG.md 2026-09-01 条目）。
2. **beam + `do_sample=True` 的语义**：beam_search 分支显式 `do_sample=True` + `temperature=1.0`（默认）→ 采样束搜索在 T=1 时≈贪心束；若 `temperature` 被设成 ≠1.0 会变成温度采样束（可能有意，但要知道）。
3. **无 clip**：默认 GRPO 目标没有 PPO 式 clip，reward 尺度突变时梯度可能震荡（论文公式是带 clip 的）。
4. **`ccc` 的 `prefix_index=3` 硬编码**：依赖 `### Response:\n` 模板；换 prompt 模板必须同步改。
5. **`cf_reward` 与数据集格式耦合**：`split("::")` 要求 `prompt2history` 存 SID 序列，混入 RLTitle2Sid 数据会 KeyError（见 rl.py 分析）。
6. **vLLM 代码残留**：`_move_model_to_vllm`、`self.llm` 等路径在 `use_vllm=True` 时才会触发，本项目不用但注意别误开。
7. **`num_generations` 整除校验**：4 卡 × `per_device_train_batch_size` 必须是 16 的倍数（如 4×4=16 ✓；4×2=8 ✗ 会直接报错）。
8. **reward 恒为负**：ndcg_rule_reward 设计使平均 reward 为负，wandb 曲线是负值属正常。

---

## 6. 与论文 3.4 节的对应关系

| 论文内容 | 代码实现 |
|---|---|
| 3.4.1 Beam Search（默认采样器） | 478-495 行 generation_config + 779-792 行 generate 路径 |
| 3.4.1 Dynamic Sampling（对比项） | 796-851 行 |
| 3.4.1 Diversity 指标 Div | 922-924 行 `cate_diversity` |
| 3.4.2 硬二值奖励 R_rule | rl.py `rule_reward` |
| 3.4.2 rank-aware 奖励 R_rank | rl.py `ndcg_rule_reward` |
| 3.4.2 collaborative reward | rl.py `cf_reward`（SASRec） |
| 3.4.2 最终奖励 = R_rule + R_rank | rl.py `reward_type="ranking"` |
| 组内归一化 advantage | 966-972 行 |
| KL 约束（β） | 1049 行 + `beta` 参数 |
