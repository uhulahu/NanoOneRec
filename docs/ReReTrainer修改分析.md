# minionerec_trainer.py 修改分析（ReReTrainer vs TRL GRPOTrainer）

> 整理日期：2026-09-01 ｜ **修订：2026-09-18（行号引用全部改为符号名）**
> 对照版本：TRL 1.12.0（环境已装）/ 旧版 TRL 0.10-0.11（代码来源）
> 结论先行：**本文件是 TRL 旧版 `grpo_trainer.py` 的深度修改副本**——GRPO 算法骨架（rollout → reward → 组内归一化 → KL 约束）原样保留，作者在此基础上加入了推荐场景专属的：SID 约束解码、beam search、训练中评估、dynamic sampling、DAPO/GSPO 变体。

> ⚠️ **为什么不再用行号**：本文初版通篇标 `minionerec_trainer.py:529-582` 这类行号。2026-09-02~09-08 的双 trie / token 级 advantage / detach / DeepSpeed 改造让该文件从 ~1100 行涨到 **1433 行**，**全部行号一次性错位 ~150 行**，文档随之报废。现改为**引用符号名**（函数/类/变量），不随行数变化。
> 需要定位时用 `grep -n '符号名' minionerec_trainer.py`。

---

## 1. 文件结构总览（符号 → 内容 → 来源）

| 符号 | 内容 | 来源 |
|---|---|---|
| 文件头 import 段 | TRL 内部工具、transformers、accelerate、vLLM | TRL 原样（连带 import 未清理） |
| `from trl import SyncRefModelCallback` | ref 模型权重同步回调 | **TRL 原样（直接 import，不再保留副本）**——初版此处是本地副本，后已改回 import |
| `RepeatRandomSampler` | 每组重复 G 次的采样器 | TRL 的 `RepeatSampler` **简化版**（去掉 mini_repeat_count/batch_size，只留 repeat_count+seed） |
| `MemTrackerCallback` | 周期性显存打点（`[MEM]`） | **新增**（非 TRL） |
| `ArchiveCheckpointCallback` | 指定 step 的 checkpoint 搬运归档 | **新增**（非 TRL）。⚠️ 用 `shutil.move`（**搬运不是复制**）→ 归档后原 run 目录里就没有那一档，见 `PROGRESS.md` 2026-09-08 节的资产依赖提醒 |
| `build_sid_hash_tries(info_file, base_model)` | **SID 前缀表构建（per-task 双 trie）+ `prefix_index` + `test_generation_config`** | **新增** |
| `ReReTrainer` docstring + `_tag_names` | | TRL docstring **逐字复制**（Example 里还残留 `GRPOTrainer(...)` 示例） |
| `ReReTrainer.__init__` | 模型/ref/优化器/回调初始化 + **新增推荐参数**；含梯度累积整除检查、进程种子、vLLM 初始化、两个 `generation_config` 分支、loss 缩放、ref model 分发 | TRL 骨架 + 修改 |
| `ReReTrainer.get_hash` / `.prefix_allowed_tokens_fn` | **SID 约束解码** | **新增** |
| `ReReTrainer._set_signature_columns_if_needed` | | TRL 原样 |
| `ReReTrainer._get_train_sampler` / `._get_eval_sampler` | RepeatRandomSampler | TRL 骨架（换采样器类） |
| `ReReTrainer._get_per_token_logps` | importance ratio 基础 | TRL 原样（含 TRL 的注释） |
| `ReReTrainer._move_model_to_vllm` | | TRL 原样（残留） |
| `ReReTrainer._prepare_inputs` | **整个 rollout + advantage 段**：prompt 取数 tokenize → `prompt2history` 反查 → `ConstrainedLogitsProcessor` 组装 → rollout 四条路径 → EOS mask / ref logps / decode → 多样性指标 → reward 调用 + gather + 组内归一化 → reward 日志 → `test_during_training` → 返回 | TRL 骨架 + 大量新增 |
| `ReReTrainer._masked_column_advantages` | 组内归一化 + (b)/(c) 的列级改造 | **新增**（2026-09-03） |
| `ReReTrainer._compute_token_advantages` | token 级 first-diff advantage | **新增**（2026-09-03） |
| `ReReTrainer.compute_loss` | **GRPO 目标 + dapo/gspo 分支（无 clip）** | **修改** |
| `ReReTrainer.prediction_step` / `.log` | `_metrics` 聚合 | TRL 原样 |
| `ReReTrainer.create_model_card` | | TRL 原样（残留） |

---

## 2. 新增部分详解（推荐场景专属）

### 2.1 SID 约束解码（`build_sid_hash_tries` + `get_hash` / `prefix_allowed_tokens_fn`）

**作用**：RL 的 action space 是 13,046 个 SID（如 `<a_5><b_23><c_55>`），生成时若不加约束，模型会吐出无意义的 token 序列，reward 全部失效。这套机制保证**每个生成的 token 都必须是某个合法 SID 的前缀**。

**机制**（与 evaluate.py 同一套，RL 训练中复用了）：

```
① build_sid_hash_tries：读 info_file（semantic_id \t title \t item_id）
   → 每个 SID 文本拼成 "### Response:\n<a_5><b_23><c_55>\n"
   → tokenize 得到 ID 序列，末尾补 EOS
   → 按 token 位置构建 hash_dict：
       hash_key = ID[:prefix_index]   （第 prefix_index 个 token 起是 SID 内容）
       hash_key = ID[prefix_index:i]  （后续每级前缀）
       hash_dict[key] = 下一合法 token 的集合
② _prepare_inputs 中每次新建 ConstrainedLogitsProcessor(ccc)
   → 每个生成步：查当前生成前缀的 hash_key → 只允许 hash_dict 里的 token
   → 前缀非法时强制 EOS（LogitProcessor.py 中的 fallback）
```

**⚠️ 双 trie（2026-09-03 新增，初版文档未覆盖）**：返回 `(hash_dict_full, hash_dict_prefix, max_sid)` 三件套。

- `hash_dict_full`：**全长 trie**（NTP 任务用）。EOS 只挂在完整路径末端——碰撞前缀在 3 级后只允许续 `<d_x>`。
- `hash_dict_prefix`：**前缀 trie**（对齐任务 title/desc→sid 用）。每条 sid 先截断到前 3 级再建树 → 任意 3 级前缀后即可停（`\n`→EOS），`<d_*>` 不可达。

**动机**：全长 trie 下对齐样本的 3 级答案永远不在解码支持集，被迫生成的 `<d_x>` 被 first-diff 误判 −1（**支持集铁律 violation**）；前缀 trie 让答案回到支持集。两棵树在 3 级之前结构完全相同，仅碰撞前缀的 3 级节点分叉。选择哪棵树由 `trie_prefix_prompts` 按行判定（详见 `docs/RL_IDEAS.md` §9）。

**注意点**：
- `prefix_index` **已不是硬编码**：`build_sid_hash_tries` 内按 `base_model.lower().find("gpt2")` 条件分支 → gpt2 取 4、其余取 3。**换 prompt 模板仍需同步改这个判据。**
- `ccc` 每次 `_prepare_inputs` 重建，`count` 从 0 开始，与 beam 步数对齐；
- 训练 rollout 用 `TemperatureLogitsWarper + ccc`，测试用纯 `ccc`——测试不做温度采样。

### 2.2 beam search rollout（`__init__` 的 beam 分支 + `_prepare_inputs` 的 beam 路径）

**作用**：论文 3.4.1 的核心决策——用约束 beam search 替代重采样，保证每组 16 条生成**互不重复且全部合法**（采样 16 次大概率重复，浪费算力且负样本多样性差）。

```python
self.generation_config = GenerationConfig(
    num_beams=num_generations,            # 16
    num_return_sequences=num_generations, # 16
    do_sample=False,                      # ← 2026-09-02 由 True 改为 False
    temperature=self.temperature,
    ...)
```

- beam 宽度 = `num_generations`，一组 16 条 = 一个 prompt 的 16 个 beam；
- **`do_sample=False`**：确定性束搜索（论文 3.4.1 口径），保证同 prompt 每次 rollout 得到相同的 16 条 beam，训练信号稳定；且与 `test_generation_config` 一致。原为 `True`（采样束），随机性使训练信号不稳定 —— 见 §5 已解决问题 #2。
- `test_generation_config` 额外带 `num_beams=test_beam`（默认 20），用于训练中评估。
- ⚠️ **else 分支（非 beam）仍是 `do_sample=True`**，那是纯采样路径，本项目默认不走。

### 2.3 test_during_training（`_prepare_inputs` 内）

**作用**：GRPO 训练中穿插 beam 评估，日志里实时输出 HR@k / NDCG@k。

```
每步：取每组第一条 prompt（i % num_generations == 0）→ 用 test_generation_config（beam=test_beam）生成
   → 按命中位置累计 HR/NDCG（hr[index] += 1 / log2(j+2)）
   → 全部组完成后除以组数 → 写进 _metrics["HR@k"] / ["NDCG@k"]
```

- 注意：此处 `dedup_prompt` 的命名是**组去重**（每组取 1 条代表），与 data.py 里 `dedup` 字段（平凡样本标记）无关；
- 与独立评估脚本 evaluate.py 的差异：这里是简化版，指标只能看趋势，最终指标仍以 evaluate.py 为准。

### 2.4 dynamic_sampling（`_prepare_inputs` 内）

**作用**：论文 3.4.1 对比过的另一种采样策略（最终未采用，但代码保留）。

```
① 每个 prompt 过采样 1.5×G = 24 条
② select_completion(completions, target)：
    - target 出现多少次就保留多少条（最多 G 条）
    - 其余槽位按出现频次降序填非 target 候选
   → 保证"组内必有 target + 多样性最大化"
```

与论文描述一致："over-sample, then pick a subset that (i) must include the ground-truth item and (ii) maximizes internal diversity"。**缺点**（论文也指出）：多 50% 的前向开销，且随训练推进多样性退化——所以默认 `dynamic_sampling=False`。

### 2.5 add_gt（`_prepare_inputs` 内）

**作用**：把 ground-truth SID 直接**混入**某组的生成序列中。

```
repeat = len(prompts) // num_categories   # 按 target 类别数分组
每组最后一个位置（(i+1)%repeat==0）：用 target_ids 替换掉该位置的生成
```

保证每组至少出现一次正例（配合 `num_categories` 的设计：组内 target 各不相同）。默认 `add_gt=False`。

### 2.6 token 级 advantage（`_compute_token_advantages` + `_masked_column_advantages`，2026-09-03 新增）

初版文档未覆盖。序列级标量奖励无法定位"错在哪一级"，这两组函数把 advantage 下沉到 token 级，并支持两种列级改造：

| 模式 | 接口 | 语义 | 实验结论 |
|---|---|---|---|
| 默认（group） | — | 组内 z-score，每组 G 条 | — |
| `--all_wrong_penalty λ`（想法 b） | `_masked_column_advantages(mode="group", all_wrong_penalty=λ)` | 对"整组该层全错"的列附加 −λ（不进 z-score），补上被 z-score 抹掉的病理信息 | ⚠️ **假说证伪一半**：非但没止住浅层收缩反而加速，但条件深层最强、HR@1 最佳 |
| `--token_norm column`（想法 c） | `_masked_column_advantages(mode="column")` | 列级 z-score 改跨组（batch/全局）标准化，让"全错是全局性的坏"重新可学 | ✅ **假说证实**：保住浅层覆盖，但条件深层被稀释 |

单测 `temp/verify_token_adv.py`；完整数据与 trade-off 面见 `docs/RL_IDEAS.md` §3/§4 与 `PROGRESS.md` 2026-09-03「想法 (b)(c) 实验归档」。

### 2.7 dapo / gspo 变体（`compute_loss` 内）

`compute_loss` 的三种计算路径：

| 模式 | 公式 | 语义 |
|---|---|---|
| 默认（GRPO） | `(per_token_loss * mask).sum(1)/mask.sum(1)` 再 mean | per-token ratio × advantage，按样本平均 |
| `dapo=True` | 全局 `sum/sum` | DAPO 风格：所有 token 一起平均（长回答 token 权重更高） |
| `gspo=True` | `s_score = exp(mean(ratio)); loss = -(s_score·adv - β·seq_kl).mean()` | GSPO：先按序列聚合 ratio 再算 importance，KL 也按序列平均 |

### 2.8 多样性指标（`_prepare_inputs` 内）

```
cate_diversity = 每组内唯一生成数 / G 的平均     → 论文 Div 指标的直接实现
token_diversity = 唯一 token 数 / 总 token 数     → 生成 token 级多样性
```

写进 `_metrics`，可在 wandb 里监控（对应论文 "diversity" 讨论）。

### 2.9 prompt2history 反查（`_prepare_inputs` 开头）

```
targets = [self.history2target[self.prompt2history[x["prompt"]]] for x in inputs]
```

reward 函数拿不到 ground truth，靠 rl.py 里构建的两级查找表反查。**注意**：只有 `add_gt` / `test_during_training` / `dynamic_sampling` 任一开启时才计算，纯采样路径不反查。

---

## 3. 修改部分详解

### 3.1 类与 `__init__` 参数

- 类名 `GRPOTrainer` → `ReReTrainer`，继承保持 `transformers.Trainer`（TRL 0.10-0.11 的继承方式；TRL 1.12 已改为 `_BaseTrainer`，API 不兼容，这也是作者不能用现装 TRL 的原因）；
- 新增参数（rl.py 侧对应）：`base_model`（约束解码的 tokenizer/模板来源）、`prompt2history`/`history2target`（reward 反查表）、`info_file`（SID 前缀表数据源）、`beam_search`、`test_during_training`、`test_beam`、`dynamic_sampling`、`add_gt`、`dapo`、`gspo`、`length_penalty`。

### 3.2 generation_config：TRL 只有采样，这里加了 beam 分支（`__init__` 内）

```python
if self.beam_search:
    GenerationConfig(num_beams=G, num_return_sequences=G, do_sample=False, temperature=...)
else:
    GenerationConfig(do_sample=True, temperature=...)
```

**已修正（2026-09-02）**：初版两个分支都保留 `do_sample=True`，beam 分支的语义是"采样束搜索"；现已改为 beam 分支 `do_sample=False`（确定性束搜索，论文 3.4.1 口径）。else 分支（纯采样）保持 `True` 不变。

### 3.3 compute_loss：无 clip 的 GRPO 目标

对照论文公式：
```
论文：min( w·A, clip(w, 1-ε, 1+ε)·A ) - β·KL      ← 有 PPO 式 clip
代码：exp(logp - logp.detach()) * A - β·KL         ← 无 clip（DPO 式简化）
```
`per_token_kl = exp(ref-new) - (ref-new) - 1` 与 TRL 逐字一致（TRL 的 KL 估计式）。
on-policy 下生成与更新之间 θ 不变 → **ratio 数值恒为 1**，故 clip 无意义（见 `docs/RL黑话与GRPO实现详解.md` §4.2）。

### 3.4 RepeatRandomSampler

TRL 原版支持 `mini_repeat_count`/`batch_size`/`shuffle` 精细控制；本项目简化为 `randperm + repeat_count`：每 epoch 洗牌一次，每个索引连续重复 G 次。效果：同一 prompt 的 G 条生成**在同一进程内相邻**，便于组内归一化（多卡时跨进程的组靠 `gather` 汇总）。

### 3.5 detach 补丁（`_prepare_inputs` 内，2026-09-03）

`prompt_completion_ids = prompt_completion_ids.detach()` —— 切断 rollout 生成图与训练图的连接，避免把生成过程的计算图带进反向（rollout 用 `no_grad` 之外还有 repeat 采样的图残留）。初版文档未覆盖。

---

## 4. 保留原样部分（可直接对照 TRL 文档）

- `_get_per_token_logps`：per-token logp 提取（`logits_to_keep` 技巧）
- `_masked_column_advantages` 的 `mode="group"` 默认路径：advantage 组内归一化 `(r - mean) / (std + 1e-4)`，每组 G 条
- EOS 截断 mask
- ref logps 计算（`sync_ref_model` 时用 ref 模型，否则 disable_adapter 后算）
- `_set_signature_columns_if_needed` / `log` / `prediction_step` / `_move_model_to_vllm`
- 梯度累积整除校验：`global_batch_size % G == 0` 必须成立

---

## 5. 潜在问题清单（跑 RL 前建议核对）

### 已解决 ✅

1. **`use_model_defaults` bug（与 evaluate.py 同款）—— 已于 2026-09-01 修复**：transformers 4.57 默认 `use_model_defaults=True`，模型 config 的默认值（Qwen3 的 `do_sample=True, temperature=0.6`）会覆盖显式 GenerationConfig——`test_generation_config` 里 `do_sample=False` 会失效变成采样，训练中评估失真（evaluate.py 之前踩过这个坑，HR@1 差 21.8%）。已给 `_prepare_inputs` 内**四处** `unwrapped_model.generate()` 调用（vLLM 那处 `self.llm.generate` 除外）补 `use_model_defaults=False`。详见 `MONITORING_LOG.md` 2026-09-01 条目。
2. ~~**beam + `do_sample=True` 的语义**~~ —— 已于 2026-09-02 解决：beam 分支改为 `do_sample=False`（确定性束搜索）。else 分支（纯采样）保持 `True`。
3. ~~**`ccc` 的 `prefix_index=3` 硬编码**~~ —— 已改为条件分支（`build_sid_hash_tries` 内按 `base_model` 是否含 `gpt2` 取 4/3）。**换 prompt 模板时仍需同步改这个判据。**

### 仍需注意 ⚠️

4. **无 clip**：默认 GRPO 目标没有 PPO 式 clip。on-policy 下 ratio 恒 1 所以无碍，但若将来引入 off-policy（多步更新）需要重新评估。
5. **`cf_reward` 与数据集格式耦合**：`split("::")` 要求 `prompt2history` 存 SID 序列，混入 RLTitle2Sid 数据会 KeyError。
6. **vLLM 代码残留**：`_move_model_to_vllm`、`self.llm` 等路径在 `use_vllm=True` 时才会触发，本项目不用但注意别误开。
7. **`num_generations` 整除校验**：4 卡 × `per_device_train_batch_size` 必须是 16 的倍数（如 4×4=16 ✓；4×2=8 ✗ 会直接报错）。
8. **reward 恒为负**：ndcg_rule_reward 设计使平均 reward 为负，wandb 曲线是负值属正常。
9. **单卡 OOM 尖峰**：历史上 batch32 在 step 2661 OOM（瞬时峰值 ~29.6G，源于 rollout 的 fp32 math SDPA 注意力矩阵随行数线性增长）。v2 减半 rollout 行数后两段各跑满 3750。单卡复现时注意 `per_device_train_batch_size`。

---

## 6. 与论文 3.4 节的对应关系

| 论文内容 | 代码实现 |
|---|---|
| 3.4.1 Beam Search（默认采样器） | `__init__` 的 `generation_config` beam 分支 + `_prepare_inputs` 的 beam 路径 |
| 3.4.1 Dynamic Sampling（对比项） | `_prepare_inputs` 的 dynamic_sampling 分支 |
| 3.4.1 Diversity 指标 Div | `_prepare_inputs` 内的 `cate_diversity` |
| 3.4.2 硬二值奖励 R_rule | rl.py `rule_reward` |
| 3.4.2 rank-aware 奖励 R_rank | rl.py `ndcg_rule_reward` |
| 3.4.2 collaborative reward | rl.py `cf_reward`（SASRec） |
| 3.4.2 最终奖励 = R_rule + R_rank | rl.py `reward_type="ranking"` |
| 组内归一化 advantage | `ReReTrainer._masked_column_advantages` |
| KL 约束（β） | `compute_loss` 的 `per_token_kl` 项 + `beta` 参数 |
