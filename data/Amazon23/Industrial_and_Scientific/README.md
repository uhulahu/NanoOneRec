# Industrial_and_Scientific 数据目录规范（2026-09-03 重组）

```
data/Amazon23/Industrial_and_Scientific/
├── raw/                              # 原始数据（不可变，来自下载/预处理）
│   ├── Industrial_and_Scientific.item.json    # item_id → {title, description}
│   ├── Industrial_and_Scientific.{train,valid,test}.inter   # 交互划分
│   └── ...（item2id/user2id/review 等）
├── emb/                              # 文本向量（文件名后缀 = pooling/归一化方式）
│   └── Industrial_and_Scientific.emb-qwen3-E-0.6B-td.npy
│       # 规范: -td        = masked mean pooling，无归一化（GPR 脚本产物）——⚠️ 当前实验用这个（off-spec）
│       #       -td-last   = last-token pooling + L2 归一化（amazon_text2emb.py --pooling last，Qwen3-Embedding 官方做法）
└── sid/<变体名>/                     # 每个 SID 码变体一个自包含目录
    ├── Industrial_and_Scientific.{codes_constrained.npy, codebooks_constrained.npz, index.json}
    ├── train/valid/test/*.csv        # 交互数据（内嵌该变体的 SID 字符串）
    └── info/*.txt                    # item 信息（sid \t title \t item_id，约束解码前缀表来源）
```

## 变体命名规则

`rqkmeans-<emb文件后缀>-<YYYYMMDD>`——emb 后缀直通，一眼可知码的来源。

| 变体 | 来源向量 | 状态 |
|---|---|---|
| `rqkmeans-td-mean-20260830` | `-td.npy`（mean pooling，无归一化） | **当前激活**：SFT baseline + 全部 RL 实验（fd/baseline/b/c）与评估都基于它 |
| `rqkmeans-td-last-20260903` | `-td-last.npy`（last-token + L2，待生成） | 待建：验证 embedding 是否拖累绝对水平的探针变体 |

## 使用规则（重要）

1. **激活变体切换流程**：① `amazon_text2emb.py --pooling last` 产出新 emb（进 `emb/`）→ ② rqkmeans 生成新码 → ③ `bash convert_dataset.sh`（改 INDEX_DIR/OUTPUT_DIR 到新变体）重建 CSV+info 进变体目录 → ④ 训练/评估脚本把 `rqkmeans-td-mean-20260830` 换成新变体名。
2. **所有 shell 的数据路径已参数化为变体目录内 glob**（sft.sh / rl.sh / rl_smoke.sh / evaluate.sh / evaluate_rl.sh），换变体只需改目录名。
3. **CSV↔index 一致性兜底**：换变体后必须重跑 convert_dataset.py；校验脚本见 temp/ 中的 CSV↔index 抽样比对。
4. ⚠️ 2026-08-30 及之前的全部实验结果使用 `-td`（mean、未归一化）向量生成的码；方法间相对对比有效，与论文绝对水平的差异可能部分源于此（见 PROGRESS.md）。
