#!/bin/bash
#
# RQ-KMeans Constrained Training Script
#

# Default parameters
DATASET="Industrial_and_Scientific"
ROOT="data/Amazon23/$DATASET"
# 2026-09-03 目录重组：向量完整路径传入，码写 sid/<变体名>/（变体名规则见 data/.../README.md）
EMB_PATH="$ROOT/emb/$DATASET.emb-qwen3-E-0.6B-td-last.npy"
SID_VARIANT="rqkmeans-td-last-20260903"
SID_OUT_DIR="$ROOT/sid/$SID_VARIANT"
K=256
L=3
MAX_ITER=100
SEED=42

echo "Dataset: $DATASET"
echo "K=$K, L=$L"

python rq/rqkmeans_constrained.py \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --emb_path "$EMB_PATH" \
    --sid_out_dir "$SID_OUT_DIR" \
    --k "$K" \
    --l "$L" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --verbose
