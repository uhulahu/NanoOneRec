#!/bin/bash
# 目录重组后（2026-09-03）：.inter 与 item.json 在 raw/，index.json 在 sid/<变体>/
PYTHON_SCRIPT="convert_dataset.py"
INPUT_DIR="data/Amazon23/Industrial_and_Scientific/raw"
INDEX_DIR="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-last-20260903"
OUTPUT_DIR="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-last-20260903"
DATASET_NAME="Industrial_and_Scientific"
echo "Start converting $DATASET_NAME ..."
python $PYTHON_SCRIPT \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --index_dir $INDEX_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --seed 42
echo "Finished!"
