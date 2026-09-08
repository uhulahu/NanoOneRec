#!/bin/bash

# python amazon23_data_process.py \
#     --dataset {domain} \
#     --metadata_file ../meta_{domain}.jsonl \
#     --reviews_file ../{domain}.jsonl \
#     --user_k 5 \
#     --st_year 2018 \
#     --st_month 10 \
#     --ed_year 2023 \
#     --ed_month 9 \
#     --output_path ./Amazon23


python data/amazon23_data_process.py \
    --dataset Industrial_and_Scientific \
    --metadata_file data/Amazon23/Industrial_and_Scientific/raw_jsonl/meta_Industrial_and_Scientific.jsonl \
    --reviews_file data/Amazon23/Industrial_and_Scientific/raw_jsonl/Industrial_and_Scientific.jsonl \
    --user_k 5 \
    --st_year 2018 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 9 \
    --output_path ./data/Amazon23

# 2026-09-03 目录重组：生成产物（脚本写 {output_path}/{dataset}/ 根）归入 raw/（不碰 raw_jsonl/emb/sid/README）
OUT=./data/Amazon23/Industrial_and_Scientific
for f in "$OUT"/*.json "$OUT"/*.user2id "$OUT"/*.item2id; do
    [ -e "$f" ] && mv "$f" "$OUT/raw/"
done
echo "Generated files moved to $OUT/raw/"