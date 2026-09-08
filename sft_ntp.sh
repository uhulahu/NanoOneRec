#!/bin/bash
# ============================================================
# SFT 任务消融：仅 next-item 预测（NTP），去掉 sid-title 互译与历史sid→title 两个 metadata 任务
# （面试/论文口径：三任务混合 SFT 的消融下限；配方与 sft.sh 完全一致：全局 batch 1024
#  = micro16×gas16×4卡、lr 3e-4 linear、10 epoch 上限 + 早停 patience3、seed42、zero2）
# 对比基线：三任务 mean-SFT（test HR [0.04046…0.09720]）与 zero2-SFT（[0.04121…0.09658]）
# 产物：./outputs_ntp_ds/run_<ts>/ → 部署 ./outputs_ntp_ds/final_checkpoint
# ============================================================
set -e
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_ntp_ds}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/final_checkpoint}"
NPROC="${NPROC:-4}"
WANDB_RUN="${WANDB_RUN:-}"   # 非空则记录到 wandb（先 wandb login）

category="Industrial_and_Scientific"
variant="rqkmeans-td-mean-20260830"
base="data/Amazon23/${category}/sid/${variant}"

train_file=$(ls -f ${base}/train/${category}*.csv | head -1)
eval_file=$(ls -f ${base}/valid/${category}*.csv | head -1)
[ -f "$train_file" ] && [ -f "$eval_file" ] || { echo "数据文件缺失"; exit 1; }

torchrun --nproc_per_node ${NPROC} \
        sft.py \
        --base_model data/pretrained_model/Qwen3-0.6B \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --category ${category} \
        --output_dir ${OUTPUT_ROOT} \
        --batch_size 1024 \
        --micro_batch_size 16 \
        --num_epochs 10 \
        --learning_rate 3e-4 \
        --cutoff_len 512 \
        --freeze_LLM False \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path ${base}/Industrial_and_Scientific.index.json \
        --item_meta_path data/Amazon23/${category}/raw/Industrial_and_Scientific.item.json \
        --train_tasks ntp \
        --wandb_run_name "${WANDB_RUN:-}" \
        --deepspeed_config config/ds_zero2.json

# === 部署 ===
newest=$(ls -dt ${OUTPUT_ROOT}/run_*/ 2>/dev/null | head -1)
[ -n "$newest" ] || { echo "未找到 run 目录"; exit 1; }
rm -rf ${MODEL_DIR}.prev_* 2>/dev/null || true
if [ -d ${MODEL_DIR} ]; then mv ${MODEL_DIR} ${MODEL_DIR}.prev_$(date +%Y%m%d_%H%M%S); fi
mkdir -p ${MODEL_DIR}
find "$newest" -maxdepth 1 -type f -exec cp {} ${MODEL_DIR}/ \;
echo "已部署到 ${MODEL_DIR}"; ls ${MODEL_DIR}/
