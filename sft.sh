#!/bin/bash
# ============================================================
# 标准 SFT 入口：mean 变体（rqkmeans-td-mean-20260830）× zero2 × 4×5090
#
# 超参复现 2026-08-30 mean-SFT of record（以 outputs/training_args.bin 为准）：
#   全局 batch 1024：--batch_size 参数 = 全局 batch（sft.py 内 gas = batch_size//micro 再 //nproc 分到每卡，
#   故传 batch 1024 + micro 16 → gas = 1024//16//4 = 16/卡，全局 = 16×16×4 = 1024，与 of-record 逐位一致）；
#   lr 3e-4 linear + warmup 20 + max_grad_norm 1.0；10 epoch 上限 + EarlyStopping(patience=3)
#   （实测 4.5 epoch 早停，best 权重 828 步）；seed 42；cutoff 512；freeze_LLM=False
# 显存：zero2 分片优化器状态（config/ds_zero2.json），每卡余量 ~7-8G；
#       若遇 OOM 优先降 micro_batch_size（保持 batch_size 为 micro 的整数倍即可保持全局 batch 不变）
# 产物：训练目录 ${OUTPUT_ROOT}/run_<时间戳>/（含 tokenizer，sft.py 已传 Trainer）；
#       成功后自动同步顶层模型文件到 ${MODEL_DIR}
#       （rl.sh / evaluate.sh 的 model_path/exp_name 指这里），旧锚点先备份为 .prev_<ts>
#
# === 可调变量（环境变量可覆盖，如：OUTPUT_ROOT=/tmp/out NPROC=2 bash sft.sh）===
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_ds}"                       # 训练输出根目录
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/final_checkpoint}"        # 部署锚点（模型产物目录）
NPROC="${NPROC:-4}"                                              # 并行进程数 = 使用的 GPU 数
WANDB_RUN="${WANDB_RUN:-}"   # 非空则本次 run 记录到 wandb（需先 wandb login）：如 WANDB_RUN=sft_ds_zero2 bash sft.sh；空 = 仅 tensorboard
# ============================================================
set -e

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 显存池可收缩，防 OOM（历史教训，见 PROGRESS）
export HF_ENDPOINT=https://hf-mirror.com

category="Industrial_and_Scientific"
variant="rqkmeans-td-mean-20260830"
base="data/Amazon23/${category}/sid/${variant}"

train_file=$(ls -f ${base}/train/${category}*.csv | head -1)
eval_file=$(ls -f ${base}/valid/${category}*.csv | head -1)
info_file=$(ls -f ${base}/info/${category}*.txt | head -1)
[ -f "$train_file" ] && [ -f "$eval_file" ] && [ -f "$info_file" ] || { echo "数据文件缺失（$base）；先确认 sid 数据就位"; exit 1; }
echo "train: ${train_file}"
echo "eval : ${eval_file}"

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
        --wandb_run_name "${WANDB_RUN:-}" \
        --deepspeed_config config/ds_zero2.json

# === 部署到 ${MODEL_DIR}（模型锚点；rl.sh / evaluate.sh 的 model_path 指这里）===
newest=$(ls -dt ${OUTPUT_ROOT}/run_*/ 2>/dev/null | head -1)
[ -n "$newest" ] || { echo "未找到 ${OUTPUT_ROOT}/run_* 训练目录"; exit 1; }
echo "训练完成：$newest"

rm -rf ${MODEL_DIR}.prev_* 2>/dev/null || true                     # 只保留最近一份备份
if [ -d ${MODEL_DIR} ]; then
    mv ${MODEL_DIR} ${MODEL_DIR}.prev_$(date +%Y%m%d_%H%M%S)
    echo "旧锚点已备份为 ${MODEL_DIR}.prev_*"
fi
mkdir -p ${MODEL_DIR}
find "$newest" -maxdepth 1 -type f -exec cp {} ${MODEL_DIR}/ \;
echo "已部署到 ${MODEL_DIR}："
ls ${MODEL_DIR}/
