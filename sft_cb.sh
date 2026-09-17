#!/bin/bash
# ============================================================
# 全参 SFT + 码本初始化新增 embedding（单卡）  2026-09-15
#
# 定位：与 sft.sh（全参 SFT of record：outputs_ds/final_checkpoint，best-828，HR@50 9.66%）
#       **并列的独立消融**，回答"用 RQ-KMeans 码本给新 token 语义初值，全参能受益多少"。
#       不触碰主线锚点 outputs_ds/final_checkpoint —— 那是 rl.sh 的起点模型。
#
# 与 sft.sh 的差异（变量控制：只改"初始化方式"和"并行度"，其余对齐）：
#   1. OUTPUT_ROOT=./outputs_sft_cb（物理隔离，绝不覆盖主线）
#   2. NPROC 默认 1（本机只有一张 5090）；sft.sh 默认 4
#   3. MICRO 可调（sft.sh 硬编码 16）—— 单卡没有 zero2 分片，优化器开销是 4 卡的 4 倍，
#      必须按实测显存重新选值。**全局 batch 严格保持 1024**，否则与主线不可比。
#   4. --init_new_emb codebook
#
# 显存账（单卡、无分片）：参数 1.19G + 梯度 1.19G + fp32 主权重 2.24G + fp32 Adam 4.47G
#   ≈ 9.1 GiB 固定开销，再加 N×4.8 MiB（N = micro × 批内实际 seq）。上限 ~30 GiB 可用。
#
# 用法：
#   冒烟：  SMOKE=1 MICRO=8 bash sft_cb.sh 2>&1 | tee logs/sft_cb_smoke.log
#   全量：  MICRO=8 bash sft_cb.sh 2>&1 | tee logs/sft_cb_$(date +%Y%m%d_%H%M%S).log
#   续跑：  RESUME=./outputs_sft_cb/run_<ts>/checkpoint-<N> MICRO=8 bash sft_cb.sh
# ============================================================
set -e

export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_sft_cb}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/final_checkpoint}"
NPROC="${NPROC:-1}"
MICRO="${MICRO:-8}"
INIT_EMB="${INIT_EMB:-codebook}"
RESUME="${RESUME:-}"
WANDB_RUN="${WANDB_RUN:-}"
SMOKE="${SMOKE:-0}"

# === 启动前守卫（把静默失败变成响亮失败）===
# ① 绝不写主线锚点：sft.sh 的默认 OUTPUT_ROOT 是 ./outputs_ds，指错了就会覆盖 RL 的起点模型
case "${OUTPUT_ROOT}" in
    ./outputs_ds|outputs_ds|./outputs_ds/*)
        echo "❌ OUTPUT_ROOT=${OUTPUT_ROOT} 指向主线锚点目录（rl.sh 的起点模型）——拒绝运行。"
        echo "   codebook 是独立消融，请用默认的 ./outputs_sft_cb"; exit 1;;
esac

# ② micro 必须整除 1024，否则 gas = 1024//micro 被截断 → 全局 batch 静默变小，与主线不可比
if [ $((1024 % MICRO)) -ne 0 ]; then
    echo "❌ MICRO=${MICRO} 不是 1024 的约数：全局 batch 会变成 $((MICRO * (1024 / MICRO))) ≠ 1024"
    echo "   合法值：1 2 4 8 16 32 64 128 256 512 1024"; exit 1
fi

# ③ NPROC 不能超过可见 GPU 数
_ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "${_ngpu}" -gt 0 ] && [ "${NPROC}" -gt "${_ngpu}" ]; then
    echo "❌ NPROC=${NPROC} > 可见 GPU 数 ${_ngpu}"; exit 1
fi

echo ">>> 生效配置: NPROC=${NPROC} MICRO=${MICRO} (gas=$((1024 / MICRO))) INIT_EMB=${INIT_EMB} SMOKE=${SMOKE}"
echo ">>> 输出: OUTPUT_ROOT=${OUTPUT_ROOT}  MODEL_DIR=${MODEL_DIR}"

category="Industrial_and_Scientific"
variant="rqkmeans-td-mean-20260830"
base="data/Amazon23/${category}/sid/${variant}"

train_file=$(ls -f ${base}/train/${category}*.csv | head -1)
eval_file=$(ls -f ${base}/valid/${category}*.csv | head -1)
info_file=$(ls -f ${base}/info/${category}*.txt | head -1)
[ -f "$train_file" ] && [ -f "$eval_file" ] && [ -f "$info_file" ] || {
    echo "❌ 数据文件缺失（$base）"; exit 1; }
echo "train: ${train_file}"; echo "eval : ${eval_file}"

# 码本文件存在性检查（--init_new_emb codebook 依赖它，缺了 sft.py 会在训练前抛错）
cb="${base}/Industrial_and_Scientific.codebooks_constrained.npz"
[ -f "$cb" ] || { echo "❌ 码本文件缺失：$cb"; exit 1; }
echo "codebook: ${cb}"

if [ "${SMOKE}" = "1" ]; then
    SAMPLE_ARG="--sample 2048"      # 小样本，只为验证"跑通/存对/不 OOM"
    EPOCHS=1
    echo ">>> 冒烟模式：sample=2048, epochs=1"
else
    SAMPLE_ARG=""
    EPOCHS=10
fi

if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME}"
    echo ">>> 从 checkpoint 续跑：${RESUME}"
else
    RESUME_ARG=""
fi

torchrun --nproc_per_node ${NPROC} \
        sft.py \
        --base_model data/pretrained_model/Qwen3-0.6B \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --category ${category} \
        --output_dir ${OUTPUT_ROOT} \
        --batch_size 1024 \
        --micro_batch_size ${MICRO} \
        --num_epochs ${EPOCHS} \
        --learning_rate 3e-4 \
        --cutoff_len 512 \
        --freeze_LLM False \
        --init_new_emb ${INIT_EMB} \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path ${base}/Industrial_and_Scientific.index.json \
        --item_meta_path data/Amazon23/${category}/raw/Industrial_and_Scientific.item.json \
        --wandb_run_name "${WANDB_RUN:-}" \
        ${SAMPLE_ARG} ${RESUME_ARG} \
        --deepspeed_config config/ds_zero2.json

# === 部署到 ${MODEL_DIR} ===
newest=$(ls -dt ${OUTPUT_ROOT}/run_*/ 2>/dev/null | head -1)
[ -n "$newest" ] || { echo "❌ 未找到 ${OUTPUT_ROOT}/run_* 训练目录"; exit 1; }
echo "训练完成：$newest"

src="${newest}final_checkpoint"
[ -f "${src}/model.safetensors" ] || {
    echo "❌ ${src} 下没有 model.safetensors —— 拒绝部署"; exit 1; }

rm -rf ${MODEL_DIR}.prev_* 2>/dev/null || true
if [ -d ${MODEL_DIR} ]; then
    mv ${MODEL_DIR} ${MODEL_DIR}.prev_$(date +%Y%m%d_%H%M%S)
    echo "旧锚点已备份为 ${MODEL_DIR}.prev_*"
fi
mkdir -p ${MODEL_DIR}
find "${src}" -maxdepth 1 -type f -exec cp {} ${MODEL_DIR}/ \;
echo "已部署到 ${MODEL_DIR}："
ls ${MODEL_DIR}/
