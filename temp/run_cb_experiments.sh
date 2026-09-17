#!/bin/bash
# ============================================================
# 码本初始化实验：串行队列（2026-09-15）
#
#   阶段 1  LoRA  + codebook 训练   → 阶段 2  评估
#   阶段 3  全参  + codebook 训练   → 阶段 4  评估
#
# 本机只有一张 5090，所以全部串行。
#
# ⭐ 每个阶段都可续跑（因为 2026-09-14 吃过意外停机的亏）：
#     - 训练：若部署锚点已有 model.safetensors → 跳过；否则找最新 checkpoint 自动 --resume_from_checkpoint
#     - 评估：eval_1gpu.sh 内部按 8 块分片，已完成的块自动跳过
#   被打断后**重跑本脚本即可**从中断处继续，不需要人工判断。
#
# 用法：bash temp/run_cb_experiments.sh 2>&1 | tee -a logs/cb_queue.log
# ============================================================
set -e
cd "$(dirname "$0")/.."

LORA_CB_ROOT="./outputs_sft_lora_cb"
FULL_CB_ROOT="./outputs_sft_cb"
FULL_CB_MICRO="${FULL_CB_MICRO:-8}"     # 单卡实测：8 → 18.7 GiB（安全）；16 在长批次有 OOM 风险

stage_done() { [ -f "$1/final_checkpoint/model.safetensors" ]; }

# ⚠️ set -e 注意：本函数返回 0 恒定，且调用处不能写 `[ -n "$x" ] && echo`——
#    那种写法在变量为空时整条命令返回 1，set -e 会直接把脚本杀掉（2026-09-15 踩过）。
latest_ckpt() {                          # $1 = OUTPUT_ROOT；无 checkpoint 时输出空串
    local d ck
    d=$(ls -dt "$1"/run_*/ 2>/dev/null | head -1 || true)
    if [ -z "$d" ]; then return 0; fi
    ck=$(ls -dt "${d}"checkpoint-* 2>/dev/null | head -1 || true)
    if [ -n "$ck" ]; then echo "$ck"; fi
    return 0
}

check_disk() {
    local avail
    avail=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
    echo ">>> 磁盘余量：${avail}G"
    if [ "${avail:-0}" -lt 12 ]; then
        echo "❌ 磁盘余量 ${avail}G < 12G，训练 checkpoint 可能写满；先清理再跑"; exit 1
    fi
}

banner() { echo; echo "############################################################"; echo "## $*"; echo "############################################################"; }

[ -f /root/autodl-tmp/MiniOneRec-main/logs ] 2>/dev/null || true
mkdir -p logs

# ---------- 阶段 1：LoRA + codebook 训练 ----------
banner "阶段 1/4：LoRA + codebook 训练"
if stage_done "${LORA_CB_ROOT}"; then
    echo ">>> 已完成（${LORA_CB_ROOT}/final_checkpoint 存在），跳过"
else
    check_disk
    RESUME_CKPT=$(latest_ckpt "${LORA_CB_ROOT}")
    if [ -n "${RESUME_CKPT}" ]; then echo ">>> 发现未完成的 checkpoint：${RESUME_CKPT}（将续跑）"; fi
    # NPROC=1：本机单卡（sft_lora.sh 默认 4，不设会被守卫拦下）。
    # MICRO 不传 → 用脚本默认 16，与 LoRA 基线保持同一变量。
    NPROC=1 INIT_EMB=codebook OUTPUT_ROOT="${LORA_CB_ROOT}" RESUME="${RESUME_CKPT}" bash sft_lora.sh
fi

# ---------- 阶段 2：LoRA + codebook 评估 ----------
banner "阶段 2/4：LoRA + codebook 测试集评估"
EXP_NAME="${LORA_CB_ROOT}/final_checkpoint" bash temp/eval_1gpu.sh

# ---------- 阶段 3：全参 + codebook 训练 ----------
banner "阶段 3/4：全参 + codebook 训练（MICRO=${FULL_CB_MICRO}）"
if stage_done "${FULL_CB_ROOT}"; then
    echo ">>> 已完成（${FULL_CB_ROOT}/final_checkpoint 存在），跳过"
else
    check_disk
    RESUME_CKPT=$(latest_ckpt "${FULL_CB_ROOT}")
    if [ -n "${RESUME_CKPT}" ]; then echo ">>> 发现未完成的 checkpoint：${RESUME_CKPT}（将续跑）"; fi
    MICRO="${FULL_CB_MICRO}" OUTPUT_ROOT="${FULL_CB_ROOT}" RESUME="${RESUME_CKPT}" bash sft_cb.sh
fi

# ---------- 阶段 4：全参 + codebook 评估 ----------
banner "阶段 4/4：全参 + codebook 测试集评估"
EXP_NAME="${FULL_CB_ROOT}/final_checkpoint" bash temp/eval_1gpu.sh

banner "队列全部完成"
