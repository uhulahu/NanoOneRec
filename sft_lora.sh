#!/bin/bash
# ============================================================
# LoRA SFT 消融入口（2026-09-13）
#
# 定位：与 sft.sh（全参 SFT of record：outputs_ds/final_checkpoint，best-828，
#       HR@50 9.66%）**并列的效率消融**，回答"用 ~1.6% 的可训练参数能否接近全参效果"。
#       不触碰主线锚点 outputs_ds/final_checkpoint —— 那是 rl.sh 的起点模型。
#
# 与 sft.sh 的三个刻意差异（变量控制：只改"训练方式"，其余全部对齐）：
#   1. OUTPUT_ROOT=./outputs_sft_lora（物理隔离，不覆盖主线 outputs_ds/final_checkpoint，
#      后者是 rl.sh 的起点模型 best-828）
#   2. 不走 deepspeed zero2 —— zero2 存在的唯一理由是分片全参的 fp32 Adam 状态
#      (4.8G/卡)；LoRA 下优化器只剩 ~76MB，分片收益消失。更重要的是 ds 分支会被
#      ds_zero2_patches.py 劫持，而它硬编码找 best/model.safetensors，PEFT 下写的是
#      adapter_model.safetensors → 早停收尾必崩。非 ds 路径下 transformers 原生支持
#      PEFT 的 best 恢复（trainer.py:3083 的 load_adapter 分支）。
#   3. 部署源是 run_*/final_checkpoint/（merge 后的完整模型），而非 run_*/ 顶层文件
#      —— 后者在 LoRA 下只有 adapter_model.safetensors，rl.py/evaluate.py 加载不了。
#
# 保持与 sft.sh 一致的 A/B 基线：全局 batch 1024（micro16×gas16×4卡）、seed 42、
#   cutoff 512、EarlyStopping(patience=3)、数据路径与 SID 变体完全相同。
#
# 显存预期：稳态 ~15-16G/卡（全参是 ~30.4G）。注意 LoRA 省的是参数+梯度+优化器
#   （约 1.1+1.1+4.8≈7G → ~1.3G），**省不掉** logits ~7.5G 与激活 ~5G（152k 词表决定），
#   所以 micro_batch 上调空间有限。
#
# 用法：
#   全量：  bash sft_lora.sh 2>&1 | tee logs/sft_lora_$(date +%Y%m%d_%H%M%S).log
#   冒烟：  SMOKE=1 bash sft_lora.sh 2>&1 | tee logs/sft_lora_smoke.log
#   调 lr： LR=2e-4 bash sft_lora.sh
# ============================================================
set -e

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 显存池可收缩，防 OOM（历史教训，见 PROGRESS）
export HF_ENDPOINT=https://hf-mirror.com

# === 可调变量（环境变量可覆盖）===
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_sft_lora}"
MODEL_DIR="${MODEL_DIR:-${OUTPUT_ROOT}/final_checkpoint}"
NPROC="${NPROC:-4}"
# micro_batch_size：显存唯一的大旋钮（每样本 ~0.75G，主要来自 logits 链路）。
# ⚠️ 必须是 1024 的约数！gas = batch_size // micro // world 是整除截断的，
#    非约数会让全局 batch 静默变小（如 micro=24 → 1024//24=42 → 全局 1008 ≠ 1024），
#    于是与全参 SFT 的 A/B 可比性被悄悄破坏。单卡 32G 实测：16→12.5G, 32→24.6G, 64→OOM。
MICRO="${MICRO:-16}"
LR="${LR:-1e-4}"              # LoRA 惯例 1e-4~2e-4（全参用 3e-4）
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"       # 注意：扫 LORA_R 时同步扫 LORA_ALPHA，保持 alpha/r=2
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
FREEZE_OLD_EMB="${FREEZE_OLD_EMB:-True}"  # 消融#2：True=保护预训练词嵌入；False=全词表可更新（零成本，内存相同）
# 消融#4：新 token 的 embedding 初始化
#   resize（默认）= transformers 默认随机初始化 → 与全参 SFT of record 可比
#   codebook      = 用 RQ-KMeans 码本向量初始化（实测两空间语义对齐 R²=0.40、
#                   level-0 簇命中率 47.3% vs 随机 0.39%）
# ⚠️ codebook 会改变实验前提（SFT 第 2 个任务本就是"学 SID 语义"），必须作为独立消融，
#    且务必配独立 OUTPUT_ROOT，否则 final_checkpoint 会覆盖 resize 基线的产物：
#      INIT_EMB=codebook OUTPUT_ROOT=./outputs_sft_lora_cb bash sft_lora.sh
INIT_EMB="${INIT_EMB:-resize}"
WANDB_RUN="${WANDB_RUN:-}"
SMOKE="${SMOKE:-0}"           # 1 = 小样本冒烟（sample 2048 / 1 epoch）
RESUME="${RESUME:-}"          # 非空 = 从该 checkpoint 续跑（意外停机后用；trainer 会恢复步数/lr/优化器）

# === 启动前守卫（把静默失败变成响亮失败）===
# ① micro 必须整除 1024，否则 gas = 1024//micro 被截断 → 全局 batch 静默变小，
#    与全参 SFT 的 A/B 可比性被破坏，而脚本不会有任何提示。
if [ $((1024 % MICRO)) -ne 0 ]; then
    echo "❌ MICRO=${MICRO} 不是 1024 的约数：gas=1024//${MICRO}=$((1024 / MICRO))，"
    echo "   全局 batch 将变成 $((MICRO * (1024 / MICRO))) ≠ 1024。"
    echo "   合法值：1 2 4 8 16 32 64 128 256 512 1024"
    exit 1
fi

# ② NPROC 不能超过可见 GPU 数，否则 torchrun 直接失败（报错信息不直观）
_ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "${_ngpu}" -gt 0 ] && [ "${NPROC}" -gt "${_ngpu}" ]; then
    echo "❌ NPROC=${NPROC} > 可见 GPU 数 ${_ngpu}。单卡请用：NPROC=1 bash sft_lora.sh"
    exit 1
fi

echo ">>> 生效配置: MICRO=${MICRO} (gas=$((1024 / MICRO))) NPROC=${NPROC} LR=${LR} " \
     "LORA_R=${LORA_R} ALPHA=${LORA_ALPHA} DROPOUT=${LORA_DROPOUT} " \
     "FREEZE_OLD_EMB=${FREEZE_OLD_EMB} INIT_EMB=${INIT_EMB} SMOKE=${SMOKE}"
echo ">>> 输出: OUTPUT_ROOT=${OUTPUT_ROOT}  MODEL_DIR=${MODEL_DIR}"

# ③ 消融实验的交叉污染提醒：非默认配置却写同一个 final_checkpoint，
#    会让"独立消融"变成"互相覆盖"，事后分不清哪个目录对应哪个配置。
if [ "${INIT_EMB}" != "resize" ] && [ "${OUTPUT_ROOT}" = "./outputs_sft_lora" ]; then
    echo "⚠️  INIT_EMB=${INIT_EMB} 却使用默认 OUTPUT_ROOT —— 会覆盖 resize 基线的 final_checkpoint。"
    echo "    建议：INIT_EMB=${INIT_EMB} OUTPUT_ROOT=./outputs_sft_lora_${INIT_EMB} bash sft_lora.sh"
fi
if [ "${FREEZE_OLD_EMB}" != "True" ] && [ "${OUTPUT_ROOT}" = "./outputs_sft_lora" ]; then
    echo "⚠️  FREEZE_OLD_EMB=${FREEZE_OLD_EMB} 却使用默认 OUTPUT_ROOT —— 同上，建议加独立 OUTPUT_ROOT。"
fi

category="Industrial_and_Scientific"
variant="rqkmeans-td-mean-20260830"
base="data/Amazon23/${category}/sid/${variant}"

train_file=$(ls -f ${base}/train/${category}*.csv | head -1)
eval_file=$(ls -f ${base}/valid/${category}*.csv | head -1)
info_file=$(ls -f ${base}/info/${category}*.txt | head -1)
[ -f "$train_file" ] && [ -f "$eval_file" ] && [ -f "$info_file" ] || { echo "数据文件缺失（$base）；先确认 sid 数据就位"; exit 1; }
echo "train: ${train_file}"
echo "eval : ${eval_file}"

# 冒烟参数（小样本 + 1 epoch，只为验证"跑通/存对"）
if [ "${SMOKE}" = "1" ]; then
    SAMPLE_ARG="--sample 2048"
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
        --learning_rate ${LR} \
        --cutoff_len 512 \
        --freeze_LLM False \
        --use_lora True \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_freeze_old_emb ${FREEZE_OLD_EMB} \
        --init_new_emb ${INIT_EMB} \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path ${base}/Industrial_and_Scientific.index.json \
        --item_meta_path data/Amazon23/${category}/raw/Industrial_and_Scientific.item.json \
        --wandb_run_name "${WANDB_RUN:-}" \
        ${SAMPLE_ARG} ${RESUME_ARG}

# === 部署到 ${MODEL_DIR} ===
# 注意：LoRA 的部署源是 run_*/final_checkpoint/（merge 后的完整模型），
# 不是 run_*/ 顶层——顶层是 adapter_model.safetensors，下游加载不了。
newest=$(ls -dt ${OUTPUT_ROOT}/run_*/ 2>/dev/null | head -1)
[ -n "$newest" ] || { echo "未找到 ${OUTPUT_ROOT}/run_* 训练目录"; exit 1; }
echo "训练完成：$newest"

src="${newest}final_checkpoint"
# 承重校验：这里必须是 merge 后的完整权重。若出现 adapter_model.safetensors 说明
# merge_and_unload() 没执行或失败，此时直接退出，避免把 adapter 当模型部署出去。
[ -f "${src}/model.safetensors" ] || {
    echo "❌ ${src} 下没有 model.safetensors（只有 adapter？）——merge 未生效，拒绝部署"; exit 1; }
[ -f "${src}/adapter_model.safetensors" ] && echo "⚠️  注意：final_checkpoint 里同时存在 adapter 文件"

rm -rf ${MODEL_DIR}.prev_* 2>/dev/null || true                     # 只保留最近一份备份
if [ -d ${MODEL_DIR} ]; then
    mv ${MODEL_DIR} ${MODEL_DIR}.prev_$(date +%Y%m%d_%H%M%S)
    echo "旧锚点已备份为 ${MODEL_DIR}.prev_*"
fi
mkdir -p ${MODEL_DIR}
find "${src}" -maxdepth 1 -type f -exec cp {} ${MODEL_DIR}/ \;
echo "已部署到 ${MODEL_DIR}："
ls ${MODEL_DIR}/
