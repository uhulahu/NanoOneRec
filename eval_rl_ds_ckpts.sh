#!/bin/bash
# ============================================================
# zero2-RL 两段 × 4 步数档（750/2250/3000/3750）共 8 个 ckpt 的 test 集 beam50 评测
# 段1 baseline(ranking)   → outputs_rl_baseline_ds/run_20260907_224824 （750 已归档 ckpt_archive/run_20260907_224824）
# 段2 first_diff          → outputs_rl_firstdiff_ds/run_20260908_010923（750 已归档 ckpt_archive/run_20260908_010923）
# 每档一条 evaluate.sh（EXP_NAME 覆盖）；结果 json → results/<slug>/，指标打在本日志
# 用法：bash eval_rl_ds_ckpts.sh 2>&1 | tee logs/eval_rl_ds_ckpts_<ts>.log
# ============================================================
RUN1="outputs_rl_baseline_ds/run_20260907_224824"
RUN2="outputs_rl_firstdiff_ds/run_20260908_010923"
ARCH="ckpt_archive"

# 顺序按 750/2250 成对优先（bsl,fd）→ 3000/3750：若遇定时关机截断，低档成对结果先完整落盘
for spec in \
    "bsl_750=${ARCH}/run_20260907_224824/checkpoint-750" \
    "fd_750=${ARCH}/run_20260908_010923/checkpoint-750" \
    "bsl_2250=${RUN1}/checkpoint-2250" \
    "fd_2250=${RUN2}/checkpoint-2250" \
    "bsl_3000=${RUN1}/checkpoint-3000" \
    "fd_3000=${RUN2}/checkpoint-3000" \
    "bsl_3750=${RUN1}/checkpoint-3750" \
    "fd_3750=${RUN2}/checkpoint-3750" \
; do
    name="${spec%%=*}"; path="${spec#*=}"
    [ -f "${path}/model.safetensors" ] || { echo "[skip] ${name}: ${path} 无 model.safetensors"; continue; }
    echo "========== 评测 ${name} : ${path} =========="
    EXP_NAME="${path}" bash evaluate.sh
    echo "[done] ${name}  exit=$?"
done
echo "[evals all done]"
