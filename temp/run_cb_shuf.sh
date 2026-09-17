#!/bin/bash
# ============================================================
# 对照实验：码本向量「打乱分配」（codebook_shuffled）  2026-09-15
#
# 目的：分离「码本的语义」与「方向的分布形状」这两个解释。
#   E2 已证明「码本方向 > MVN 随机方向」（初始 loss 13.18 vs 15.66）。
#   本实验把码本向量在**同一级内**错位重排（forced derangement，无不动点）——
#   向量集合、范数、层级结构全不变，只有「哪个语义向量给了哪个 token」被破坏。
#
# 判读：
#   打乱后 ≈ 正确  → 起作用的是「像真 embedding 的方向分布」，语义对应无关
#   打乱后 ≈ resize（变差）→ 语义对应真的起作用
#   ⚠️ 已知预检信号（单 batch 初始 loss）：resize 15.657 / 正确 13.175 / 打乱 13.765
#      —— 打乱偏向「正确」一侧，初步提示方向分布的贡献更大。但这是**单 batch 的初始 loss**，
#      真正的答案要看测试集指标。另外：码本向量本身有共同方向（正确 vs 打乱 cos=+0.28），
#      所以打乱并未完全破坏信号，这是该对照的固有局限。
#
# 配置与 E2 完全一致（唯一变量 = 分配方式），可断点续跑。
# 用法：bash temp/run_cb_shuf.sh 2>&1 | tee -a logs/cb_shuf.log
# ============================================================
set -e
cd "$(dirname "$0")/.."

ROOT="./outputs_sft_lora_cb_shuf"

stage_done() { [ -f "$1/final_checkpoint/model.safetensors" ]; }

latest_ckpt() {
    local d ck
    d=$(ls -dt "$1"/run_*/ 2>/dev/null | head -1 || true)
    if [ -z "$d" ]; then return 0; fi
    ck=$(ls -dt "${d}"checkpoint-* 2>/dev/null | head -1 || true)
    if [ -n "$ck" ]; then echo "$ck"; fi
    return 0
}

echo "############################################################"
echo "## 对照组：码本打乱分配（codebook_shuffled）"
echo "############################################################"

# 磁盘守卫：阈值可调，因为**从零跑**和**尾部续跑**的需求差很多。
#   从零跑满 8h：4 个 checkpoint(5.2G) + merge(1.2G) + adapter(0.65G) + 部署(1.2G) ≈ 8.3G → 默认 12G 留余量
#   尾部续跑（如只剩几百步）：checkpoint 已在盘上且轮换不增长，只需 merge+adapter+部署 ≈ 3.1G → 可传 5
# 2026-09-16 加：意外停机后续跑时磁盘只剩 8G，被写死的 12G 拦下；改成可参数化而非删数据。
MIN_DISK_GB="${MIN_DISK_GB:-12}"
avail=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
echo ">>> 磁盘余量：${avail}G（本轮要求 ≥ ${MIN_DISK_GB}G）"
if [ "${avail:-0}" -lt "${MIN_DISK_GB}" ]; then
    echo "❌ 磁盘不足：${avail}G < ${MIN_DISK_GB}G"
    echo "   尾部续跑可用 MIN_DISK_GB=5 放宽（实际只需 ~3G）；从零跑请勿低于 12"
    exit 1
fi

if stage_done "${ROOT}"; then
    echo ">>> 训练已完成，跳过"
else
    RESUME_CKPT=$(latest_ckpt "${ROOT}")
    if [ -n "${RESUME_CKPT}" ]; then echo ">>> 续跑自 ${RESUME_CKPT}"; fi
    # 与 E2 的唯一差异：INIT_EMB=codebook_shuffled
    NPROC=1 INIT_EMB=codebook_shuffled OUTPUT_ROOT="${ROOT}" RESUME="${RESUME_CKPT}" bash sft_lora.sh
fi

echo
echo ">>> 测试集评估"
EXP_NAME="${ROOT}/final_checkpoint" bash temp/eval_1gpu.sh

echo
echo "✅ 对照组完成"
