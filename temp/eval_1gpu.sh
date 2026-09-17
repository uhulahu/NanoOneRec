#!/bin/bash
# ============================================================
# 单卡测试集评估 —— 分块可续跑版（evaluate.sh 的 1-GPU 变体）
# 2026-09-14 初版；2026-09-15 改为分块可续跑
#
# 为什么需要：
#   1. evaluate.sh 写死了 4 卡（split.py --cuda_list "0,1,2,3" + cudalist="0 1 2 3"），
#      本机只有一张 5090。逻辑照搬，只改卡数，结果目录命名规则保持一致以便对比。
#   2. 2026-09-14 10:00 意外停机，把跑到 28%（573/2021）的评估连同 tmux 一起杀掉——
#      evaluate.py 只在**全部跑完**后写一次 JSON，所以前功尽弃。
#      改为切成 NCHUNKS 块**顺序**跑、每块各自落盘，重启时已完成的块直接跳过。
#      再停机最多丢 1 块（~4 分钟），而不是 30 分钟。
#
# 用法：EXP_NAME=./outputs_sft_lora/final_checkpoint bash temp/eval_1gpu.sh
#       NCHUNKS=8 可调（默认 8）
# ============================================================
set -e

EXP_NAME="${EXP_NAME:?需要设置 EXP_NAME，如 EXP_NAME=./outputs_sft_lora/final_checkpoint}"
NCHUNKS="${NCHUNKS:-8}"

category="Industrial_and_Scientific"
base="data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830"

test_file=$(ls ${base}/test/${category}*.csv 2>/dev/null | head -1)
info_file=$(ls ${base}/info/${category}*.txt 2>/dev/null | head -1)
[ -f "$test_file" ] && [ -f "$info_file" ] || { echo "❌ 数据文件缺失：$base"; exit 1; }

# 与 evaluate.sh 完全一致的 slug 命名（下划线化全路径），保证 results 目录可对比
exp_name_clean=$(echo "$EXP_NAME" | sed 's|^\./||; s|/|_|g')
temp_dir="./temp/${category}-${exp_name_clean}"
output_dir="./results/${exp_name_clean}"

# 切片名用 0..N-1。注意 split.py 用 fire：传 "0,1,2" 会被 ast.literal_eval 解析成元组 (0,1,2)，
# 传 "0" 会解析成 int 0 —— 两种路径 split.py 都处理了。
CUDA_LIST=$(seq -s, 0 $((NCHUNKS - 1)))

echo ">>> EXP_NAME       : ${EXP_NAME}"
echo ">>> exp_name_clean : ${exp_name_clean}"
echo ">>> NCHUNKS        : ${NCHUNKS}  (分块 ${CUDA_LIST})"
echo ">>> temp_dir       : ${temp_dir}"
echo ">>> output_dir     : ${output_dir}"
echo ">>> test_file      : ${test_file}"
echo

# 承重校验：必须是自包含的 HF 模型目录
[ -f "${EXP_NAME}/model.safetensors" ] || {
    echo "❌ ${EXP_NAME} 下没有 model.safetensors —— 不是完整模型目录，evaluate.py 加载不了"; exit 1; }

mkdir -p "$temp_dir" "$output_dir"

# === 分块配置守卫：分块方式变了就必须重切，否则旧 JSON 与新 CSV 不对应 ===
# ⚠️ 教训（2026-09-15）：不能只凭 "0.csv 存在" 就跳过切分 —— 上一版单块脚本留下的
# 0.csv 是**全量**测试集，会被误判成"已切分"，于是把全量当块 0 跑。
# 判据必须是"**本脚本写的**配置文件存在且一致"，无配置文件 ⇒ 来路不明的 CSV，一律清掉重切。
cfg_file="${temp_dir}/.chunk_config"
if [ -f "$cfg_file" ]; then
    if [ "$(cat "$cfg_file")" != "$CUDA_LIST" ]; then
        echo "⚠️  分块配置从 [$(cat $cfg_file)] 变为 [${CUDA_LIST}] —— 旧块结果失效，清空重来"
        rm -f "$temp_dir"/*.csv "$temp_dir"/*.json
    fi
else
    echo ">>> 无分块配置文件（首次运行/旧版遗留）—— 清空 temp 目录后重新切分"
    rm -f "$temp_dir"/*.csv "$temp_dir"/*.json
fi
echo "$CUDA_LIST" > "$cfg_file"

echo ">>> [1/4] 切分测试集（${NCHUNKS} 块）"
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$CUDA_LIST"

# 承重校验：必须真的有 NCHUNKS 个分片，且第 0 块行数 ≈ 全量/NCHUNKS
n_csv=$(ls "$temp_dir"/*.csv 2>/dev/null | wc -l)
[ "$n_csv" -eq "$NCHUNKS" ] || { echo "❌ 期望 ${NCHUNKS} 个分片，实际 ${n_csv} 个"; exit 1; }
total_lines=$(wc -l < "$test_file")
chunk_lines=$(wc -l < "${temp_dir}/0.csv")
echo "    分片数 ${n_csv} ✓  全量 ${total_lines} 行 / 每块约 ${chunk_lines} 行"

echo
echo ">>> [2/4] 逐块评估（beam=50, batch=8, max_new_tokens=256）—— 已有结果的块跳过"
for i in $(seq 0 $((NCHUNKS - 1))); do
    if [ -f "${temp_dir}/${i}.json" ]; then
        echo "--- 块 ${i}/${NCHUNKS} 已有结果，跳过 ---"
        continue
    fi
    echo "--- 块 ${i}/${NCHUNKS} 开始 ---"
    CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
        --base_model "$EXP_NAME" \
        --info_file "$info_file" \
        --category ${category} \
        --test_data_path "$temp_dir/${i}.csv" \
        --result_json_data "$temp_dir/${i}.json" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 256 \
        --length_penalty 0.0
done

echo
echo ">>> [3/4] 合并结果"
python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "$output_dir/final_result_${category}.json" \
    --cuda_list "$CUDA_LIST"

echo
echo ">>> [4/4] 计算指标"
python ./calc.py \
    --path "$output_dir/final_result_${category}.json" \
    --item_path "$info_file"

echo
echo "✅ 评估完成：$output_dir/final_result_${category}.json"
