#!/bin/bash
# RL checkpoint 评估（参照 evaluate.sh 流水线：split → 4卡并行 beam search → merge → calc）
# 用法：
#   bash evaluate_rl.sh                          # 评估 outputs_rl 下最新 run 的最大 checkpoint
#   bash evaluate_rl.sh <模型路径>               # 评估指定 checkpoint（如 outputs_rl/run_xxx/checkpoint-750）
# 结果存 results/<run名>_<ckpt名>/，可用 calc.py 输出对比 SFT 基准（HR@1 4.07% / HR@20 7.33%，见 docs/PROGRESS.md）

# 默认：最新 run 的最大 checkpoint
DEFAULT_CKPT="ckpt_archive/run_20260904_005316/checkpoint-750"
exp_name="${1:-$DEFAULT_CKPT}"

if [[ ! -f "$exp_name/config.json" ]]; then
    echo "Error: model path not found: $exp_name"
    exit 1
fi

for category in "Industrial_and_Scientific"
do
    exp_name_clean=$(echo "$exp_name" | sed 's|outputs_rl/||; s|/|_|g')
    echo "Processing category: $category with model: $exp_name_clean"

    train_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/test/${category}*.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt 2>/dev/null | head -1)

    if [[ ! -f "$test_file" ]] || [[ ! -f "$info_file" ]]; then
        echo "Error: test/info file not found for category $category"
        continue
    fi

    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"

    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3"

    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi

    cudalist="0 1 2 3"
    echo "Starting parallel evaluation..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 8 \
                --num_beams 50 \
                --max_new_tokens 256 \
                --length_penalty 0.0 &
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait

    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi

    output_dir="./results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"

    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"

    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi

    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"

    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------"
done

echo "All categories processed!"
