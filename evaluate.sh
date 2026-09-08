# Industrial_and_Scientific
# Office_Products
for category in "Industrial_and_Scientific"
do
    # your model path（可用环境变量覆盖，如：EXP_NAME=./outputs_ds/final_checkpoint bash evaluate.sh）
    exp_name="${EXP_NAME:-./outputs/final_checkpoint}"

    # 2026-09-03 slug 撞名修复：结果/临时目录用全路径下划线化命名（原来 basename 会让
    # ./outputs/final_checkpoint 与 ./outputs_lastpooling/run_*/final_checkpoint 都叫 final_checkpoint，
    # 互相覆盖 results/——mean-SFT 的逐样本 json 就是这样被 td-last SFT 评测覆盖丢失的）
    exp_name_clean=$(echo "$exp_name" | sed 's|^\./||; s|/|_|g')
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/test/${category}*.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    # 把 test_csv 按卡数切分
    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    # 每张卡同时跑评估
    cudalist="0 1 2 3"  
    echo "Starting parallel evaluation (STANDARD MODE)..."
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
