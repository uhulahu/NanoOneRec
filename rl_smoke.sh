#!/bin/bash
# 快速冒烟验证脚本：验证 token 级 first-diff 奖励（ranking_firstdiff 混合模式）整条链路能跑通
# 用法：与 rl.sh 相同的 4 卡环境（accelerate），全流程 ~10-20 分钟（48 个训练步 + 多次 eval）
# 验证点：(1) 不崩、loss/KL 数值正常 (2) rewards/first_diff_reward 指标出现且量级合理
#          (3) eval 的 NDCG/HR 有输出 (4) checkpoint 正常保存
# 跑通后再用 rl.sh（全量数据）正式训练。

export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/valid/${category}*.csv)
    info_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --num_processes 2 --main_process_port 29504 \
                                    rl.py \
                        --model_path ./outputs/final_checkpoint \
                        --train_batch_size 32 \
                        --eval_batch_size 32 \
                        --num_train_epochs 1 \
                        --gradient_accumulation_steps 1 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --sample 2048 \
                        --eval_sample 256 \
                        --eval_step 0.0999 \
                        --reward_type ranking_firstdiff \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 0.04 \
                        --dapo False \
                        --output_dir outputs_rl_smoke \
                        --sid_index_path ./data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json
done
