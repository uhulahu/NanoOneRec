#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 显存池可收缩，防 OOM（与 sft.sh 一致）
WANDB_RUN="${WANDB_RUN:-}"      # 非空则本次 run 记录到 wandb（需先 wandb login）：如 WANDB_RUN=bT_750 bash rl.sh；空 = 不记录

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/valid/${category}*.csv)
    info_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --num_processes 4 --main_process_port 29503 \
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
                        --sample 10000 \
                        --eval_sample 10000 \
                        --eval_step 0.1999 \
                        --save_steps 0.2 \
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
                        --output_dir outputs_rl \
                        --sid_index_path ./data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json \
                        --token_norm group \
                        --all_wrong_penalty 1.0 \
                        --archive_steps 750 \
                        --archive_dir ckpt_archive \
                        --wandb_run_name "${WANDB_RUN:-}" \
                        --deepspeed_config config/ds_zero2.json
done

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/valid/${category}*.csv)
    info_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --num_processes 4 --main_process_port 29503 \
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
                        --sample 10000 \
                        --eval_sample 10000 \
                        --eval_step 0.1999 \
                        --save_steps 0.2 \
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
                        --output_dir outputs_rl \
                        --sid_index_path ./data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json \
                        --token_norm column \
                        --all_wrong_penalty 0.0 \
                        --archive_steps 750 \
                        --archive_dir ckpt_archive \
                        --wandb_run_name "${WANDB_RUN:-}" \
                        --deepspeed_config config/ds_zero2.json
done