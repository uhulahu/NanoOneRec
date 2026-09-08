#!/bin/bash
# ============================================================
# zero2-RL 双段（2026-09-07）：
#   段1 baseline  = ranking（朴素，token_norm=group, penalty=0）   → outputs_rl_baseline_ds
#   段2 first_diff= ranking_firstdiff（group, penalty=0）          → outputs_rl_firstdiff_ds
# 每段完整 1 epoch（sample10000×3 任务 ≈ 3750 步），验证 zero2 是否消除历史 0.6-0.8 epoch OOM；
# 750 步 ckpt 自动归档 ckpt_archive（对照锚点）；起点模型 ./outputs/final_checkpoint（与历史一致）
# zero2：--deepspeed_config config/ds_zero2.json（engine ckpt 已跳过 → 每 ckpt ~1.5G，补丁见
#        ds_zero2_patches.py，rl.py 顶部自动加载）；wandb：WANDB_RUN=xxx bash rl_ds.sh
#
# v2（2026-09-08 凌晨）：v1 在 step 2661（0.71 epoch，恰是历史 OOM 点）仍 OOM——zero2 只降了优化器
# 基座（进程 26.0G 时崩），尖峰来自 rollout 前向（fp32 math SDPA，禁 flash 保持数学后端与历史一致）
# 且随每步 prompts 数线性。修复：per-device 32→16 + gas 1→2 → 单次 rollout 行数减半，而全局语义
# 16×2×4卡=128 prompts/步 与旧 32×1×4 完全一致（步数/样本集/优化器更新时机不变，锚点可比性无污染）。
# ============================================================
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 显存池可收缩，防 OOM
WANDB_RUN="${WANDB_RUN:-}"      # 非空则本次 run 记录到 wandb（需先 wandb login）；空 = 不记录

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/valid/${category}*.csv)
    info_file=$(ls -f ./data/Amazon23/${category}/sid/rqkmeans-td-mean-20260830/info/${category}*.txt)

    # ---- 段1：baseline（ranking）----
    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --num_processes 4 --main_process_port 29503 \
                                    rl.py \
                        --model_path ./outputs/final_checkpoint \
                        --train_batch_size 16 \
                        --eval_batch_size 16 \
                        --num_train_epochs 1 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --sample 10000 \
                        --eval_sample 10000 \
                        --eval_step 0.1999 \
                        --save_steps 0.2 \
                        --reward_type ranking \
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
                        --output_dir outputs_rl_baseline_ds \
                        --sid_index_path ./data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json \
                        --token_norm group \
                        --all_wrong_penalty 0.0 \
                        --archive_steps 750 \
                        --archive_dir ckpt_archive \
                        --wandb_run_name "${WANDB_RUN:-}" \
                        --deepspeed_config config/ds_zero2.json

    # ---- 段2：first_diff（ranking_firstdiff + group + penalty 0）----
    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --num_processes 4 --main_process_port 29503 \
                                    rl.py \
                        --model_path ./outputs/final_checkpoint \
                        --train_batch_size 16 \
                        --eval_batch_size 16 \
                        --num_train_epochs 1 \
                        --gradient_accumulation_steps 2 \
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
                        --output_dir outputs_rl_firstdiff_ds \
                        --sid_index_path ./data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json \
                        --token_norm group \
                        --all_wrong_penalty 0.0 \
                        --archive_steps 750 \
                        --archive_dir ckpt_archive \
                        --wandb_run_name "${WANDB_RUN:-}" \
                        --deepspeed_config config/ds_zero2.json
done
