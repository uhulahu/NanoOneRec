#!/bin/bash

# accelerate launch --num_processes 8 amazon_text2emb.py \
#     --dataset Industrial_and_Scientific \
#     --root ../../data/Amazon18/Industrial_and_Scientific \
#     --plm_checkpoint your_emb_model_path

python rq/text2emb/amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root data/Amazon23/Industrial_and_Scientific \
    --plm_name qwen3-E-0.6B \
    --plm_checkpoint data/pretrained_model/Qwen3-Embedding-0.6B \
    --pooling last