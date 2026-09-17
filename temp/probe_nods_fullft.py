#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实测：全参微调 + HF Trainer + 不走 DeepSpeed 时，优化器状态是什么精度？
复刻 sft.sh 的配方，只去掉 --deepspeed_config。
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

class Toy(Dataset):
    def __init__(s, n=64, L=16): s.n, s.L = n, L
    def __len__(s): return s.n
    def __getitem__(s, i):
        ids = [(i*7+j) % 256 for j in range(s.L)]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(ids),
                "attention_mask": torch.ones(s.L, dtype=torch.long)}

cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()      # ← 模拟 sft.py:311
print(f"模型参数 dtype: {next(model.parameters()).dtype}")

args = TrainingArguments(output_dir="temp/_nods_out", bf16=True, max_steps=1,
                         per_device_train_batch_size=4, report_to=[], logging_steps=1,
                         save_strategy="no", optim="adamw_torch",               # ← sft.sh 同款
                         learning_rate=1e-4, disable_tqdm=True)
tr = Trainer(model=model, args=args, train_dataset=Toy())
print(f"deepspeed 启用? {tr.is_deepspeed_enabled}   optimizer 类: {tr.args.optim}")
tr.train()

opt = tr.optimizer
print(f"\n底层优化器类名: {opt.__class__.__name__}")
print(f"它优化的参数 dtype: {opt.param_groups[0]['params'][0].dtype}")
print(f"\n=== Adam 状态张量实际 dtype（实测）===")
n = 0
for pid, st in opt.state.items():
    for k, v in st.items():
        if torch.is_tensor(v) and v.numel() > 1:
            print(f"  state[{k:<11}] dtype={str(v.dtype):<16} shape={tuple(v.shape)}")
    n += 1
    if n >= 3: break
