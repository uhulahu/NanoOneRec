#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性实测：DeepSpeed zero2 + adamw_torch + bf16 下，优化器状态到底是什么精度？

完全复刻 sft.sh 的配方：
  - config/ds_zero2.json（stage 2）
  - torch.optim.AdamW（= TrainingArguments 的 optim="adamw_torch"）
  - 模型以 bfloat16 加载
  - accelerate 会往 ds config 注入 "bf16": {"enabled": true}（因为 bf16=True）
"""
import json
import os
import torch

# 单进程分布式环境
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import deepspeed
from transformers import LlamaConfig, LlamaForCausalLM

cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()   # ← 模拟 sft.py:311
print(f"模型参数 dtype: {next(model.parameters()).dtype}")

ds_config = json.load(open("config/ds_zero2.json"))
ds_config["train_batch_size"] = 4
ds_config["train_micro_batch_size_per_gpu"] = 4
ds_config["gradient_accumulation_steps"] = 1
# "auto" 需要 HF Trainer 解析；裸 DS 下直接给数值
for k in ("gradient_clipping",):
    ds_config.pop(k, None)
# ⚠️ accelerate 在 bf16=True 时会注入这一段（utils/dataclasses.py:1425），
#    所以即使 config/ds_zero2.json 里没有，DS 实际收到的配置里也有
ds_config["bf16"] = {"enabled": True}
print(f"ds 配置: stage={ds_config['zero_optimization']['stage']}, bf16={ds_config['bf16']}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)      # ← optim="adamw_torch"

engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)
O = engine.optimizer

print()
print("=" * 66)
print("DeepSpeed 最终选中的优化器")
print("=" * 66)
print(f"  类名                             : {O.__class__.__name__}")
print(f"  zero stage                       : {ds_config['zero_optimization']['stage']}")

for attr in ("master_weights_and_grads_dtype", "low_precision_master_weights_and_grads",
             "fp16_master_weights_and_gradients", "bf16_master_weights_and_gradients",
             "bf16_optimizer_states"):
    if hasattr(O, attr):
        print(f"  {attr:<32}: {getattr(O, attr)}")

print()
print("  === ZeRO 的两份权重副本 ===")
for attr in ("bit16_groups_flat", "single_partition_of_fp32_groups"):
    if hasattr(O, attr):
        g = getattr(O, attr)
        print(f"  {attr:<32}: {len(g)} 组, dtype={g[0].dtype}, shape={tuple(g[0].shape)}")

print()
print("=== 跑一步训练，然后看优化器内部状态 ===")
ids = torch.randint(0, 256, (4, 32)).cuda()
loss = engine(ids, labels=ids).loss
engine.backward(loss)
engine.step()
print(f"  loss = {loss.item():.4f}")

inner = O.optimizer                                   # 底层 torch 优化器
print(f"  底层优化器类名     : {inner.__class__.__name__}")
print(f"  它优化的参数 dtype : {inner.param_groups[0]['params'][0].dtype}   ← 注意这是 fp32 flat buffer")

print()
print("  === Adam 状态张量的实际 dtype（实测，非推算）===")
for pid, st in list(inner.state.items())[:2]:
    for k, v in st.items():
        if torch.is_tensor(v):
            print(f"    state[{k:<11}] dtype={str(v.dtype):<16} shape={tuple(v.shape)}")
        else:
            print(f"    state[{k:<11}] = {v}  (非张量)")
    break

# 对照：如果直接在 bf16 参数上建 Adam（无 DS、无 fp32 master weights）
print()
print("  === 对照组：普通 torch AdamW 直接跑 bf16 参数（无 DeepSpeed）===")
p = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16, device="cuda"))
o2 = torch.optim.AdamW([p], lr=1e-4)
p.grad = torch.randn_like(p)
o2.step()
for k, v in o2.state[p].items():
    if torch.is_tensor(v):
        print(f"    state[{k:<11}] dtype={str(v.dtype):<16}  ← 跟着参数走")
