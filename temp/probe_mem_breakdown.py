#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""显存逐项盘点：完整复刻 sft.py 的 LoRA 配置，跑一步训练，按大小给所有张量分类。"""
import json, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE="data/pretrained_model/Qwen3-0.6B"
SID="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json"
TARGETS=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
MB=2**20; GB=2**30
MICRO=int(sys.argv[1]) if len(sys.argv)>1 else 4
SEQ=int(sys.argv[2]) if len(sys.argv)>2 else 256

tk=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
tk.pad_token=tk.eos_token; tk.pad_token_id=tk.eos_token_id
tk.add_tokens(sorted({t for v in json.load(open(SID)).values() for t in v}))
ORIG=151669

model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16).cuda()
model.resize_token_embeddings(len(tk))
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",
        task_type="CAUSAL_LM",target_modules=TARGETS))
emb=model.get_input_embeddings(); emb.weight.requires_grad=True
emb.weight.register_hook(lambda g:(g[:ORIG].zero_(),g)[1])
model.config.use_cache=False; model.train()
trained=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trained,lr=1e-4)

V=len(tk)
def alloc(): return torch.cuda.memory_allocated()/GB
def resv():  return torch.cuda.memory_reserved()/GB

print(f"=== 配置: micro={MICRO}, seq={SEQ}, N={MICRO*SEQ} token, V={V} ===")
print(f"    可训练参数 {sum(p.numel() for p in trained):,}")
print()
print(f"[A] 加载+Lora+优化器构建后        allocated={alloc():6.2f}G reserved={resv():6.2f}G")

torch.cuda.reset_peak_memory_stats()
ids=torch.randint(0,V,(MICRO,SEQ)).cuda(); am=torch.ones_like(ids); lb=ids.clone()
out=model(input_ids=ids,attention_mask=am,labels=lb)
print(f"[B] forward 后                    allocated={alloc():6.2f}G "
      f"peak={torch.cuda.max_memory_allocated()/GB:6.2f}G")
out.loss.backward()
print(f"[C] backward 后                   allocated={alloc():6.2f}G "
      f"peak={torch.cuda.max_memory_allocated()/GB:6.2f}G")
peak_fb=torch.cuda.max_memory_allocated()/GB
opt.step(); opt.zero_grad(set_to_none=True)
print(f"[D] optimizer.step 后             allocated={alloc():6.2f}G "
      f"peak={torch.cuda.max_memory_allocated()/GB:6.2f}G")
print(f"    ★ 整步峰值 = {peak_fb:.2f}G (forward+backward)   reserved={resv():.2f}G")

print()
print("=== 峰值时刻的内存快照：按块大小分类 ===")
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
ids=torch.randint(0,V,(MICRO,SEQ)).cuda(); am=torch.ones_like(ids); lb=ids.clone()
out=model(input_ids=ids,attention_mask=am,labels=lb); out.loss.backward()
snap=torch.cuda.memory_snapshot()
buckets=Counter()
for seg in snap:
    for b in seg.get("blocks",[]):
        if b.get("state")!="active_allocated": continue
        sz=b.get("size") or 0
        if sz>=4*MB: buckets[round(sz/MB)]+=1
tot=0
for mb,n in sorted(buckets.items(), reverse=True)[:18]:
    print(f"    {mb:>6} MiB × {n:<4} = {mb*n:>7} MiB")
    tot+=mb*n
print(f"    {'≥4MiB 小计':<16} = {tot:>7} MiB")
print(f"    全部 active_allocated      = {torch.cuda.memory_allocated()/MB:.0f} MiB")
print()
print(f"=== 参照：一批 [N,V] 张量 = {MICRO*SEQ*V*2/MB:.0f} MiB (bf16) / {MICRO*SEQ*V*4/MB:.0f} MiB (fp32) ===")
print(f"    模型参数 = {sum(p.numel() for p in model.parameters())*2/MB:.0f} MiB (bf16)")
