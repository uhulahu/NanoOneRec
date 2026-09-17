#!/usr/bin/env python3
"""验证修正后的码本初始化：直接调用 sft.py 的真实函数，真实数据测 loss。"""
import json, sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from data import SidSFTDataset
from sft import init_new_emb_from_codebook

BASE="data/pretrained_model/Qwen3-0.6B"; CAT="industrial and scientific items"
D="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830"
TRAIN=f"{D}/train/Industrial_and_Scientific_5_2018-10-2023-9.csv"; ORIG=151669

tk=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
tk.pad_token=tk.eos_token; tk.pad_token_id=tk.eos_token_id; tk.padding_side="left"
idx=json.load(open(f"{D}/Industrial_and_Scientific.index.json"))
tk.add_tokens(sorted({t for v in idx.values() for t in v})); V=len(tk)

ds=SidSFTDataset(train_file=TRAIN,tokenizer=tk,max_len=512,sample=256,seed=42,category=CAT)
coll=DataCollatorForSeq2Seq(tk,pad_to_multiple_of=8,return_tensors="pt",padding=True)
batch={k:v.cuda() for k,v in coll([ds[i] for i in range(16)]).items()}
print(f"真实 batch: {tuple(batch['input_ids'].shape)}\n")

for tag,use_cb in [("resize 基线",False),("codebook(修正后)",True)]:
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16).cuda()
    m.resize_token_embeddings(V)
    n0=m.get_input_embeddings().weight[ORIG:].float().norm(dim=1).mean().item()
    if use_cb:
        init_new_emb_from_codebook(m,tk,ORIG,f"{D}/Industrial_and_Scientific.index.json")
    emb=m.get_input_embeddings().weight
    n1=emb[ORIG:].float().norm(dim=1).mean().item()
    with torch.no_grad():
        out=m(**batch); lg=out.logits[0,-1].float()
    print(f"--- {tag} ---")
    print(f"  resize 初始范数 : {n0:.4f}")
    print(f"  写入后范数      : {n1:.4f}")
    print(f"  loss            : {out.loss.item():.3f}")
    print(f"  新区间 logits   : mean={lg[ORIG:].mean():+.3f} std={lg[ORIG:].std():.3f} max={lg[ORIG:].max():.2f}")
    print()
    del m; torch.cuda.empty_cache()
