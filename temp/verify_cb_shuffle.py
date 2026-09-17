#!/usr/bin/env python3
"""对照组的预检：打乱码本分配后，初始 loss 落在哪？"""
import json, sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from data import SidSFTDataset
from sft import init_new_emb_from_codebook

BASE="data/pretrained_model/Qwen3-0.6B"; CAT="industrial and scientific items"
D="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830"
TRAIN=f"{D}/train/Industrial_and_Scientific_5_2018-10-2023-9.csv"; ORIG=151669
TK_ARGS=dict(trust_remote_code=True)

tk=AutoTokenizer.from_pretrained(BASE,**TK_ARGS)
tk.pad_token=tk.eos_token; tk.pad_token_id=tk.eos_token_id; tk.padding_side="left"
idx=json.load(open(f"{D}/Industrial_and_Scientific.index.json"))
tk.add_tokens(sorted({t for v in idx.values() for t in v})); V=len(tk)
ds=SidSFTDataset(train_file=TRAIN,tokenizer=tk,max_len=512,sample=256,seed=42,category=CAT)
coll=DataCollatorForSeq2Seq(tk,pad_to_multiple_of=8,return_tensors="pt",padding=True)
batch={k:v.cuda() for k,v in coll([ds[i] for i in range(16)]).items()}
print(f"真实 batch {tuple(batch['input_ids'].shape)}\n")

mats={}
for tag,mode in [("resize 基线",None),("码本(正确)",False),("码本(打乱)",True)]:
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16).cuda()
    m.resize_token_embeddings(V)
    if mode is not None:
        init_new_emb_from_codebook(m,tk,ORIG,f"{D}/Industrial_and_Scientific.index.json",shuffle=mode)
    W=m.get_input_embeddings().weight[ORIG:].float()
    mats[tag]=W.clone()
    with torch.no_grad(): out=m(**batch)
    print(f"  {tag:<12} loss={out.loss.item():.3f}   范数均值={W.norm(dim=1).mean():.4f}")
    del m; torch.cuda.empty_cache()

print()
a,b,c=mats["码本(正确)"],mats["码本(打乱)"],mats["resize 基线"]
print(f"向量多重集是否相同（正确 vs 打乱）: 排序后逐元素最大差 = {(a.norm(dim=1).sort().values-b.norm(dim=1).sort().values).abs().max():.2e}")
print(f"两两 cos 矩阵对角均值（正确 vs 打乱）= {torch.nn.functional.cosine_similarity(a,b,dim=1).mean():+.4f}  ← 应接近 0")
