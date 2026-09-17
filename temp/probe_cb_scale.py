#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""码本初始化尺度扫描（2026-09-15）

背景：初版实现把码本向量重标定到 old_norm（老词表行范数 0.926），结果首步 loss
29.76 vs resize 基线 6.80 —— 新 token 的 logits std 从 1.88 暴涨到 17.71，把正确答案淹没。
诊断：老词表的 0.926 是**训练出来的自信**，不是新 token 该有的起点；且码本方向与 hidden
部分对齐（R²=0.40）→ logits 均值被抬到 10.86（resize 2.44），系统性抢概率。

本探针：保留码本**方向**（语义），按不同尺度重标定**范数**，用真实数据测初始 loss，
找出与 resize 基线"同样中性"的尺度 —— 这才是干净的 A/B（唯一变量 = 方向）。
"""
import json, sys, torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from data import SidSFTDataset

BASE = "data/pretrained_model/Qwen3-0.6B"
CAT = "industrial and scientific items"
D = "data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830"
TRAIN = f"{D}/train/Industrial_and_Scientific_5_2018-10-2023-9.csv"
ORIG = 151669

tk = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tk.pad_token = tk.eos_token; tk.pad_token_id = tk.eos_token_id; tk.padding_side = "left"
idx = json.load(open(f"{D}/Industrial_and_Scientific.index.json"))
tk.add_tokens(sorted({t for v in idx.values() for t in v}))
V = len(tk)
added = tk.added_tokens_encoder

# 码本 → {level: {code: token}}
cb = __import__("numpy").load(f"{D}/Industrial_and_Scientific.codebooks_constrained.npz")
codes = __import__("numpy").load(f"{D}/Industrial_and_Scientific.codes_constrained.npy")
nlv = len([k for k in cb.files if k.startswith("codebook_")])
lmap = [dict() for _ in range(nlv)]
for i in range(len(codes)):
    toks = idx[str(i)]
    for lv in range(nlv):
        lmap[lv].setdefault(int(codes[i, lv]), toks[lv])

ds = SidSFTDataset(train_file=TRAIN, tokenizer=tk, max_len=512, sample=256, seed=42, category=CAT)
coll = DataCollatorForSeq2Seq(tk, pad_to_multiple_of=8, return_tensors="pt", padding=True)
batch = {k: v.cuda() for k, v in coll([ds[i] for i in range(16)]).items()}
print(f"真实 batch: {tuple(batch['input_ids'].shape)}  (V={V}, 新增 token={V-ORIG})")

DIRS = {}
for lv in range(nlv):
    C = torch.tensor(cb[f"codebook_{lv}"], dtype=torch.float32)
    DIRS[lv] = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)   # 只保留方向

def build(norm):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    m.resize_token_embeddings(V)
    emb = m.get_input_embeddings()
    with torch.no_grad():
        written = set()
        for lv in range(nlv):
            for code, tok in lmap[lv].items():
                tid = added.get(tok)
                if tid is None: continue
                emb.weight.data[tid] = (DIRS[lv][code] * norm).to(emb.weight.dtype)
                written.add(tid)
        for i in range(ORIG, V):                       # <d_*>：随机方向，同尺度
            if i in written: continue
            v = emb.weight.data[i].float()
            emb.weight.data[i] = (v / v.norm().clamp_min(1e-12) * norm).to(emb.weight.dtype)
    return m, emb

print()
print(f"{'尺度(范数)':>12} | {'loss':>9} | {'新token范数':>11} | {'新区间logit mean':>16} {'std':>8} {'max':>8}")
print("-" * 88)
for norm in [None, 0.05, 0.1, 0.2, 0.3026, 0.5, 0.9262]:
    m, emb = build(norm) if norm else (None, None)
    if m is None:                                       # resize 基线
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
        m.resize_token_embeddings(V); emb = m.get_input_embeddings()
        tag = "resize基线"
    else:
        tag = f"{norm}"
    with torch.no_grad():
        out = m(**batch)
        lg = out.logits[0, -1].float()
        nn_ = emb.weight[ORIG:].float().norm(dim=1).mean().item()
    print(f"{tag:>12} | {out.loss.item():>9.3f} | {nn_:>11.4f} | "
          f"{lg[ORIG:].mean():>16.3f} {lg[ORIG:].std():>8.3f} {lg[ORIG:].max():>8.2f}")
    del m; torch.cuda.empty_cache()
