#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""micro_batch_size 显存扫描探针 v2（2026-09-13）

v1 发现的问题：micro 16→32 显存从 12.85G 跳到 24.81G（+12G），远超 logits 线性增长的
预期（~+2.5G）。且 v1 漏设了 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
（sft_lora.sh 里有），测量可能失真。

v2 修正：
  1. 显式设置与 sft_lora.sh 一致的环境变量（必须在 import torch 之前）
  2. 分阶段报告 allocated/reserved：forward / backward / optimizer.step
  3. 报 reserved-allocated（碎片量），定位是不是缓存分配器问题
  4. 同时测「随机样本」和「最长样本」两种 batch（历史 OOM 都是长 prompt 尖峰）

结论要看「最长样本」那一列——那才是真实训练里会不会崩的判据。
"""
import json
import os
import sys

# ⚠️ 必须在 import torch 之前设置，与 sft_lora.sh 完全一致
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

sys.path.insert(0, ".")
from data import SidSFTDataset

BASE = "data/pretrained_model/Qwen3-0.6B"
CATEGORY_CN = "industrial and scientific items"
SID_DIR = "data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830"
INDEX = f"{SID_DIR}/Industrial_and_Scientific.index.json"
TRAIN = f"{SID_DIR}/train/Industrial_and_Scientific_5_2018-10-2023-9.csv"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = 1024
CUTOFF = 512
GB = 2 ** 30


def stat(**kw):
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    return f"alloc={a:6.2f}G reserved={r:6.2f}G frag={r-a:5.2f}G"


def main():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    tk = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tk.pad_token = tk.eos_token
    tk.pad_token_id = tk.eos_token_id
    tk.padding_side = "left"
    idx = json.load(open(INDEX))
    tk.add_tokens(sorted({t for v in idx.values() for t in v}))
    orig_vocab = 151669

    ds = SidSFTDataset(train_file=TRAIN, tokenizer=tk, max_len=CUTOFF,
                       sample=2000, seed=42, category=CATEGORY_CN)
    rows = [ds[i] for i in range(len(ds))]
    lens = sorted(len(r["input_ids"]) for r in rows)
    print(f"数据: {len(rows)} 条 | 长度 中位={lens[len(lens)//2]} p99={lens[int(len(lens)*0.99)]} "
          f"最大={lens[-1]}", flush=True)
    long_rows = sorted(rows, key=lambda r: -len(r["input_ids"]))[:512]
    collator = DataCollatorForSeq2Seq(tk, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
    model.resize_token_embeddings(len(tk))
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGETS))
    emb = model.get_input_embeddings()
    emb.weight.requires_grad = True
    emb.weight.register_hook(lambda g: (g[:orig_vocab].zero_(), g)[1])
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    print(f"模型就位（LoRA r=16）: {stat()}\n", flush=True)

    total = torch.cuda.get_device_properties(0).total_memory
    print(f"{'micro':>5} {'样本':>6} {'真实seq':>7} | {'fwd后':>22} {'bwd后':>22} {'step后':>22} | 峰值")
    print("-" * 118)

    for micro in [16, 32, 48, 64]:
        gas, rem = divmod(BATCH_SIZE, micro)
        if rem:
            print(f"{micro:>5} {'—':>6} {'—':>7} |  ⚠️ 非 1024 约数（gas={gas}, 全局={gas*micro}），跳过")
            continue
        for label, pool in [("随机", rows), ("最长", long_rows)]:
            batch = collator([pool[i % len(pool)] for i in range(micro)])
            seq = batch["input_ids"].shape[1]
            batch = {k: v.to("cuda") for k, v in batch.items()}
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                out = model(**batch)
                f = stat()
                out.loss.backward()
                b = stat()
                opt.step()
                opt.zero_grad(set_to_none=True)
                s = stat()
                peak = torch.cuda.max_memory_allocated() / GB
                free = (total - torch.cuda.max_memory_allocated()) / GB
                flag = "✅" if free > 3 else ("⚠️" if free > 1 else "❌")
                print(f"{micro:>5} {label:>6} {seq:>7} | {f} | {b} | {s} | "
                      f"{peak:5.2f}G 余{free:5.2f}G {flag}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{micro:>5} {label:>6} {seq:>7} | ❌ OOM", flush=True)
                torch.cuda.empty_cache()
                break
            finally:
                del batch
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
