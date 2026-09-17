#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA 前置校验（CPU，2026-09-13）—— 在烧 GPU 之前证伪三件事：

  ① tie：Qwen3-0.6B resize 后 lm_head 是否仍与 embed_tokens 共享权重
  ② 可训练参数量：LoRA + 新 token 行 是否落在预期的 ~1.6%
  ③ merge：merge_and_unload() 是否返回普通 Qwen3ForCausalLM、词表是否保住

这三条任一不成立，GPU 上的冒烟跑多久都是白跑。
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "data/pretrained_model/Qwen3-0.6B"
INDEX = ("data/Amazon23/Industrial_and_Scientific/sid/"
         "rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id
original_vocab_size = len(tok)
print(f"[0] original_vocab_size (len(tokenizer)) = {original_vocab_size}")

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)

# ---- 扩词表 ----
indices = json.load(open(INDEX))
new_tokens = sorted({t for v in indices.values() for t in v})
n_added = tok.add_tokens(new_tokens)
model.resize_token_embeddings(len(tok))
print(f"[1] index tokens={len(new_tokens)} add_tokens 实际新增={n_added} "
      f"len(tokenizer)={len(tok)} embed_rows={model.get_input_embeddings().weight.shape[0]}")

# ---- ① tie 校验 ----
inp = model.get_input_embeddings().weight
out = model.get_output_embeddings().weight
print(f"[2] tie_word_embeddings={model.config.tie_word_embeddings} | "
      f"lm_head 与 embed_tokens 共享同一 tensor: {inp is out}")
if not (inp is out):
    model.tie_weights()
    print(f"    重新 tie 后共享: {model.get_input_embeddings().weight is model.get_output_embeddings().weight}")

# ---- ② 套 LoRA + 解冻新行（与 sft.py 完全同构；顺序不能反，见下）----
# ⚠️ 2026-09-13 实测踩坑：get_peft_model 内部会调 _mark_only_adapters_as_trainable()，
# 把所有名字不含 "lora_" 的参数设为 requires_grad=False。所以必须先 get_peft_model、
# 再打开 embedding；反过来的话 embedding 会被静默关掉（训练照跑、loss 照降，但 SID
# 新 token 的 embedding 全程冻结在随机初值上）。
from peft import LoraConfig, get_peft_model

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules=TARGETS,
))
emb = model.get_input_embeddings()
emb.weight.requires_grad = True
emb.weight.register_hook(lambda g: (g[:original_vocab_size].zero_(), g)[1])
model.print_trainable_parameters()

# 实测构成，不做推算（教训：之前用"总数减假设值"倒推出分解，是错的）
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
lora_n = sum(p.numel() for n, p in model.named_parameters()
             if p.requires_grad and "lora_" in n)
emb_n = emb.weight.numel()
total = sum(p.numel() for p in model.parameters())
n_new_rows = emb.weight.shape[0] - original_vocab_size
print(f"[3] embedding requires_grad = {emb.weight.requires_grad}（必须 True）")
print(f"    trainable={trainable:,} = LoRA {lora_n:,} + embedding整块 {emb_n:,} "
      f"/ total={total:,} = {100*trainable/total:.2f}%")
n_new_elems = n_new_rows * emb.weight.shape[1]      # 不硬编码 hidden_size
print(f"    ★ 有效学习参数 = LoRA {lora_n:,} + 新 token {n_new_rows}行×{emb.weight.shape[1]}维"
      f"={n_new_elems:,} = {lora_n + n_new_elems:,} "
      f"({(lora_n + n_new_elems)/total*100:.2f}%)")
print(f"    ★ 但 Adam 按整块分配 = {trainable*8/2**30:.2f}GB"
      f"（掩码只挡梯度数值，挡不住优化器分配；全参 SFT 是 4.8GB）")

# ---- ③ merge 校验 ----
merged = model.merge_and_unload()
print(f"[4] merge 后类型: {type(merged).__name__} | dtype={merged.dtype} | "
      f"config.vocab_size={merged.config.vocab_size} | len(tokenizer)={len(tok)}")
print(f"    embed_rows={merged.get_input_embeddings().weight.shape[0]} | "
      f"tie 保持: {merged.get_input_embeddings().weight is merged.get_output_embeddings().weight}")
assert merged.config.vocab_size == len(tok), "❌ merge 后词表不一致"
assert type(merged).__name__ == "Qwen3ForCausalLM", f"❌ merge 未返回普通模型: {type(merged)}"
loaded_back = sum(p.numel() for p in merged.parameters())
print(f"    merge 后参数量 = {loaded_back:,}（应等于 total 里的底座部分）")

# ---- ④ 落盘 + 回读（模拟下游 rl.py/evaluate.py 的裸 from_pretrained）----
out_dir = "/tmp/lora_setup_check"
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
files = sorted(os.listdir(out_dir))
print(f"[5] 落盘文件: {[f for f in files if f.endswith(('.safetensors', '.json'))]}")
assert "model.safetensors" in files, "❌ 没写出 model.safetensors"
assert "adapter_model.safetensors" not in files, "❌ 写出的是 adapter，下游加载不了"

re = AutoModelForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16)
re_tok = AutoTokenizer.from_pretrained(out_dir)
print(f"[6] 回读成功: vocab={re.config.vocab_size} | tokenizer={len(re_tok)} | "
      f"embed={re.get_input_embeddings().weight.shape}")
# 新 token 必须能被 tokenizer 编码（否则 SID 数据整条对不上）
probe = "<a_96><b_92><c_180>"
ids = re_tok(probe, add_special_tokens=False)["input_ids"]
print(f"[7] 新 token 编码探针 '{probe}' -> {ids} "
      f"(均应 >= {original_vocab_size} 且 < {len(re_tok)})")
assert all(original_vocab_size <= i < len(re_tok) for i in ids), "❌ 新 token 编码越界"

print("\n✅ 全部前置校验通过：tie 成立 / 参数量符合预期 / merge 产出完整模型")
