#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从指定的 LoRA checkpoint 合并出完整 HF 模型目录 —— 逐行复刻 sft.py 的 merge 路径。

为什么需要（2026-09-16）：
  E5（码本打乱对照）跑到 2602/2760 时服务器意外关机。续跑时发现代码库里有一条既有补丁：
      [MONITORING] 跳过 optimizer/scheduler 加载（显存限制），仅恢复模型权重
  → **resume 是有损的**：优化器状态与 scheduler 都不恢复，warmup 重跑、lr 从 0 重新爬到 1e-4
    （实测 0 → 5e-06 → 1e-05 …，而正确值应是从 1.007e-05 继续衰减）。
  若让续跑跑完，末尾 276 步会跑在 ~1e-04（高 10 倍），污染结果；且 save_total_limit=4 会把
  干净的 best（checkpoint-2208）轮换删掉。
  → 故停掉续跑，直接从**停机前训练出的干净 best**（step 2208, eval_loss 2.4043）合并。
  这与 trainer 在 load_best_model_at_end=True 下的选择一致（best 之后 2346/2484 均未改善）。

用法：python temp/merge_lora_ckpt.py <checkpoint目录> <输出目录>
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sft import TokenExtender

BASE = "data/pretrained_model/Qwen3-0.6B"
SID  = "data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json"

def main(ckpt, out):
    # === 与 sft.py 完全一致的 tokenizer 构建（行 373-376, 383-391）===
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    te = TokenExtender(data_path=os.path.dirname(SID),
                       dataset=os.path.basename(SID).split(".")[0])
    new_tokens = te.get_new_tokens()
    tok.add_tokens(new_tokens)
    print(f"tokenizer 词表 = {len(tok)}（新增 {len(new_tokens)}）")

    # === 与 sft.py 一致的 base + resize ===
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))

    # === 载入 adapter（含 modules_to_save 的 embedding 行）并合并 —— 等价 trainer.model.merge_and_unload() ===
    peft_model = PeftModel.from_pretrained(model, ckpt)
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True                         # 部署产物是推理用的：开 KV cache
                                                           # （sft.py:685 已于 2026-09-17 从 False 修正为 True）

    got = merged.get_input_embeddings().weight.shape
    assert merged.config.vocab_size == len(tok), f"vocab 不一致 {merged.config.vocab_size} vs {len(tok)}"
    assert got[0] == len(tok), f"embedding 行数 {got[0]} != 词表 {len(tok)}"
    print(f"merge 完成：vocab={merged.config.vocab_size} embedding={tuple(got)} dtype={merged.dtype}")

    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print("已写出:", out)
    for f in sorted(os.listdir(out)):
        print("   ", f)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
