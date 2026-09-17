#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：HF Trainer 的 bf16=True 到底做了什么？（2026-09-14）

要回答三个可证伪的问题：
  Q1. bf16=True 会不会自动把 fp32 的模型转成 bf16？
  Q2. 训练时 forward 内部到底有没有 autocast 上下文？
  Q3. 如果有，autocast 的 dtype 是什么？

做法：用 fp32 建一个玩具模型，在 forward 里探测运行时状态。
关键判据是「**模型参数**的 dtype」和「**forward 运行时**的 autocast 状态」——
这两个是分开的两件事，文档里那句 "mixed precision" 把它们含混在一起了。
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, ".")
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

RECORD = {}


class Probe(nn.Module):
    """夹在 embedding 后面的探针：记录 forward 运行时的精度环境。"""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x):
        if "fwd" not in RECORD:
            import traceback
            RECORD["fwd"] = {
                "is_autocast_enabled": torch.is_autocast_enabled(),
                "autocast_dtype": str(torch.get_autocast_dtype("cuda")) if torch.is_autocast_enabled() else "—",
                "input_ids_dtype": str(x.dtype),
                # 谁开的 autocast？保留所有帧，打印时再过滤
                "stack": [f"{fr.filename.split('site-packages/')[-1]}:{fr.lineno} {fr.name}"
                          for fr in traceback.extract_stack()],
            }
        return self.wrapped(x)


class Toy(Dataset):
    def __init__(self, n=64, L=16):
        self.n, self.L = n, L

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ids = [(i * 7 + j) % 100 for j in range(self.L)]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(ids),
                "attention_mask": torch.ones(self.L, dtype=torch.long)}


def main():
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    # ⚠️ 关键：故意用 fp32 建模型（不加 dtype=torch.bfloat16）
    model = LlamaForCausalLM(cfg)
    p = next(model.parameters())
    print(f"建模型时（不指定 dtype）: 参数 dtype = {p.dtype}")
    assert p.dtype == torch.float32

    model.model.embed_tokens = Probe(model.model.embed_tokens)

    args = TrainingArguments(
        output_dir="temp/_probe_out", bf16=True, max_steps=1,
        per_device_train_batch_size=4, report_to=[], logging_steps=1,
        save_strategy="no", use_cpu=False, gradient_accumulation_steps=1,
    )
    try:  # accelerate 1.x 不再从顶层导出 AcceleratorState
        from accelerate.state import AcceleratorState
        print(f"accelerate 读到的 mixed_precision = {AcceleratorState._shared_state.get('mixed_precision')}")
    except Exception as e:  # noqa: BLE001
        print(f"(读 AcceleratorState 失败: {type(e).__name__})")
    print(f"ACCELERATE_MIXED_PRECISION 环境变量 = {os.environ.get('ACCELERATE_MIXED_PRECISION')}")

    tr = Trainer(model=model, args=args, train_dataset=Toy())
    print(f"Trainer 构造后: use_cpu_amp={tr.use_cpu_amp} use_apex={tr.use_apex}")
    print(f"                 accelerator.native_amp={tr.accelerator.native_amp}")

    tr.train()

    print()
    print("=" * 62)
    print("训练结束后的实测结果")
    print("=" * 62)
    p_after = next(tr.model.parameters())
    print(f"Q1 参数 dtype: 训练前 fp32 → 训练后 {p_after.dtype}"
          f"   {'❌ 没转（Trainer 不改模型 dtype）' if p_after.dtype == torch.float32 else '✅ 转了'}")
    print(f"Q2 forward 时 autocast 开启? {RECORD['fwd']['is_autocast_enabled']}"
          f"   {'❌ 训练时没有 autocast' if not RECORD['fwd']['is_autocast_enabled'] else ''}")
    print(f"Q3 autocast dtype: {RECORD['fwd']['autocast_dtype']}")
    print(f"   input_ids dtype: {RECORD['fwd']['input_ids_dtype']}")
    print("   调用栈（只显示 trainer/accelerate 的帧 = 谁开的 autocast）：")
    for line in RECORD["fwd"]["stack"]:
        if any(k in line for k in ("trainer.py", "accelerate/", "trainer_", "sft.py", "probe_bf16")):
            print(f"     {line}")

    print()
    print("为何 Trainer 里找不到 autocast？—— accelerate 把 forward 整体替换了：")
    f = tr.model.forward
    for depth in range(5):
        print(f"     {'  ' * depth}{type(f).__name__}  ({getattr(f, '__qualname__', '')})")
        f = getattr(f, "__wrapped__", None) or getattr(f, "__func__", None)
        if f is None:
            break
    print(f"     model._original_forward 存在? {hasattr(tr.model, '_original_forward')}")


if __name__ == "__main__":
    main()
