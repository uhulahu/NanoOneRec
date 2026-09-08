#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zero2（transformers 原生 deepspeed 集成）配套 monkeypatch —— sft.py / rl.py 共用（2026-09-07）

背景（详见 docs/PROGRESS.md 2026-09-07 条目）：
1. transformers Trainer 在 ds 启用时，ckpt 保存的"优化器/调度器"一步被改道为 ds engine 的
   save_checkpoint → 写 global_step*/（zero2 分片后的 fp32 优化器状态 ≈7.8G/次，0.6B 全参），
   ckpt 目录共 ~9.2G。本机单卷 110G 放不下 best+newest 双目录 + 写入瞬态 → 磁盘写满崩。
2. 同时 transformers 的 load_best_model_at_end 在 ds 分支强制 deepspeed_load_checkpoint
   （需要 engine 文件）→ 引擎文件被跳过保存后，训练正常早停却在收尾时崩。

修复：ds 启用时
  a) 跳过 engine 的 optimizer 落盘 → ckpt 目录仅 model.safetensors+tokenizer+trainer_state（~1.5G）。
     模型权重由 _save_checkpoint 前段的 save_model 正常写，eval/from_pretrained 不受影响；
     resume 只恢复权重（与 sft.py 既有"跳过 optimizer 加载"的哲学一致）。
  b) 末端 best 恢复不走 deepspeed_load_checkpoint，改各 rank 直接 HF 加载 best 目录的
     model.safetensors（zero2 每 rank 持全量权重，各自加载同一文件即可，瞬态 +1.2G/rank）。

非 ds 路径两个补丁都原样回退；apply_ds_zero2_patches() 幂等，sft.py 与 rl.py 顶部各调一次。
注意：此设计专为 zero2（参数每 rank 全量）；zero3 参数分片，以上假设不成立，勿在 zero3 下用。
"""
import os
import time

import transformers.trainer as _tt


_APPLIED = False


def make_run_dir_name(prefix: str = "run") -> str:
    """rank0 生成 run 目录名并广播给全体 rank（2026-09-08 事故修复）。

    此前 sft.py/rl.py 各自 time.strftime（秒级）拼 run_<ts>：4 个 rank 启动恰跨秒时 output_dir
    不一致 → rank 间 trainer_state/ckpt 目录错位，末端 best 恢复崩溃（消融首跑事故）。"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            name = time.strftime("%Y%m%d_%H%M%S")
        else:
            name = None
        names = [name]
        dist.broadcast_object_list(names, src=0)
        name = names[0]
    else:
        name = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{name}"


def apply_ds_zero2_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    _orig_save = _tt.Trainer._save_optimizer_and_scheduler

    def _skip_ds_engine_optimizer_save(self, output_dir):
        if getattr(self, "is_deepspeed_enabled", False):
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print("[MONITORING] zero2: 跳过 ds engine optimizer checkpoint 落盘（省 ~7.8G/次）", flush=True)
            return
        return _orig_save(self, output_dir)

    _tt.Trainer._save_optimizer_and_scheduler = _skip_ds_engine_optimizer_save

    _orig_load_best = _tt.Trainer._load_best_model

    def _load_best_model_hf_for_zero2(self):
        if not (self.args.load_best_model_at_end and self.state.best_model_checkpoint):
            return
        if not getattr(self, "is_deepspeed_enabled", False):
            return _orig_load_best(self)
        bpath = self.state.best_model_checkpoint
        model_file = os.path.join(bpath, "model.safetensors")
        if not os.path.isfile(model_file):
            print(f"[MONITORING] best 目录无 model.safetensors（{bpath}），回退 transformers 默认", flush=True)
            return _orig_load_best(self)
        from safetensors.torch import load_file
        sd = load_file(model_file)
        self.accelerator.unwrap_model(self.model).load_state_dict(sd, strict=True)
        del sd
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[MONITORING] zero2: best 权重已从 {bpath} HF 直载", flush=True)

    _tt.Trainer._load_best_model = _load_best_model_hf_for_zero2

    _APPLIED = True
