# -*- coding: utf-8 -*-
"""_masked_column_advantages 语义单测（想法 (b)(c)，见 docs/RL_IDEAS.md）

构造 num_groups=2、G=4、C=3（两列 SID + 一列 EOS）的监督矩阵，验证：
  1. group 模式混合列有对比（z 非零且符号正确）
  2. group 模式"整组同列全 -1"（全组同位置分歧）→ z 恰为 0（原实现的抵消病理）
  3. (b) all_wrong_penalty：上述病理列变为 -λ，且只作用于该 (组,列)
  4. (c) column 模式：A 组全错、B 组全对的列恢复跨组对比（A 负 B 正）
  5. 屏蔽位恒 0（两种模式 + 惩罚后）
  6. (c)+(b) 叠加不冲突
  7. 非法 mode 抛 ValueError
"""
import sys

sys.path.insert(0, "/root/autodl-tmp/MiniOneRec-main")
import torch
from minionerec_trainer import ReReTrainer

f = ReReTrainer._masked_column_advantages

# 布局: groups=2, G=4, C=3
# group0: 列0 全 -1（全组在位置0分歧——病理）| 列1 混合(+1,+1,-1,-1) | 列2 全 -1（全组停止位分歧）
# group1: 列0 全 +1（全组前缀对——常量，无对比）| 列1 混合 | 列2 无监督(全 0 mask)
g0 = [[-1, +1, -1], [-1, +1, -1], [-1, -1, -1], [-1, -1, -1]]
m0 = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
g1 = [[+1, +1, 0], [+1, +1, 0], [+1, -1, 0], [+1, -1, 0]]
m1 = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
scores = torch.tensor([g0, g1], dtype=torch.float32)
masks = torch.tensor([m0, m1], dtype=torch.float32)

ok = True


def check(name, cond, detail=""):
    global ok
    mark = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{mark}] {name}  {detail}")


# ---- 1. group 混合列有对比 ----
adv = f(scores, masks, mode="group")
col1_g0 = adv[0, :, 1]           # 列1 组0: +1,+1,-1,-1
check("group 混合列 z 非零且 ± 对称",
      torch.allclose(col1_g0, torch.tensor([1.0, 1.0, -1.0, -1.0]), atol=2e-3),
      f"z={col1_g0.tolist()}")

# ---- 2. group 全 -1 列 → 0（病理）----
col0_g0 = adv[0, :, 0]
check("group 全组同列 -1 → z=0（抵消病理）",
      torch.allclose(col0_g0, torch.zeros(4), atol=1e-6), f"z={col0_g0.tolist()}")
col0_g1 = adv[1, :, 0]           # 全 +1 常量也 0
check("group 全 +1 常量列 → z=0",
      torch.allclose(col0_g1, torch.zeros(4), atol=1e-6), f"z={col0_g1.tolist()}")

# ---- 3. (b) 全错列惩罚 ----
adv_b = f(scores, masks, mode="group", all_wrong_penalty=2.0)
check("(b) 全 -1 列 → -λ 且列内一致",
      torch.allclose(adv_b[0, :, 0], torch.full((4,), -2.0), atol=1e-6),
      f"z={adv_b[0, :, 0].tolist()}")
check("(b) 混合列不受惩罚影响（仍是 ±1）",
      torch.allclose(adv_b[0, :, 1], torch.tensor([1.0, 1.0, -1.0, -1.0]), atol=2e-3),
      f"z={adv_b[0, :, 1].tolist()}")
check("(b) 全 +1 列不加惩罚（仍是 0）",
      torch.allclose(adv_b[1, :, 0], torch.zeros(4), atol=1e-6), f"z={adv_b[1, :, 0].tolist()}")
check("(b) 无监督列（group1 列2）保持 0 不受惩罚",
      torch.allclose(adv_b[1, :, 2], torch.zeros(4), atol=1e-6), f"z={adv_b[1, :, 2].tolist()}")

# ---- 4. (c) column 模式：跨组对比恢复 ----
adv_c = f(scores, masks, mode="column")
# 列0 所有监督值 = group0 全 -1(4条) + group1 全 +1(4条) → mean 0, std 1 → ±1
c0 = adv_c[:, :, 0]
check("(c) 跨组列 z：全错组为负、全对组为正",
      torch.allclose(c0[0], torch.full((4,), -1.0), atol=2e-3) and
      torch.allclose(c0[1], torch.full((4,), +1.0), atol=2e-3),
      f"g0={c0[0, 0].item():+.3f} g1={c0[1, 0].item():+.3f}")
# 列1 两组合并: g0(+1,+1,-1,-1) g1(+1,+1,-1,-1) → 与组内一致
c1_g0 = adv_c[0, :, 1]
check("(c) 混合列跨组结果与组内一致",
      torch.allclose(c1_g0, torch.tensor([1.0, 1.0, -1.0, -1.0]), atol=2e-3),
      f"z={c1_g0.tolist()}")

# ---- 5. 屏蔽位恒 0（c 模式下 group1 列2 全屏蔽）----
check("屏蔽位恒 0（column 模式）",
      torch.allclose(adv_c[1, :, 2], torch.zeros(4), atol=1e-8))
check("屏蔽位恒 0（group+惩罚）",
      torch.allclose(adv_b[1, :, 2], torch.zeros(4), atol=1e-8))

# ---- 6. (c)+(b) 叠加：跨组已恢复对比的列加惩罚后更负 ----
adv_cb = f(scores, masks, mode="column", all_wrong_penalty=1.0)
check("(c)+(b)：全错组列0 = -1 - λ",
      torch.allclose(adv_cb[0, :, 0], torch.full((4,), -2.0), atol=2e-3),
      f"z={adv_cb[0, :, 0].tolist()}")
check("(c)+(b)：全对组列0 = +1 不受惩罚",
      torch.allclose(adv_cb[1, :, 0], torch.full((4,), +1.0), atol=2e-3))

# ---- 7. 非法 mode ----
try:
    f(scores, masks, mode="bogus")
    check("非法 mode 抛 ValueError", False)
except ValueError:
    check("非法 mode 抛 ValueError", True)

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
