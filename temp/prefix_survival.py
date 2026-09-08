# -*- coding: utf-8 -*-
"""逐 token 前缀存活率分析（firstdiff 是否改善早期 SID 路由）

输入：evaluate_rl.sh 产出的 final_result json（含 output=target SID、predict=按序 50 条 beam）
对每条样本，目标 SID 序列 t = [<a_x>, <b_y>, ...]（长度 T=1..4）：
  depth d 存活  = 至少一条 beam 的前 d 个 SID token == t[:d]
  depth d top1  = rank0 的 beam 前 d 个 token == t[:d]（贪心前缀命中）
  full-SID 存活 = 某 beam 前缀含完整 t[:T]（不管它是否继续/停止）
  exact hit     = 某 beam 完整等于 target（含停止决策，即 calc.py 的 HR@50 口径）
  stop 正确率   = exact hit / full-SID 存活（前缀找对后，"停/续" 决策的对错）
逐深度只统计 T>=d 的样本（分母一致）。
"""
import json
import re
import sys

PAT = re.compile(r"<[abcd]_\d+>")


def sids(s):
    return PAT.findall(s.strip('\n" '))


def analyze(path):
    samples = json.load(open(path))
    n = len(samples)
    t_dist = {1: 0, 2: 0, 3: 0, 4: 0}          # 目标 SID 长度分布（各模型共用同一 test 集）
    denom = {1: 0, 2: 0, 3: 0, 4: 0}            # T>=d 的样本数
    hit_all = 0                                  # exact full-string hit（= HR@50 口径）
    full_alive = {1: 0, 2: 0, 3: 0, 4: 0}        # T>=d 且完整 t[:T=d?]...
    # 按深度聚合：仅当 T>=d 时统计
    top1_hit = {1: 0, 2: 0, 3: 0, 4: 0}          # rank0 前 d token == t[:d]
    alive5 = {1: 0, 2: 0, 3: 0, 4: 0}            # top-5 内有前缀存活
    alive50 = {1: 0, 2: 0, 3: 0, 4: 0}           # 50 条 beam 内前缀存活
    full_alive_T = 0                             # 完整 SID 前缀存活（T 级，不限停止）

    # 按 T 拆分统计（T=3 与 T=4 的漏斗形状可能不同）
    split = {3: {1: 0, 2: 0, 3: 0}, 4: {1: 0, 2: 0, 3: 0, 4: 0}}  # split[T][d] = d 级 top50 存活数

    for s in samples:
        t = sids(s["output"])
        T = len(t)
        if T < 1 or T > 4:
            continue
        t_dist[T] += 1
        beams = [sids(p) for p in s["predict"]]
        exact = any(g == t for g in beams)
        if exact:
            hit_all += 1
        # 记录某 beam 完整前缀= t（可能继续生成下级 token）
        full_alive = any(len(g) >= T and g[:T] == t for g in beams)
        if full_alive:
            full_alive_T += 1
        for d in range(1, T + 1):
            denom[d] += 1
            pref = t[:d]
            if beams and len(beams[0]) >= d and beams[0][:d] == pref:
                top1_hit[d] += 1
            if any(len(g) >= d and g[:d] == pref for g in beams[:5]):
                alive5[d] += 1
            if any(len(g) >= d and g[:d] == pref for g in beams):
                alive50[d] += 1
                if T in split and d in split[T]:
                    split[T][d] += 1

    out = {"n": n, "t_dist": t_dist, "hit_all": hit_all}
    for d in range(1, 5):
        out[f"d{d}"] = {
            "denom": denom[d],
            "top1": top1_hit[d] / denom[d] if denom[d] else None,
            "alive_top5": alive5[d] / denom[d] if denom[d] else None,
            "alive_top50": alive50[d] / denom[d] if denom[d] else None,
        }
    out["full_sid_alive"] = full_alive_T / n
    out["exact_hit"] = hit_all / n
    out["stop_correct_given_full_alive"] = hit_all / full_alive_T if full_alive_T else None
    # 条件存活：split[T][d] 已含全部 T 级样本 d 级存活的分子；
    # 存活的样本 ⊂ 上一级存活的样本，故条件存活 = alive[d] / alive[d-1]（d>=2）
    out["cond"] = {}
    for T in (3, 4):
        out["cond"][T] = {}
        for d in range(2, T + 1):
            prev = split[T][d - 1]
            out["cond"][T][d] = (split[T][d] / prev) if prev else None
    return out


def fmt(p):
    return f"{p*100:.1f}%" if p is not None else "  - "


if __name__ == "__main__":
    paths = sys.argv[1:]
    labels = [p.split("/results/")[-1].replace("/final_result_Industrial_and_Scientific.json", "") for p in paths]
    results = [analyze(p) for p in paths]

    # 表头
    header = "        " + "".join(f"{lab[:34]:>36}" for lab in labels)
    print(header)
    n0 = results[0]["n"]
    print(f"target 长度分布 (n={n0}): " + json.dumps(results[0]["t_dist"]))
    rows = []
    for d in range(1, 5):
        rows.append((f"d{d}: top1 前缀命中", [r["d" + str(d)]["top1"] for r in results]))
        rows.append((f"d{d}: top5 前缀存活", [r["d" + str(d)]["alive_top5"] for r in results]))
        rows.append((f"d{d}: top50 前缀存活", [r["d" + str(d)]["alive_top50"] for r in results]))
    rows.append(("完整 SID 前缀存活(T级)", [r["full_sid_alive"] for r in results]))
    rows.append(("exact hit (HR@50 口径)", [r["exact_hit"] for r in results]))
    rows.append(("停止正确率(前缀活→整串)", [r["stop_correct_given_full_alive"] for r in results]))
    rows.append(("T3 条件存活 d2|d1", [r["cond"][3][2] for r in results]))
    rows.append(("T3 条件存活 d3|d2", [r["cond"][3][3] for r in results]))
    rows.append(("T4 条件存活 d2|d1", [r["cond"][4][2] for r in results]))
    rows.append(("T4 条件存活 d3|d2", [r["cond"][4][3] for r in results]))
    rows.append(("T4 条件存活 d4|d3", [r["cond"][4][4] for r in results]))
    for name, vals in rows:
        print(f"{name:<24}" + "".join(f"{fmt(v):>36}" for v in vals))
