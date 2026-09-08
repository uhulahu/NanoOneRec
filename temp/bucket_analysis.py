#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seen/unseen × 3/4-token 分桶分析 + SFT→RL 逐样本迁移 + 配对 bootstrap（2026-09-03）

回答三个问题：
  Q1 内容语义（LLM 对齐）能否让"训练交互中从未见过的 target item"被召回？
  Q2 extra token（<d_x> 身份消歧）是否构成命中瓶颈？
  Q3 RL 相对 SFT 的净提升内部结构：救回/损失各多少（逐样本配对）；first-diff 的收益
     是否来自中深层路由；浅层覆盖收缩造成多少损失；单 seed 提升在配对 bootstrap 下是否稳定。

分桶轴：
  seen  = target item 是否出现在 train 交互（train csv history_item_sid ∪ item_sid）
  tok4  = target SID 长度 4（含 extra <d_x>，即碰撞桶成员）
  9 桶：all / seen / unseen / tok3 / tok4 / S3 / S4 / U3 / U4

每桶指标（行=模型）：
  n 样本、n 商品、HR@1/5/10/20/50、NDCG@10/20/50（口径同 calc.py）、
  d1..d4 top1/top50 前缀存活（denom=T>=d，口径同 prefix_survival.py）、exact(=HR@50)、stop 正确率。

迁移（--sft-json 与各模型逐样本配对，先校验 input 序列一致）：
  K∈{1,10,50} 的 2×2（both/RLnew/RLlost/neither）+ 配对 bootstrap ΔHR@K、ΔNDCG@K 95%CI。

用法：
  python temp/bucket_analysis.py \
      --index data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
      --train-csv "data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/train/*.csv" \
      --sft-json results/final_checkpoint/final_result_Industrial_and_Scientific.json \
      --model fd750=results/run_20260902_211136_checkpoint-750/final_result_Industrial_and_Scientific.json \
      --out results/bucket_analysis.txt
"""
import argparse
import ast
import csv
import glob
import json
import re
import time

import numpy as np

PAT = re.compile(r"<[abcd]_\d+>")
HR_KS = [1, 5, 10, 20, 50]
NDCG_KS = [10, 20, 50]
BUCKETS = ["all", "seen", "unseen", "tok3", "tok4", "S3", "S4", "U3", "U4"]
MIG_KS = [1, 10, 50]


def load_index(path):
    """(full2id, leaf3)；item id 与 .inter/csv 一致。"""
    idx = json.load(open(path))
    full2id, leaf3 = {}, set()
    for iid, toks in idx.items():
        k = tuple(PAT.findall("".join(toks)))
        full2id[k] = iid
        if len(k) == 3:
            leaf3.add(k)
    return full2id, leaf3


def load_seen(train_csv_glob, full2id):
    """train 交互出现过的全部 item id（history ∪ target 的 sid → 反查 item id）。"""
    paths = glob.glob(train_csv_glob)
    assert paths, f"no train csv matched: {train_csv_glob}"
    seen = set()
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    for h in ast.literal_eval(row["history_item_sid"]):
                        k = tuple(PAT.findall(h))
                        if k in full2id:
                            seen.add(full2id[k])
                except Exception:
                    pass
                k = tuple(PAT.findall(row["item_sid"]))
                if k in full2id:
                    seen.add(full2id[k])
    return seen


class SampleMetrics:
    """单样本所有指标向量；T 供逐深度分母（T>=d）使用。"""

    __slots__ = ("seen", "need_d", "T", "hr", "ndcg", "exact", "full_alive", "t1", "t50")

    def __init__(self, output, predict, seen_flag):
        self.seen = seen_flag
        t = tuple(PAT.findall(output))
        self.T = len(t)
        self.need_d = self.T == 4
        beams = [tuple(PAT.findall(p)) for p in predict]
        hr = np.zeros(len(HR_KS), dtype=bool)
        ndcg = np.zeros(len(NDCG_KS))
        rank = None
        for j, g in enumerate(beams):
            if g == t:
                rank = j
                break
        if rank is not None:
            for i, k in enumerate(HR_KS):
                hr[i] = rank < k
            for i, k in enumerate(NDCG_KS):
                if rank < k:
                    ndcg[i] = 1.0 / np.log2(rank + 2)
        self.hr, self.ndcg = hr, ndcg
        self.exact = rank is not None and rank < 50
        self.full_alive = any(len(g) >= self.T and g[:self.T] == t for g in beams)
        t1 = np.zeros(4, dtype=bool)
        t50 = np.zeros(4, dtype=bool)
        if beams:
            b0 = beams[0]
            for d in range(1, min(self.T, 4) + 1):
                if len(b0) >= d and b0[:d] == t[:d]:
                    t1[d - 1] = True
                if any(len(g) >= d and g[:d] == t[:d] for g in beams):
                    t50[d - 1] = True
        self.t1, self.t50 = t1, t50


def load_json(path):
    return json.load(open(path))


def preprocess(json_path, full2id, leaf3, seen):
    """→ (samples, metas, metrics, n_unres)。metas[i] = item_id（不可解析=None）。"""
    samples = load_json(json_path)
    metas, metrics = [], []
    n_unres = 0
    for s in samples:
        t = tuple(PAT.findall(s["output"]))
        if len(t) == 4:
            iid = full2id.get(t)
        elif len(t) == 3 and t in leaf3:
            iid = full2id[t]
        else:
            iid = None
        if iid is None:
            n_unres += 1
        metas.append(iid)
        metrics.append(SampleMetrics(s["output"], s["predict"], iid is not None and iid in seen))
    return samples, metas, metrics, n_unres


def make_buckets(metas, metrics):
    """每桶样本 idx（桶集合稳定：同 test 集各模型一致）。"""
    members = {b: [] for b in BUCKETS}
    for i, sm in enumerate(metrics):
        if sm.seen and not sm.need_d:
            b = "S3"
        elif sm.seen:
            b = "S4"
        elif not sm.need_d:
            b = "U3"
        else:
            b = "U4"
        members[b].append(i)
        members["all"].append(i)
        members["seen" if sm.seen else "unseen"].append(i)
        members["tok3" if not sm.need_d else "tok4"].append(i)
    return {b: np.array(v, dtype=int) for b, v in members.items()}


def agg(metrics, idx):
    """桶聚合 → dict。前缀存活按 T>=d 归一（混合 T 的 marginal 桶分母不同）。"""
    n = len(idx)
    if n == 0:
        return None
    hr = np.zeros(len(HR_KS))
    ndcg = np.zeros(len(NDCG_KS))
    t1s = np.zeros(4)
    t50s = np.zeros(4)
    denom = np.zeros(4)
    exact = full_alive = 0
    for i in idx:
        sm = metrics[i]
        hr += sm.hr.astype(float)
        ndcg += sm.ndcg
        t1s += sm.t1.astype(float)
        t50s += sm.t50.astype(float)
        denom[:sm.T] += 1
        exact += sm.exact
        full_alive += sm.full_alive
    return {
        "n": n, "hr": hr / n, "ndcg": ndcg / n,
        "t1": t1s / np.maximum(denom, 1), "t50": t50s / np.maximum(denom, 1),
        "exact": exact / n, "stop": exact / full_alive if full_alive else float("nan"),
    }


def pct(v, nd=2):
    return f"{v*100:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=json_path")
    ap.add_argument("--sft-json", default=None)
    ap.add_argument("--out", default="results/bucket_analysis.txt")
    ap.add_argument("--bootstrap", type=int, default=1999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    full2id, leaf3 = load_index(args.index)
    print(f"[load] index: {len(full2id)} items")
    seen = load_seen(args.train_csv, full2id)
    print(f"[load] train-seen items: {len(seen)}")

    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        samples, metas, metrics, n_unres = preprocess(path, full2id, leaf3, seen)
        buckets = make_buckets(metas, metrics)
        models.append({"name": name, "path": path, "n": len(samples), "n_unres": n_unres,
                       "metas": metas, "metrics": metrics, "buckets": buckets,
                       "items": {b: {metas[i] for i in buckets[b]} for b in BUCKETS}})
        print(f"[load] {name}: {len(samples)} samples, {n_unres} unresolvable")

    sft = None
    if args.sft_json:
        samples, metas, metrics, n_unres = preprocess(args.sft_json, full2id, leaf3, seen)
        sft = {"name": "SFT", "samples": samples, "metas": metas, "metrics": metrics,
               "n_unres": n_unres}
        print(f"[load] SFT: {len(samples)} samples, {n_unres} unresolvable")

    out = []
    def emit(s=""):
        out.append(s)

    agg_cache = {}
    def get_agg(m, b):
        key = (m["name"], b)
        if key not in agg_cache:
            agg_cache[key] = agg(m["metrics"], m["buckets"][b])
        return agg_cache[key]

    # ================= 1. 分桶明细 =================
    emit("=" * 150)
    emit("分桶分析  (mean 变体 / test 集 / beam50 约束解码)")
    emit("口径: HR@K/NDCG@K 同 calc.py；d_k top1/top50 同 prefix_survival.py (denom=T>=k)；"
         "exact=HR@50；stop=exact/完整前缀存活")
    emit("=" * 150)
    col_w = [f"{m['name']:>17}" for m in models]
    for b in BUCKETS:
        idx0 = models[0]["buckets"][b]
        n, n_itm = len(idx0), len(models[0]["items"][b])
        if n == 0:
            continue
        emit(f"\n### bucket={b:<7}  n={n}  items={n_itm}")
        hdr = f"{'metric':<10}" + "".join(col_w)
        emit(hdr)
        rows = [
            ("HR@1", "hr", 0, pct), ("HR@5", "hr", 1, pct), ("HR@10", "hr", 2, pct),
            ("HR@20", "hr", 3, pct), ("HR@50", "hr", 4, pct),
            ("NDCG@10", "ndcg", 0, lambda v: f"{v:.4f}"), ("NDCG@20", "ndcg", 1, lambda v: f"{v:.4f}"),
            ("NDCG@50", "ndcg", 2, lambda v: f"{v:.4f}"),
            ("d1 top1", "t1", 0, pct), ("d1 top50", "t50", 0, pct),
            ("d2 top1", "t1", 1, pct), ("d2 top50", "t50", 1, pct),
            ("d3 top1", "t1", 2, pct), ("d3 top50", "t50", 2, pct),
            ("d4 top1", "t1", 3, pct), ("d4 top50", "t50", 3, pct),
            ("exact", "exact", None, pct), ("stop", "stop", None, pct),
        ]
        for label, key, j, fmt in rows:
            cells = []
            for m in models:
                a = get_agg(m, b)
                if a is None:
                    cells.append("-")
                else:
                    val = a[key] if j is None else a[key][j]
                    cells.append(fmt(val))
            emit(f"{label:<10}" + "".join(f"{c:>17}" for c in cells))
    emit("\n\n")

    # ================= 2. SFT→RL 迁移 + 配对 bootstrap =================
    if sft is None:
        emit("(无 --sft-json：跳过迁移/配对分析)")
    else:
        emit("=" * 150)
        emit(f"SFT→RL 逐样本迁移 + 配对 bootstrap 95%CI（seed={args.seed}, B={args.bootstrap}）")
        emit("=" * 150)
        sft_inputs = [s["input"] for s in sft["samples"]]
        if sft["n_unres"]:
            emit(f"[warn] SFT json 有 {sft['n_unres']} 样本无法解析——可能不是当前变体的评测，配对结果不可信")
        for m in models:
            mod_inputs = [s["input"] for s in load_json(m["path"])]
            order_ok = sft_inputs == mod_inputs
            if not order_ok:
                emit(f"\n### {m['name']}: input 序列与 SFT 不一致 → 跳过配对（需同 split 重新评测）")
                continue
            emit(f"\n### {m['name']} vs SFT  (行序一致)")
            for b in BUCKETS:
                idx = models[0]["buckets"][b]
                if len(idx) == 0:
                    continue
                n_itm = len(models[0]["items"][b])
                emit(f"  -- bucket {b}  n={len(idx)}  items={n_itm} --")
                hr_s = np.array([sft["metrics"][i].hr for i in idx], dtype=float)
                hr_m = np.array([m["metrics"][i].hr for i in idx], dtype=float)
                nd_s = np.array([sft["metrics"][i].ndcg for i in idx])
                nd_m = np.array([m["metrics"][i].ndcg for i in idx])
                for K in MIG_KS:
                    ki = HR_KS.index(K)
                    a, bb = hr_s[:, ki].astype(bool), hr_m[:, ki].astype(bool)
                    both = int((a & bb).sum())
                    new = int((~a & bb).sum())
                    lost = int((a & ~bb).sum())
                    none = int((~a & ~bb).sum())
                    emit(f"    K={K:<3} both={both:<6} RLnew={new:<6} RLlost={lost:<6} "
                         f"neither={none:<6} Δ={new - lost:+d} ({(new - lost) / len(idx) * 100:+.2f}pp)")
                d_hr = hr_m - hr_s
                d_nd = nd_m - nd_s
                rng = np.random.default_rng(args.seed)
                B = args.bootstrap
                res = rng.integers(0, len(idx), size=(B, len(idx)))
                bhr = d_hr[res].mean(axis=1)
                bnd = d_nd[res].mean(axis=1)
                lo_h, hi_h = np.percentile(bhr, 2.5, axis=0), np.percentile(bhr, 97.5, axis=0)
                lo_n, hi_n = np.percentile(bnd, 2.5, axis=0), np.percentile(bnd, 97.5, axis=0)
                for j, k in enumerate(HR_KS):
                    if k in MIG_KS:
                        emit(f"      ΔHR@{k:<2} = {d_hr[:, j].mean() * 100:+.2f}pp  "
                             f"95%CI [{lo_h[j] * 100:+.2f}, {hi_h[j] * 100:+.2f}]")
                for j, k in enumerate(NDCG_KS):
                    if k in (10, 50):
                        emit(f"      ΔNDCG@{k:<2} = {d_nd[:, j].mean():+.4f}  "
                             f"95%CI [{lo_n[j]:+.4f}, {hi_n[j]:+.4f}]")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[write] {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
