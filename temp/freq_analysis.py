#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""target 商品在训练交互中的出现频率分桶（2026-09-04）

unseen（0 次交互）= 冷启动，已做专门分桶；本分析把 seen 按训练交互频次细分，
检验"长尾改善"：RL 相对 SFT 的提升是否偏向低频商品，高频段是否反而退化。

频次口径（train csv 每行 = 同一用户序列前缀展开快照：history 前 k 个交互 + target 第 k+1 个，
history 列只保留最近 10 个，长序列是滑动窗口）：
  cnt_all = 按用户重建真实交互序列后每位置计一次（早位置不被后续行重复累计；连续重复购买为独立
            位置。重建正确性：last(H_{k+1})==T_k 不变量全量 0/109269 违例）
  cnt_tgt = 仅作为 target 出现的次数（辅助口径，= 除每用户首位置外全部位置）
  bucket 依据 cnt_all；0 次 = unseen（冷启动对照，同先前分析口径）

每桶：行级 HR@1/5/10/20/50、NDCG@50（同 calc.py）+ 商品级 macro HR@1/10/50
（商品内样本均值 → 商品等权，避免高频商品行级主导，与 item_level_analysis.py 一致）；
Δ(SFT→RL)：行级配对 bootstrap 95%CI（样本非独立，仅供参照）+ item-cluster bootstrap +
new/lost/both 商品数（K=50，逐商品 anyhit）。

用法：
  python temp/freq_analysis.py \
      --index data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/Industrial_and_Scientific.index.json \
      --train-csv "data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-mean-20260830/train/*.csv" \
      --sft-json results/outputs_final_checkpoint/final_result_Industrial_and_Scientific.json \
      --model fd750=results/run_20260902_211136_checkpoint-750/final_result_Industrial_and_Scientific.json \
      --out results/freq_analysis.txt
"""
import argparse
import ast
import csv
import glob
import json
import re
import sys
import time

import numpy as np

sys.path.insert(0, "temp")
from bucket_analysis import PAT, HR_KS, load_index, preprocess  # noqa: E402

# 频率桶：左闭右闭；None = 无上限
FREQ_BUCKETS = [
    ("F0", 0, 0),          # unseen（冷启动对照）
    ("F1", 1, 1),
    ("F2", 2, 2),
    ("F3-4", 3, 4),
    ("F5-9", 5, 9),
    ("F10-24", 10, 24),
    ("F25-99", 25, 99),
    ("F100+", 100, None),
]
NDCG_KS = [10, 20, 50]
MIG_KS = [1, 10, 50]


def freq_label(f):
    for label, lo, hi in FREQ_BUCKETS:
        if (hi is None and f >= lo) or (hi is not None and lo <= f <= hi):
            return label
    return None


def load_freq(train_csv_glob, idx_ids=None):
    """按用户重建真实 train 交互序列、每交互位置计一次（2026-09-04 修正版）。

    旧口径逐行累计 history 有前缀污染：csv 行 = 同一用户前缀展开的快照（第 k 行 history = 前 k 个
    交互、target = 第 k+1 个），早位置被后续每行重复携带。且 history 列**只保留最近 10 个**：
    前缀长到 10 后行内是滑动窗口（2205/20027 个超长用户的最长行只是窗口，不能当"完整序列"取）。
    修正口径：不变量 last(H_{k+1}) == T_k 全量验证 0/109269 违例 → 行序即时序、target 链即真实
    序列；每用户 S = 首行 history + 各行 target，每位置计一次（连续重复购买 = 独立位置，如实计数）。
    16 行完全重复的 (history,target) 快照按一行去重。
    """
    paths = glob.glob(train_csv_glob)
    assert paths, f"no train csv matched: {train_csv_glob}"
    rows_by_user = {}
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                rows_by_user.setdefault(row["user_id"], []).append(row)
    cnt_all, cnt_tgt = {}, {}
    n_pos = n_dup = n_viol = 0
    for rs in rows_by_user.values():
        seq = []
        prev = None
        for k, row in enumerate(rs):
            hist = [str(x) for x in ast.literal_eval(row["history_item_id"])]
            tgt = row["item_id"]
            if prev is not None and hist == prev[0] and tgt == prev[1]:
                n_dup += 1          # 完全重复的快照行：同一交互，只计一次
                continue
            if k == 0:
                seq.extend(hist)    # 首行 = 用户序列起点（已全量验证 len==1）
            seq.append(tgt)
            if k + 1 < len(rs):     # 不变量审计：本行 target 应为下一行 history 的末位
                nxt = [str(x) for x in ast.literal_eval(rs[k + 1]["history_item_id"])]
                if not nxt or nxt[-1] != tgt:
                    n_viol += 1
            prev = (hist, tgt)
        n_pos += len(seq)
        for j, iid in enumerate(seq):
            if idx_ids is not None and iid not in idx_ids:
                continue
            cnt_all[iid] = cnt_all.get(iid, 0) + 1
            if j > 0:               # 除序列首位置外，每位置都是一行的 target
                cnt_tgt[iid] = cnt_tgt.get(iid, 0) + 1
    print(f"[load_freq] users={len(rows_by_user)} 序列总位置={n_pos} "
          f"(首行起点 {sum(1 for v in rows_by_user.values())}，去重行 {n_dup}) 不变量违例={n_viol}")
    return cnt_all, cnt_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--sft-json", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=json_path")
    ap.add_argument("--out", default="results/freq_analysis.txt")
    ap.add_argument("--bootstrap", type=int, default=1999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    full2id, leaf3 = load_index(args.index)
    cnt_all, cnt_tgt = load_freq(args.train_csv, set(full2id.values()))
    print(f"[load] index {len(full2id)} items, train 出现过 {len(cnt_all)} 个商品")

    _, sft_metas, sft_metrics, n_unres = preprocess(args.sft_json, full2id, leaf3, set())
    print(f"[load] SFT: {len(sft_metas)} samples, {n_unres} unresolvable")
    assert n_unres == 0, "SFT json 有无法解析样本——可能不是 mean 变体评测"

    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        _, metas, metrics, nu = preprocess(path, full2id, leaf3, set())
        models.append({"name": name, "path": path, "n": len(metas), "n_unres": nu,
                       "metrics": metrics})
        print(f"[load] {name}: {len(metas)} samples, {nu} unresolvable")
    n = len(sft_metas)
    assert all(m["n"] == n for m in models), "模型 json 样本数不一致"

    # ---- 样本 → 频率桶（基于 SFT metas：target 稳定，桶成员各模型一致） ----
    labels = [b[0] for b in FREQ_BUCKETS]
    members = {lab: [] for lab in labels}
    for i, iid in enumerate(sft_metas):
        if iid is None:
            continue
        f = cnt_all.get(iid, 0)
        lab = freq_label(f)
        if lab is not None:
            members[lab].append(i)
    members = {lab: np.array(v, dtype=int) for lab, v in members.items()}

    # ---- 每商品样本 idx（macro + cluster bootstrap 用） ----
    item2idx = {}
    for i, iid in enumerate(sft_metas):
        if iid is not None:
            item2idx.setdefault(iid, []).append(i)
    item_ids = sorted(item2idx.keys(), key=int)
    j_of = {iid: j for j, iid in enumerate(item_ids)}
    J = len(item_ids)

    def bucket_of_item(iid):
        f = cnt_all.get(iid, 0)
        for lab, lo, hi in FREQ_BUCKETS:
            if (hi is None and f >= lo) or (hi is not None and lo <= f <= hi):
                return lab
        return None

    item_bucket = np.array([bucket_of_item(i) for i in item_ids])
    item_freq = np.array([cnt_all.get(i, 0) for i in item_ids])

    # per-model 商品级向量（macro = 商品内样本 hr 均值；any = 商品内任一命中）
    vecs = {"SFT": None}
    ki_of = {k: HR_KS.index(k) for k in MIG_KS}
    for m in models:
        name = m["name"]
        vecs[name] = None

    def build_vecs(metrics):
        macro = {k: np.zeros(J) for k in MIG_KS}
        anyhit = {k: np.zeros(J, dtype=bool) for k in MIG_KS}
        for j, iid in enumerate(item_ids):
            idxs = item2idx[iid]
            for k in MIG_KS:
                h = np.array([metrics[i].hr[ki_of[k]] for i in idxs])
                macro[k][j] = h.mean()
                anyhit[k][j] = h.any()
        return {"macro": macro, "any": anyhit}

    sft_vecs = build_vecs(sft_metrics)
    model_vecs = {m["name"]: build_vecs(m["metrics"]) for m in models}

    def members_mask(b):
        return item_bucket == b

    rng = np.random.default_rng(args.seed)
    B = args.bootstrap
    out = []
    def emit(s=""):
        out.append(s)

    # ================= 0. 频率分布 =================
    emit("=" * 140)
    emit(f"频率分桶  (mean 变体 / test 集 {n} 样本 / beam50 约束解码 / train 交互总商品 {len(cnt_all)})")
    emit(f"频次口径: cnt_all=history∪target 逐项计数（分桶依据）；cnt_tgt=仅 target 计数（报告均值对照）")
    emit("=" * 140)
    emit(f"{'bucket':<8}{'freq范围':<10}{'n样本':>8}{'n商品':>8}{'均cnt_tgt':>10}"
         f"{'样本均cnt_all':>13}{'tok4样本占比':>12}")
    for lab, lo, hi in FREQ_BUCKETS:
        idx = members[lab]
        if len(idx) == 0:
            emit(f"{lab:<8}{f'{lo}-{hi if hi is not None else ""}':<10}{0:>8}{0:>8}")
            continue
        ids = [sft_metas[i] for i in idx]
        n_it = len(set(ids))
        mtgt = np.mean([cnt_tgt.get(i, 0) for i in ids])
        mall = np.mean([cnt_all.get(i, 0) for i in ids])
        tok4 = sum(1 for i in idx if sft_metrics[i].need_d) / len(idx)
        emit(f"{lab:<8}{f'{lo}-{hi if hi is not None else ""}':<10}{len(idx):>8}{n_it:>8}"
             f"{mtgt:>10.2f}{mall:>13.2f}{tok4*100:>11.1f}%")

    # ================= 1. 分桶明细：行级 + 商品级 =================
    emit("\n\n" + "=" * 140)
    emit("分桶明细：行级 HR@K / NDCG@50（样本等权，高频商品行多贡献多）")
    emit("+ 商品级 macro HR@K（商品内样本均值 → 商品等权，频次效应下的稳健口径）")
    emit("=" * 140)
    all_names = ["SFT"] + [m["name"] for m in models]
    W = max(len(x) for x in all_names) + 2
    col_hdr = "".join(f"{x:>{W}}" for x in all_names)
    for lab, lo, hi in FREQ_BUCKETS:
        idx = members[lab]
        if len(idx) == 0:
            continue
        n_it = len({sft_metas[i] for i in idx})
        emit(f"\n### bucket {lab} (train 交互 {lo}-{hi if hi is not None else ''} 次)"
             f"  n样本={len(idx)}  n商品={n_it}")
        # 行级指标矩阵
        row = {}
        for name, metrics in [("SFT", sft_metrics)] + [(m["name"], m["metrics"]) for m in models]:
            hr = np.zeros(len(HR_KS))
            ndcg = np.zeros(len(NDCG_KS))
            for i in idx:
                sm = metrics[i]
                hr += sm.hr.astype(float)
                ndcg += sm.ndcg
            row[name] = {"hr": hr / len(idx), "ndcg": ndcg / len(idx)}
        # macro 矩阵（桶内**去重商品**等权：商品内样本均值 → 商品等权）
        # 注意不能用样本展开索引：同商品多次出现会把均值拉回行级口径（mac≡row 是病征）
        it_ids_b = sorted({sft_metas[i] for i in idx})
        memb_j = np.array([j_of[iid] for iid in it_ids_b], dtype=int)
        mac = {}
        for name in all_names:
            v = sft_vecs if name == "SFT" else model_vecs[name]
            mac[name] = {k: v["macro"][k][memb_j].mean() for k in MIG_KS}
        emit(f"{'metric':<12}{col_hdr}")
        for ki, k in enumerate(HR_KS):
            emit(f"{'HR@' + str(k):<12}" + "".join(f"{row[x]['hr'][ki]*100:>{W}.2f}%" for x in all_names))
        for j, k in enumerate(NDCG_KS):
            if k == 50:
                emit(f"{'NDCG@' + str(k):<12}" + "".join(f"{row[x]['ndcg'][j]:>{W}.4f}" for x in all_names))
        emit(f"{'macHR@1':<12}" + "".join(f"{mac[x][1]*100:>{W}.2f}%" for x in all_names))
        emit(f"{'macHR@10':<12}" + "".join(f"{mac[x][10]*100:>{W}.2f}%" for x in all_names))
        emit(f"{'macHR@50':<12}" + "".join(f"{mac[x][50]*100:>{W}.2f}%" for x in all_names))

    # ================= 2. Δ(SFT→RL)：行级配对 bootstrap + item cluster =================
    emit("\n\n" + "=" * 140)
    emit(f"Δ(SFT→RL) 每桶 (seed={args.seed}, B={B})")
    emit("行级配对 ΔHR@K + 95%CI：样本非独立（同商品多行），CI 偏窄，仅供参照；")
    emit("商品级 ΔmacHR@50 + item-cluster bootstrap 95%CI 与 new/lost/both 商品数(K=50) 为稳健口径")
    emit("=" * 140)
    sft_inputs = [s["input"] for s in json.load(open(args.sft_json))]
    for m in models:
        mod_inputs = [s["input"] for s in json.load(open(m["path"]))]
        emit(f"\n### {m['name']} vs SFT  {'(行序一致)' if sft_inputs == mod_inputs else '(行序不一致→配对不可信, 跳过)'}")
        if sft_inputs != mod_inputs:
            continue
        mm = m["metrics"]
        for lab, lo, hi in FREQ_BUCKETS:
            idx = members[lab]
            if len(idx) == 0:
                continue
            n_it = len({sft_metas[i] for i in idx})
            # 行级
            hr_s = np.array([sft_metrics[i].hr for i in idx], dtype=float)
            hr_m = np.array([mm[i].hr for i in idx], dtype=float)
            d = hr_m - hr_s
            # 商品级（去重商品索引；nIt 行已报样本展开数，此处必须以商品为单位）
            it_ids_b = sorted({sft_metas[i] for i in idx})
            memb_j = np.array([j_of[iid] for iid in it_ids_b], dtype=int)
            d_m = model_vecs[m["name"]]["macro"][50][memb_j] - sft_vecs["macro"][50][memb_j]
            draws = rng.integers(0, len(memb_j), size=(B, len(memb_j)))
            lo_b, hi_b = np.percentile(d_m[draws].mean(axis=1), 2.5), np.percentile(d_m[draws].mean(axis=1), 97.5)
            sft_any = sft_vecs["any"][50][memb_j]
            any50 = model_vecs[m["name"]]["any"][50][memb_j]
            newc = int(((~sft_any) & any50).sum())
            lostc = int((sft_any & (~any50)).sum())
            bothc = int((sft_any & any50).sum())
            fstr = [f"ΔHR@{k}={d[:, HR_KS.index(k)].mean()*100:+.2f}pp"
                    for k in MIG_KS]
            emit(f"  bucket {lab}  n={len(idx)} nIt={n_it} | 行级 "
                 + "  ".join(fstr)
                 + f" | ΔmacHR@50={d_m.mean()*100:+.2f}pp [{lo_b*100:+.2f}, {hi_b*100:+.2f}]"
                 + f" | newIt={newc} lostIt={lostc} bothIt={bothc}")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[write] {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
