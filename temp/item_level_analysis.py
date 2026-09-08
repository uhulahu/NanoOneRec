#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按目标商品的 cluster bootstrap / macro-HR + unseen 对齐曝光分桶（2026-09-03，零训练成本）

A. 行级 bootstrap 把同一商品多次测试出现当独立样本（unseen ~8 次 vs seen ~2.4 次/商品），CI 可能偏窄。
   → 以 target item 为簇：每桶 new/lost 命中涉及的不同商品数、商品等权 macro HR、
   item 级 cluster bootstrap 的 Δ(RL−SFT) HR@50 CI。unseen 提升仍显著 ⇒ 排除"少数高频冷商品撑场"。

B. unseen 按 RL 对齐曝光（RLTitle2SidDataset 复刻：data 构造 + random.seed(0)+sample(10000)）分：
   E1 = 自身 (title/desc → 自身前缀) 记录被采样（文本↔前缀必须配对；空 desc 行被采≠空 desc 商品曝光）；
   E2 = 自身未采样，但商品前缀受任一对齐样本监督（含 seen/全目录其他商品）；
   E3 = 前缀未获 RL 对齐监督（≠ RL 没看过内容：SFT 全量曝光过，U4 前缀可能经其他商品出现在 NTP）。
   机制判别：仅 E1 升=直接内容适配；E2 也升=共享语义前缀组内迁移；E3 仍升=跨商品语义泛化。

用法：
  python temp/item_level_analysis.py \
      --index .../Industrial_and_Scientific.index.json \
      --train-csv ".../train/*.csv" \
      --item-json data/Amazon23/Industrial_and_Scientific/raw/Industrial_and_Scientific.item.json \
      --sft-json results/outputs_final_checkpoint/final_result_Industrial_and_Scientific.json \
      --model fd750=... --model base750=... [--model ...] --out results/item_level_analysis.txt
"""
import argparse
import json
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, "temp")
from bucket_analysis import PAT, HR_KS, load_index, load_seen, preprocess  # noqa: E402

ALIGN_SAMPLE = 10000
ALIGN_SEED = 0
ROW_KS = [1, 10, 50]
KI = {k: HR_KS.index(k) for k in ROW_KS}


def replicate_alignment_exposure(item_json_path, index_path):
    """复刻 RLTitle2SidDataset.data（文本→3 级前缀的映射，dict 折叠语义照抄）+ random.sample(seed=0)。

    返回 (sampled_rows, sampled_prefixes, text2pref, stats)：
      sampled_rows:  {(task, text, target_3prefix)}，task ∈ {'t','d'}——保存目标前缀是关键：
                     title/desc 可能重复或为空，单存文本会把"空 desc 行被采"误判成所有空 desc 商品的曝光
      sampled_prefixes: {prefix of sampled rows}（E2 用：前缀受任一对齐样本监督）
      text2pref:      (title2pref, desc2pref) —— data.py 的 dict 折叠（同文本后者覆盖，前缀=最后映射商品）
    """
    item_feat = json.load(open(item_json_path))
    indices = json.load(open(index_path))
    title2pref, desc2pref = {}, {}
    for item_id, sids in indices.items():
        if item_id in item_feat:
            title = item_feat[item_id]["title"]
            description = item_feat[item_id]["description"]
            if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
                try:
                    lst = eval(description)
                    description = lst[0] if lst else description
                except Exception:
                    pass
            if len(sids) >= 3:
                p3 = tuple(PAT.findall("".join(sids)))[:3]
                title2pref[title] = p3
                desc2pref[description] = p3
    data = [("t", t) for t in title2pref] + [("d", d) for d in desc2pref]
    random.seed(ALIGN_SEED)
    sampled = random.sample(data, ALIGN_SAMPLE)
    sampled_rows = set()
    for task, text in sampled:
        pref = title2pref[text] if task == "t" else desc2pref[text]
        sampled_rows.add((task, text, pref))
    sampled_prefixes = {r[2] for r in sampled_rows}
    stats = {"rows": len(data), "titles": len(title2pref), "descs": len(desc2pref)}
    return sampled_rows, sampled_prefixes, (title2pref, desc2pref), stats


def item_texts(item_json_path, index_path, iids):
    """逐商品 title/desc（desc 按 data.py 归一）+ 3 级前缀。返回 (title_map, desc_map, pref_map)。"""
    item_feat = json.load(open(item_json_path))
    idx = json.load(open(index_path))
    titles, descs, prefs = {}, {}, {}
    for iid in iids:
        feat = item_feat.get(iid, {})
        t = str(feat.get("title", ""))
        d = str(feat.get("description", ""))
        if d.startswith("['") and d.endswith("']"):
            try:
                lst = eval(d)
                d = lst[0] if lst else d
            except Exception:
                pass
        titles[iid], descs[iid], prefs[iid] = t, d, tuple(PAT.findall("".join(idx[iid])))[:3]
    return titles, descs, prefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--item-json", required=True)
    ap.add_argument("--sft-json", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=json_path")
    ap.add_argument("--out", default="results/item_level_analysis.txt")
    ap.add_argument("--bootstrap", type=int, default=1999)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    full2id, leaf3 = load_index(args.index)
    seen = load_seen(args.train_csv, full2id)
    _, sft_metas, sft_metrics, _ = preprocess(args.sft_json, full2id, leaf3, seen)
    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        _, metas, metrics, n_unres = preprocess(path, full2id, leaf3, seen)
        models.append({"name": name, "metas": metas, "metrics": metrics})
        print(f"[load] {name}: {len(metas)} samples, {n_unres} unresolvable")
    n = len(sft_metas)
    assert all(len(m["metas"]) == n for m in models), "模型 json 样本数不一致"

    # ---- 每 item 的样本 idx + 桶标签 ----
    item2idx = {}
    for i, iid in enumerate(sft_metas):
        if iid is not None:
            item2idx.setdefault(iid, []).append(i)
    item_ids = sorted(item2idx.keys(), key=int)
    j_of = {iid: j for j, iid in enumerate(item_ids)}
    J = len(item_ids)

    def bucket_of(iid):
        sm = sft_metrics[item2idx[iid][0]]
        return ("S3" if not sm.need_d else "S4") if sm.seen else ("U3" if not sm.need_d else "U4")

    item_bucket = np.array([bucket_of(i) for i in item_ids])
    BUCKET_LABELS = ["all", "seen", "unseen", "S3", "S4", "U3", "U4"]

    # ---- per-model 商品级向量 ----
    vecs = {}
    for name, metrics in [("SFT", sft_metrics)] + [(m["name"], m["metrics"]) for m in models]:
        macro = {k: np.zeros(J) for k in ROW_KS}
        anyhit = {k: np.zeros(J, dtype=bool) for k in ROW_KS}
        for j, iid in enumerate(item_ids):
            idxs = item2idx[iid]
            for k in ROW_KS:
                h = np.array([metrics[i].hr[KI[k]] for i in idxs])
                macro[k][j] = h.mean()
                anyhit[k][j] = h.any()
        vecs[name] = {"macro": macro, "any": anyhit}

    def members_mask(b):
        if b == "all":
            return np.ones(J, dtype=bool)
        if b == "seen":
            return np.isin(item_bucket, ["S3", "S4"])
        if b == "unseen":
            return np.isin(item_bucket, ["U3", "U4"])
        return item_bucket == b

    out = []
    def emit(s=""):
        out.append(s)

    rng = np.random.default_rng(args.seed)
    B = args.bootstrap

    # ================= A. 商品级（cluster bootstrap） =================
    emit("=" * 125)
    emit("A. 商品级分析：rowHR50 参照 | macro HR@1/10/50（商品内样本均值→商品间等权）| "
         "Δmacro@50(SFT→RL) item-cluster bootstrap 95%CI | new/lost 商品数（K=50）")
    emit("=" * 125)
    emit(f"{'bucket':<8}{'model':<9}{'rowHR50':>9}{'macHR1':>8}{'macHR10':>8}{'macHR50':>8}"
         f"{'Δmac50':>9}{'CI[lo':>9}{'hi]':>9}{'n_newIt':>8}{'n_lostIt':>9}{'n_bothIt':>9}")
    for b in BUCKET_LABELS:
        mask = members_mask(b)
        memb = np.nonzero(mask)[0]
        if len(memb) == 0:
            continue
        n_smp = sum(len(item2idx[item_ids[j]]) for j in memb)
        for m in [None] + models:
            name = "SFT" if m is None else m["name"]
            v = vecs[name]
            if m is None:
                row = sum(1 for j in memb for i in item2idx[item_ids[j]] if sft_metrics[i].hr[4]) / n_smp
                d_m = None
            else:
                rl = vecs[name]
                row = sum(1 for j in memb for i in item2idx[item_ids[j]] if m["metrics"][i].hr[4]) / n_smp
                d_m = (rl["macro"][50][memb] - vecs["SFT"]["macro"][50][memb])
                draws = rng.integers(0, len(memb), size=(B, len(memb)))
                boot = d_m[draws].mean(axis=1)
                lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            cell = (f"{b:<8}{name:<9}{row*100:>8.2f}%"
                    f"{v['macro'][1][memb].mean()*100:>8.2f}%{v['macro'][10][memb].mean()*100:>8.2f}%"
                    f"{v['macro'][50][memb].mean()*100:>8.2f}%")
            if d_m is None:
                cell += f"{'-':>9}{'-':>9}{'-':>9}"
                newc = lostc = bothc = "-"
            else:
                cell += f"{d_m.mean()*100:>+8.2f}pp{lo*100:>+9.2f}{hi*100:>+9.2f}"
                sft_any50 = vecs["SFT"]["any"][50][memb]
                any50 = v["any"][50][memb]
                newc = int(((~sft_any50) & any50).sum())
                lostc = int((sft_any50 & (~any50)).sum())
                bothc = int((sft_any50 & any50).sum())
            cell += f"{str(newc):>8}{str(lostc):>9}{str(bothc):>9}"
            emit(cell)
        emit("")

    # ================= B. unseen 对齐曝光分桶（2026-09-03 修正版） =================
    # E1 = 商品自身 (title 或 desc, 其前缀) 记录被采样（文本↔前缀必须匹配：同文本可能映射到别的商品前缀，
    #      空 desc 行被采不代表空 desc 商品都曝光——旧版只存文本导致 804 个空 desc 商品全被误判 E1）
    # E2 = 自身记录未采样，但该商品前缀 ∈ sampled_prefixes（任一对齐样本监督过此前缀，含 seen/其他商品）
    # E3 = 该前缀未被 RL 对齐任务监督（≠"RL 没看过该内容"：SFT 全量看过；U4 前缀可能经其他商品出现在 NTP）
    emit("=" * 125)
    emit("B. unseen 按 RL 对齐前缀监督分 E1/E2/E3（RLTitle2Sid 复刻 sample=10000 seed=0，修正版）")
    emit("=" * 125)
    sampled_rows, sampled_prefixes, (title2pref, desc2pref), stats = \
        replicate_alignment_exposure(args.item_json, args.index)
    emit(f"[replicate] 对齐 data rows={stats['rows']} (title {stats['titles']} + desc {stats['descs']}), "
         f"sampled={ALIGN_SAMPLE}, 不同前缀受监督数={len(sampled_prefixes)}")
    unseen_mask = members_mask("unseen")
    u_ids = [item_ids[j] for j in np.nonzero(unseen_mask)[0]]
    u_titles, u_descs, u_prefs = item_texts(args.item_json, args.index, u_ids)
    # 采样行索引：(task, text) → prefix
    sampled_tpref = {(r[0], r[1]): r[2] for r in sampled_rows}
    E = {}
    for iid in u_ids:
        p = u_prefs[iid]
        t, d = u_titles[iid], u_descs[iid]
        own = ((("t", t) in sampled_tpref and sampled_tpref[("t", t)] == p) or
               (("d", d) in sampled_tpref and sampled_tpref[("d", d)] == p))
        E[iid] = 1 if own else (2 if p in sampled_prefixes else 3)

    emit(f"\n{'E组':<4}{'items':>6}{'samples':>8}{'U3/U4':>10}   定义")
    for g in (1, 2, 3):
        g_ids = [i for i in u_ids if E[i] == g]
        n_smp = sum(len(item2idx[i]) for i in g_ids)
        u3 = sum(1 for i in g_ids if not sft_metrics[item2idx[i][0]].need_d)
        note = {1: "自身 (title/desc→自身前缀) 记录被采样",
                2: "自身未采样，但前缀被其他对齐样本监督（含 seen/其他商品）",
                3: "前缀未获 RL 对齐任务监督（≠ RL 没看过内容：SFT 全量曝光）"}[g]
        emit(f"E{g:<4}{len(g_ids):>6}{n_smp:>8}{u3:>5}/{len(g_ids)-u3:<5}   {note}")

    emit(f"\n{'E组':<4}{'model':<9}{'rowHR50 SFT/RL':>17}{'n_newIt':>8}{'n_lostIt':>9}"
         f"{'Δmac50':>9}{'CI[lo':>9}{'hi]':>9}")
    for g in (1, 2, 3):
        g_ids = [i for i in u_ids if E[i] == g]
        if not g_ids:
            continue
        memb_j = np.array([j_of[i] for i in g_ids])
        sft_row = sum(1 for j in memb_j for i in item2idx[item_ids[j]] if sft_metrics[i].hr[4]) / \
            sum(len(item2idx[item_ids[j]]) for j in memb_j)
        for m in models:
            v = vecs[m["name"]]
            row = sum(1 for j in memb_j for i in item2idx[item_ids[j]] if m["metrics"][i].hr[4]) / \
                sum(len(item2idx[item_ids[j]]) for j in memb_j)
            d_m = v["macro"][50][memb_j] - vecs["SFT"]["macro"][50][memb_j]
            draws = rng.integers(0, len(memb_j), size=(B, len(memb_j)))
            boot = d_m[draws].mean(axis=1)
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            sft_any = vecs["SFT"]["any"][50][memb_j]
            any50 = v["any"][50][memb_j]
            newc = int(((~sft_any) & any50).sum())
            lostc = int((sft_any & (~any50)).sum())
            emit(f"E{g:<4}{m['name']:<9}{sft_row*100:>7.2f}% -> {row*100:>7.2f}%"
                 f"{newc:>8}{lostc:>9}{d_m.mean()*100:>+8.2f}pp{lo*100:>+9.2f}{hi*100:>+9.2f}")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[write] {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
