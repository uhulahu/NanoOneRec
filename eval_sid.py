#!/usr/bin/env python3
"""Evaluate SID quality for the MiniOneRec pipeline: codebook utilization,
purity, NMI/ARI, entropy, collision / extra-token usage.

Current-project inputs (all under ``root``, dataset named ``<d>``):
    <d>.codes_constrained.npy   codes from rq/rqkmeans_constrained.py, shape (N, L),
                                values in [0, K) per layer
    <d>.item.json               item metadata dict {item_id: {...}} with
                                'categories' (comma-joined path) and 'brand'
    <d>.index.json              (optional) final SID index {item_id: ['<a_5>', ...]};
                                entries with len > L needed an extra disambiguation
                                token (the collision-free form fed to the model)

Run standalone (2026-09-03 目录重组后)：
    python eval_sid.py --dataset Industrial_and_Scientific \
        --root data/Amazon23/Industrial_and_Scientific \
        --variant rqkmeans-td-mean-20260830 --k 256 --l 3
（默认解析：codes/index 在 <root>/sid/<variant>/，item.json 在 <root>/raw/；找不到时回退旧扁平布局 <root>/ 下）
or import:
    from eval_sid import evaluate_sid
    report = evaluate_sid(sid_path, meta_path, index_path,
                          n_levels=3, codebook_sizes=[256, 256, 256])

``codebook_sizes`` gives the per-layer sizes of the *real* (non-extra)
codebook layers; ``n_levels`` is how many of those to evaluate.  If the SID
file carries an extra collision-resolution column, it is excluded from the
quality metrics but used for the "Full (with extra)" collision line.
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def load_json(path):
    with open(path) as f:
        return json.load(f)


def entropy(counts):
    """Shannon entropy H = -Σ p·log2(p)."""
    _, cnt = np.unique(counts, return_counts=True)
    p = cnt / cnt.sum()
    return -np.sum(p * np.log2(p))


def evaluate_sid(sid_path, meta_path, index_path=None, n_levels=3,
                 codebook_sizes=None):
    """Run full SID quality evaluation and return the report as a string.

    Args:
        sid_path:    path to the SID .npy from rqkmeans_constrained.py,
                     shape (N, n_levels), values in [0, K) per layer.
        meta_path:   path to the item metadata json dict {item_id: {...}}
                     ('categories' comma-joined, 'brand' optional).
        index_path:  optional path to the final index.json
                     {item_id: ['<a_5>', ...]}; entries longer than n_levels
                     needed an extra disambiguation token.
        n_levels:       number of *real* (non-extra) codebook layers to evaluate.
        codebook_sizes: per-layer codebook sizes for the real layers
                        (len == n_levels).  Default [256]*n_levels (backward compat).
    """
    if codebook_sizes is None:
        codebook_sizes = [256] * n_levels
    assert len(codebook_sizes) == n_levels, \
        f"codebook_sizes {codebook_sizes} must match n_levels {n_levels}"

    codes = np.load(sid_path).astype(int)
    if codes.ndim != 2:
        raise ValueError(f"SID codes must be 2D (N, n_levels), got shape {codes.shape}")
    codes = codes[:, :n_levels]          # real layers only
    N = len(codes)

    meta = load_json(meta_path)

    # item.json 的 key 就是 0..N-1 的行号，与 codes 行一一对应，无需额外映射
    # coarse category = 逗号分隔路径的第一段（例如 "Industrial & Scientific, ..." → "Industrial & Scientific"）
    cat_of = {}
    for iid, m in meta.items():
        cats = str(m.get('categories', '')).strip()
        first = cats.split(',')[0].strip() if cats else ''
        cat_of[int(iid)] = first or 'Unknown'

    has_cat = np.array([(i in cat_of) and (cat_of[i] != 'Unknown') for i in range(N)])
    cats = np.array([cat_of.get(i, 'Unknown') for i in range(N)])

    brand_of = {}
    for iid, m in meta.items():
        brand_of[int(iid)] = str(m.get('brand', '')).strip() or 'Unknown'
    brand_arr = np.array([brand_of.get(i, 'Unknown') for i in range(N)])

    # ── Header ──
    lines = []
    lines.append("=" * 55)
    lines.append("  SID Quality Evaluation")
    lines.append("  arch (real layers): " + "×".join(map(str, codebook_sizes)))
    lines.append("=" * 55)
    lines.append("")

    # ── 0. Collision ──
    lines.append("─ 0. Collision ─")
    unique_prefix = len(set(map(tuple, codes)))
    n_col = N - unique_prefix
    lines.append(f"  Prefix (L1..L{n_levels}): {n_col}/{N} collisions "
                 f"({100 * n_col / N:.2f}%)")
    # index.json 里的条目若长于 n_levels，说明该 item 前缀碰撞、追加了去重 token
    if index_path and os.path.exists(index_path):
        index = load_json(index_path)
        n_extra = sum(1 for toks in index.values() if len(toks) > n_levels)
        max_extra_len = max((len(toks) for toks in index.values()), default=n_levels)
        lines.append(f"  Extra tokens: {n_extra}/{N} items need them "
                     f"({100 * n_extra / N:.2f}%), max extra length {max_extra_len - n_levels}")
        if len(index) != N:
            lines.append(f"  !! index.json covers {len(index)} items, codes have {N} rows")

    # ── 0b. Collision bucket-size distribution 碰撞桶大小分布 ──
    # For every prefix bucket holding ≥2 items, record its size.  Large buckets
    # mean the extra token must do a lot of identity disambiguation within them.
    prefix_counts = Counter(tuple(codes[i]) for i in range(N))
    coll_sizes = np.array([s for s in prefix_counts.values() if s > 1])
    lines.append("")
    lines.append("  Collision bucket-size distribution (prefixes with ≥2 items):")
    if len(coll_sizes) > 0:
        lines.append(f"    buckets: {len(coll_sizes)}   items: {coll_sizes.sum()}   "
                     f"min: {coll_sizes.min()}   median: {np.median(coll_sizes):.0f}   "
                     f"mean: {coll_sizes.mean():.1f}   p90: {np.percentile(coll_sizes, 90):.0f}   "
                     f"p95: {np.percentile(coll_sizes, 95):.0f}   max: {coll_sizes.max()}")
        hist = Counter(int(s) for s in coll_sizes)
        sorted_hist = sorted(hist.items())
        parts = [f"{k}:{v}" for k, v in sorted_hist[:8]]
        tail = sorted_hist[8:]
        if tail:
            parts.append(f">={tail[0][0]}:{sum(v for _, v in tail)}")
        lines.append("    size→buckets: " + "  ".join(parts))
    else:
        lines.append("    none (every prefix unique)")
    lines.append("")

    # ── 1. Codebook utilization ──
    lines.append("─ 1. Codebook utilization ─")
    for level in range(n_levels):
        used = len(set(codes[:, level]))
        total = codebook_sizes[level]
        lines.append(f"  L{level+1}: {used}/{total} entries used ({100*used/total:.1f}%)")

    # ── 2. Prefix Purity ──
    lines.append("")
    lines.append("─ 2. Prefix Purity ─")
    lines.append(f"  {'L':<3} {'Groups':<8} {'Purity':<10} {'Cat.NMI':<10} {'Cat.ARI':<10} {'BrandPurity':<12}")
    lines.append(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for L in range(1, n_levels + 1):
        prefix = [tuple(codes[i, :L]) for i in range(N)]
        groups = defaultdict(list)
        for i, p in enumerate(prefix):
            groups[p].append(i)

        total_cat, correct_cat = 0, 0
        total_brand, correct_brand = 0, 0
        prefix_ids = np.empty(N, dtype=int)
        for gid, (p, members) in enumerate(groups.items()):
            for m in members:
                prefix_ids[m] = gid
            m_cats = [cats[m] for m in members if has_cat[m]]
            m_brands = [brand_arr[m] for m in members if brand_arr[m] != 'Unknown']
            if m_cats:
                correct_cat += Counter(m_cats).most_common(1)[0][1]
                total_cat += len(m_cats)
            if m_brands:
                correct_brand += Counter(m_brands).most_common(1)[0][1]
                total_brand += len(m_brands)

        purity = correct_cat / total_cat if total_cat else 0
        brand_purity = correct_brand / total_brand if total_brand else 0
        nmi = normalized_mutual_info_score(cats[has_cat], prefix_ids[has_cat])
        ari = adjusted_rand_score(cats[has_cat], prefix_ids[has_cat])

        lines.append(f"  L{L:<3} {len(groups):<8} {purity:<10.4f} {nmi:<10.4f} {ari:<10.4f} {brand_purity:<12.4f}")

    # ── L1 per-category ──
    lines.append("")
    lines.append("─ L1 assignment by coarse category (top-10 L1 codes) ─")
    l1_counter = Counter(int(codes[i, 0]) for i in range(N) if has_cat[i])
    for l1code, _ in l1_counter.most_common(10):
        members = [i for i in range(N) if has_cat[i] and codes[i, 0] == l1code]
        dist = Counter(cats[m] for m in members)
        majority = dist.most_common(1)[0]
        lines.append(f"  Code {l1code:>3} ({len(members):>4} items): "
                     f"majority={majority[0]} ({majority[1]/len(members):.0%}), "
                     f"dist={{{', '.join(f'{k}:{v}' for k,v in dist.most_common(5))}}}")
    lines.append(f"\n  Total items with category: {has_cat.sum()} / {N}")

    # ── 3. Bucket-size ──
    lines.append("")
    lines.append("─ 3. Prefix bucket-size distribution ─")
    lines.append(f"  {'L':<3} {'Groups':<8} {'Min':<6} {'p25':<6} {'Median':<8} "
                 f"{'Mean':<8} {'p75':<6} {'p95':<6} {'Max':<6} {'Std':<8}")
    lines.append(f"  {'-'*3} {'-'*8} {'-'*6} {'-'*6} {'-'*8} "
                 f"{'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for L in range(1, n_levels + 1):
        prefix = [tuple(codes[i, :L]) for i in range(N)]
        groups = defaultdict(list)
        for i, p in enumerate(prefix):
            groups[p].append(i)
        sizes = np.array([len(v) for v in groups.values()])
        lines.append(f"  L{L:<3} {len(groups):<8} {sizes.min():<6} "
                     f"{np.percentile(sizes, 25):<6.0f} "
                     f"{np.median(sizes):<8.0f} {sizes.mean():<8.1f} "
                     f"{np.percentile(sizes, 75):<6.0f} "
                     f"{np.percentile(sizes, 95):<6.0f} {sizes.max():<6} {sizes.std():<8.1f}")

    # ── 4. Conditional code usage (branching factor) ──
    lines.append("")
    lines.append("─ 4. Conditional code usage (branching factor) ─")
    for L in range(1, n_levels):
        groups = defaultdict(set)
        for i in range(N):
            groups[tuple(codes[i, :L])].add(int(codes[i, L]))
        branch = np.array([len(v) for v in groups.values()])
        K = codebook_sizes[L]
        lines.append(f"  L{L}→L{L+1}: per prefix, {len(groups)} prefixes → "
                     f"mean branching: {branch.mean():.1f}, "
                     f"median: {np.median(branch):.0f}, "
                     f"min={branch.min()}, max={branch.max()}, "
                     f"fraction of {K}: {branch.mean()/K:.3f}")

    # ── 5. Entropy ──
    lines.append("")
    lines.append("─ 5. Entropy (bits) ─")
    max_h = [np.log2(codebook_sizes[i]) for i in range(n_levels)]
    h_level = [entropy(codes[:, i]) for i in range(n_levels)]
    for i in range(n_levels):
        lines.append(f"  H(L{i+1})        = {h_level[i]:.4f}  "
                     f"(max {max_h[i]:.4f}, {100*h_level[i]/max_h[i]:.1f}% used)")
    # mixed-radix prefix encoding → unique integer per prefix
    def prefix_value(k):
        v = codes[:, 0].copy()
        for i in range(1, k):
            v = v * codebook_sizes[i] + codes[:, i]
        return v
    h_joint = [entropy(prefix_value(k)) for k in range(1, n_levels + 1)]
    for k in range(1, n_levels + 1):
        lines.append(f"  H(L1..L{k})     = {h_joint[k-1]:.4f}")
    for i in range(1, n_levels):
        lines.append(f"  H(L{i+1}|L1..L{i}) = {h_joint[i] - h_joint[i-1]:.4f}")
    lines.append("")
    lines.append("  Information breakdown (% of total):")
    total_h = entropy(codes.ravel())
    for i in range(n_levels):
        lines.append(f"  L{i+1}: {h_level[i]:.2f} bits "
                     f"({100*h_level[i]/total_h:.1f}% of token-level entropy)")
    lines.append(f"  Joint entropy of full SID: {h_joint[-1]:.4f}")

    return "\n".join(lines)


# ── Standalone entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate SID quality (rqkmeans_constrained codes + item.json [+ index.json])")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (e.g., Industrial_and_Scientific)")
    parser.add_argument('--root', type=str, default='data/Amazon23/Industrial_and_Scientific',
                        help="category 目录（2026-09-03 起：含 raw/ 与 sid/<variant>/）")
    parser.add_argument('--variant', type=str, default='rqkmeans-td-mean-20260830',
                        help="SID 变体目录名，codes/index 在 <root>/sid/<variant>/ 下")
    parser.add_argument('--k', type=int, default=256, help="Per-layer codebook size (uniform)")
    parser.add_argument('--l', type=int, default=3, help="Number of codebook levels")
    parser.add_argument('--codes_file', type=str, default=None,
                        help="Override SID codes .npy path")
    parser.add_argument('--index_file', type=str, default=None,
                        help="Override final index.json path (skip extra-token stats if absent)")
    args = parser.parse_args()

    def resolve(rel, name):
        """新布局路径优先；文件不存在时回退旧扁平布局 <root>/<name>。"""
        p = os.path.join(args.root, rel, name)
        if os.path.isfile(p):
            return p
        legacy = os.path.join(args.root, name)
        print(f"[eval_sid] {p} 不存在，回退 {legacy}")
        return legacy

    sid_path = args.codes_file or resolve(
        os.path.join('sid', args.variant), f'{args.dataset}.codes_constrained.npy')
    meta_path = resolve('raw', f'{args.dataset}.item.json')
    index_path = args.index_file or resolve(
        os.path.join('sid', args.variant), f'{args.dataset}.index.json')

    
    lines = []
    lines.append(f"[eval_sid] codes: {sid_path}\n          meta: {meta_path}\n          index: {index_path}")

    lines.append(evaluate_sid(sid_path, meta_path, index_path,
                       n_levels=args.l, codebook_sizes=[args.k] * args.l))

    with open(f'{os.path.join(args.root, 'sid', args.variant, 'sid_eval.txt')}', 'w') as f:
        for line in lines:
            f.write(line)


