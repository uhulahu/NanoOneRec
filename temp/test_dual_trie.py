#!/usr/bin/env python3
"""per-task 双 trie 行为单测（2026-09-03，直接验证生产函数 build_sid_hash_tries）。

核心断言：
  1. 碰撞 item（4 token，含 <d_x>）：全长 trie 在 [s1,s2,s3] 节点允许续 <d_x>（且无 \n/EOS）；
     前缀 trie 在 [s1,s2,s3] 节点只允许 \n→EOS，<d_x> 不可达。
  2. unique item（3 token）：两棵树在 3 级节点行为完全相同。
  3. 全量扫描：前缀 trie 的任何 allowed token 都解码不出 "<d_"（身份后缀彻底不可达）。
  4. 复刻 ConstrainedLogitsProcessor 的 count-window 语义逐级走树（template → s1 → [s1,s2] → ...
     → \n → EOS），两棵树都能走到 EOS。

用法：python temp/test_dual_trie.py [--sample 300]
"""
import argparse
import glob
import os
import random
import re
from collections import Counter

from minionerec_trainer import build_sid_hash_tries
from transformers import AutoTokenizer

ROOT = "data/Amazon23/Industrial_and_Scientific"
VARIANT = "rqkmeans-td-mean-20260830"   # rl.sh 当前 info_file 指向
BASE_MODEL = "./outputs/final_checkpoint"  # RL base model（SFT ckpt，tokenizer 含 SID added tokens）

SID_RE = re.compile(r"<[abcd]_\d+>")


def h(key):
    return '-'.join(str(_) for _ in key)


def walk_to_eos(tree, template, sid_ids, nl_id, eos, tag, s, check):
    """复刻 decode count-window 走树：key = 已生成 sid tokens（首级 = 模板尾）。"""
    allowed = set(tree.get(h(template), []))
    check(sid_ids[0] in allowed, f"{tag} {s}: level1 断链")
    cur = [sid_ids[0]]
    for want in sid_ids[1:]:
        allowed = set(tree.get(h(cur), []))
        check(want in allowed, f"{tag} {s}: 走树断链 at {want} (key={cur})")
        cur = cur + [want]
    allowed = set(tree.get(h(cur), []))
    check(nl_id in allowed, f"{tag} {s}: 末端应允许 \\n，实际 {allowed}")
    allowed = set(tree.get(h(cur + [nl_id]), []))
    check(eos in allowed, f"{tag} {s}: \\n 后应能 EOS，实际 {allowed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(42)

    info_cands = glob.glob(os.path.join(ROOT, "sid", VARIANT, "info", "*.txt"))
    assert info_cands, f"info file not found under {ROOT}/sid/{VARIANT}/info/"
    info_file = info_cands[0]
    print(f"[test] info_file = {info_file}")
    print(f"[test] base_model = {BASE_MODEL}")

    full_sids = [l.split('\t')[0].strip() for l in open(info_file)]
    lens = sorted({len(SID_RE.findall(s)) for s in full_sids})
    print(f"[test] {len(full_sids)} items, sid-length set = {lens}")
    assert lens == [3, 4], f"unexpected sid lengths: {lens}"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    eos = tok.eos_token_id

    # —— 生产树 ——
    hash_full, hash_pref, max_sid = build_sid_hash_tries(info_file, BASE_MODEL)
    print(f"[test] max_sid = {max_sid}, full keys = {len(hash_full)}, prefix keys = {len(hash_pref)}")
    assert max_sid == 4

    def walk_tokens(s):
        """按行 tokenize 解析结构：前 3 = 模板尾，随后 n 个 sid token，末尾 1 个 \\n。"""
        ids = tok(f"### Response:\n{s}\n").input_ids
        n = len(SID_RE.findall(s))
        assert len(ids) == 3 + n + 1, \
            f"unexpected line token structure for {s}: len(ids)={len(ids)} != {3 + n + 1}"
        template, sid_ids, nl_id = ids[:3], ids[3:3 + n], ids[3 + n]
        assert tok.decode([nl_id]).strip() == "", f"tail not newline: {tok.decode([nl_id])!r}"
        return template, sid_ids, nl_id

    # 抽样：一半碰撞一半 unique
    s4 = [s for s in full_sids if len(SID_RE.findall(s)) == 4]
    s3 = [s for s in full_sids if len(SID_RE.findall(s)) == 3]
    sample = rng.sample(s4, min(args.sample // 2, len(s4))) + rng.sample(s3, min(args.sample // 2, len(s3)))

    n_fail = 0

    def check(cond, msg):
        nonlocal n_fail
        if not cond:
            n_fail += 1
            print(f"  FAIL: {msg}")

    for s in sample:
        template, sid_ids, nl_id = walk_tokens(s)
        is_collide = len(sid_ids) == 4
        tag = "collide" if is_collide else "unique "

        # 1/2 级两树一致且可达
        for depth in range(3):
            key = h(template) if depth == 0 else h(sid_ids[:depth])
            nxt = sid_ids[depth]
            check(nxt in set(hash_full.get(key, [])), f"{tag} {s} d{depth+1}: full trie 缺 {nxt}")
            check(nxt in set(hash_pref.get(key, [])), f"{tag} {s} d{depth+1}: prefix trie 缺 {nxt}")

        key3 = h(sid_ids[:3])
        full3 = set(hash_full.get(key3, []))
        pref3 = set(hash_pref.get(key3, []))
        if is_collide:
            # 全长 trie：3 级后必须能续 <d_x>，且不允许停止
            check(sid_ids[3] in full3, f"collide {s}: full trie 3 级后缺 <d> 续接")
            check(eos not in full3 and nl_id not in full3,
                  f"collide {s}: full trie 3 级后错误允许停止")
            # 前缀 trie：3 级后只允许 \n→EOS，<d_x> 不可达
            check(sid_ids[3] not in pref3, f"collide {s}: prefix trie 竟允许 <d> 续接")
            check(pref3 == {nl_id}, f"collide {s}: prefix trie 3 级后 allowed={pref3}，应只 \\n")
        else:
            check(full3 == pref3, f"unique {s}: 两树 3 级节点不一致 full={full3} pref={pref3}")

        # 两树各走一遍到 EOS
        walk_to_eos(hash_pref, template, sid_ids[:3], nl_id, eos, tag + "prefix", s, check)
        if is_collide:
            walk_to_eos(hash_full, template, sid_ids, nl_id, eos, tag + "full  ", s, check)

    # —— 全量扫描：前缀 trie 的 allowed token 不得解码出 "<d_" ——
    bad = [(key, tid) for key, allowed in hash_pref.items() for tid in allowed if "<d_" in tok.decode([tid])]
    print(f"[test] prefix trie 中 <d_> 泄漏 allowed token 数: {len(bad)}")
    check(len(bad) == 0, f"prefix trie 泄漏 <d_>: {bad[:5]}")

    # —— 全长 trie 抽查 5 个碰撞桶：d 层允许集 ⊇ 桶内各 item 的 4th token ——
    pref_counter = Counter(tuple(SID_RE.findall(s)[:3]) for s in full_sids)
    for p3 in rng.sample([p for p, c in pref_counter.items() if c > 1], 5):
        members = [s for s in full_sids if tuple(SID_RE.findall(s)[:3]) == p3]
        _, m0_ids, _ = walk_tokens(members[0])
        key3 = h(m0_ids[:3])  # 注意 [:3]：节点 key = 3 级前缀，不是全长
        allowed = set(hash_full.get(key3, []))
        for m in members:
            _, mids, _ = walk_tokens(m)
            check(mids[3] in allowed, f"full trie d 层缺 {mids[3]} for bucket {p3}")

    print(f"\n=== {'PASS' if n_fail == 0 else f'{n_fail} FAILURES'} ===")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
