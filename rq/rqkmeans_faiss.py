#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS ResidualQuantizer  +  Sinkhorn-based Uniform Semantic Mapping
===================================================================
"""

import argparse
import json
import os
from collections import defaultdict

import faiss
import numpy as np
from tqdm import tqdm


def pairwise_sq_dists_batch(X, C, C_norm2=None):
    """
    X: (B,d)   C: (K,d)
    """
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)              # (K,)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)   # (B,1)
    dots = X @ C.T                                   # (B,K)
    return X_norm2 + C_norm2[None, :] - 2.0 * dots


def train_faiss_rq(emb, num_levels=3, codebook_size=256, verbose=True):
    N, d = emb.shape
    if verbose:
        print("Training FAISS ResidualQuantizer")
        print(f"  data={N}  dim={d}  levels={num_levels}  "
              f"codebook={codebook_size}  total_codes={codebook_size ** num_levels:,}")

    nbits = int(np.log2(codebook_size))
    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    rq.train(np.ascontiguousarray(emb.astype(np.float32)))
    if verbose:
        print("  training completed\n")
    return rq

def unpack_rq_codes(codes, nbits, num_levels):
    """
    Unpack FAISS's bit-packed codes into integer index arrays

    Parameters:
    codes: (N, M_bytes) uint8 array, from rq.compute_codes
    nbits: number of bits per level (e.g., 512 -> 9 bits)
    num_levels: number of levels (e.g., 3)

    Returns:
    (N, num_levels) int32 array containing the unpacked indices
    """
    N = codes.shape[0]
    # FAISS uses Little Endian packing
    packed_ints = np.zeros(N, dtype=np.int64)
    for i in range(codes.shape[1]):
        packed_ints |= codes[:, i].astype(np.int64) << (8 * i)
    unpacked_codes = np.zeros((N, num_levels), dtype=np.int32)
    mask = (1 << nbits) - 1  # e.g., mask for 9 bits is 511 (0x1FF)
    for i in range(num_levels):
        unpacked_codes[:, i] = (packed_ints >> (i * nbits)) & mask
    return unpacked_codes

def encode_with_rq(rq, data, codebook_size, verbose=True):
    data = np.ascontiguousarray(data.astype(np.float32))
    nbits = int(np.log2(codebook_size))  # 表示每层码本的索引占多少比特
    if verbose:
        print(f"Encoding {data.shape[0]} vectors ...")
    codes_packed = rq.compute_codes(data)

    if codes_packed.ndim == 1:
        n_bytes = (rq.M * nbits + 7) // 8  # 总比特数向上取整到字节数
        codes_packed = codes_packed.reshape(-1, n_bytes)

    # 把 ResidualQuantizer 输出的字节流还原成标准的整数索引数组 (N, M)
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        codes = unpack_rq_codes(codes_packed, nbits, rq.M)

    codes = codes.astype(np.int32)
    if verbose:
        print(f"  done, codes.shape={codes.shape}\n")
    return codes


def get_rq_codebooks(rq):
    M, d = rq.M, rq.d
    nbits0 = get_first_nbits(rq)  # 第一层码本索引占的比特数，最多能编码 2^nbits0 个不同值
    K = 1 << nbits0  # 这是 1 左移 nbits0 位，相当于 2^nbits0，也就是首层码本大小
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)  # 取所有簇心
    return cb_flat.reshape(M, K, d)  # (M,K,d) 这里假设码本所有层都一样大小


def compute_residuals_upto_level(rq, data, codes, upto_level, codebooks=None):
    if codebooks is None:
        codebooks = get_rq_codebooks(rq)
    residuals = np.ascontiguousarray(data.astype(np.float32)).copy()  # 初始化残差为输入向量
    for l in range(upto_level):
        residuals -= codebooks[l][codes[:, l]]  # codes里存的就是索引
    return residuals


def estimate_tau(residuals, centroids, sample_size=4000,
                 percentile=90, min_tau=1e-6):
    N = residuals.shape[0]
    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)
    spread = np.percentile(D - D.min(axis=1, keepdims=True),
                           percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(residuals, centroids, capacities=None, *,
                           batch_size=8192, iters=30, tau=None,
                           verbose=True, topk=32, seed=42):
    import ot                                     
    rng = np.random.RandomState(seed)
    N, d = residuals.shape
    K = centroids.shape[0]  # 每层码本大小

    # 每个码字(质心)的目标容量
    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    capacities = capacities.astype(np.int64)
    assert capacities.sum() == N

    if tau is None:
        tau = estimate_tau(residuals, centroids)
    if verbose:
        print(f"  Sinkhorn level: N={N}  K={K}  tau={tau:.5g}  "
              f"iters={iters}  batch={batch_size}")

    a = np.ones(N) / N         # 行的边缘分布
    b = capacities / float(N)  # 列的边缘分布 = 容量归一化
    Cn2 = np.sum(centroids * centroids, axis=1)


    def cost_fun(X):
        return pairwise_sq_dists_batch(X, centroids, Cn2)

    D_full = cost_fun(residuals).astype(np.float64)  # item-centroid 平方L2距离，作为初始分配矩阵
    P = ot.sinkhorn(a, b, D_full, tau)  # 这是直接调用现成库实现的 sinkhorn 算法，得到分配矩阵

    # argmax P 分配，给每个商品分配码字
    remaining = capacities.copy()
    assign = np.empty(N, dtype=np.int32)
    order = np.arange(N)
    rng.shuffle(order)
    for i in order:
        probs = P[i]  # 商品 i 被分配给每个质心的概率
        if topk and topk < K:
            # argpartition：把第k小的元素放在它的有序位置(k-1下标处)，并使其左边的元素全都 <= 它，右边的元素全都 >= 它
            cand = np.argpartition(-probs, topk - 1)[:topk]  
            cand = cand[np.argsort(-probs[cand])]
        else:
            cand = np.argsort(-probs)
        # 给商品分配码字时，只从概率最高的topk码字中选。
        # 为什么不是直接argmax？因为每个码字有容量限制
        chosen = -1
        for c in cand:
            if remaining[c] > 0:
                chosen = c
                break
        if chosen < 0:  # 如果概率最高的topk码字全满了，需要一个兜底，但是这里兜底逻辑是错的
            c = int(np.argmax(probs))
            if remaining[c] == 0:
                c = int(np.argmin(remaining))        
            chosen = c
        remaining[chosen] -= 1
        assign[i] = chosen

    if verbose:
        used = capacities - remaining
        print(f"    level balanced: min={used.min()}  max={used.max()}")

    return assign


def sinkhorn_uniform_mapping(rq, emb, codes, *, batch_size=8192,
                             iters=30, tau=None, verbose=True,
                             topk=32, seed=42):
    codebooks = get_rq_codebooks(rq)
    N, M = codes.shape
    K = codebooks.shape[1]

    codes_bal = codes.copy()
    for l in range(M):
        if verbose:
            print(f"\n=== Sinkhorn uniform mapping  level {l+1}/{M} ===")
        # 计算残差
        residuals = compute_residuals_upto_level(rq, emb, codes_bal, upto_level=l, codebooks=codebooks)
        # 给每个码字分配容量
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1

        new_ids = sinkhorn_balance_level(
            residuals, codebooks[l], capacities=capacities,
            batch_size=batch_size, iters=iters, tau=tau,
            verbose=verbose, topk=topk, seed=seed + l)

        codes_bal[:, l] = new_ids
    return codes_bal


def analyze_codes(codes, title="", verbose=True):
    N, M = codes.shape
    if verbose:
        if title:
            print(title)
        print(f"  total={N}")
        for l in range(M):
            print(f"  L{l+1}: unique={len(np.unique(codes[:, l]))}")
        combos = len(set(map(tuple, codes)))  # full-path 桶数量
        collision_rate = 1 - combos / N  # 碰撞率
        print(f"  unique full-paths={combos}  collision_rate={collision_rate:.4f}")
    return


def save_indices_json(codes, path, use_prefix=True):
    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    idx = {}
    for i, code in enumerate(codes):
        if use_prefix:
            idx[i] = [tpl[j].format(int(c)) for j, c in enumerate(code)]
        else:
            idx[i] = [int(c) for c in code]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print("Saved indices:", path)


def main():
    parser = argparse.ArgumentParser(
        description="FAISS-RQ + Sinkhorn uniform mapping")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--data_path", type=str, default=None, help="embedding 路径")

    parser.add_argument("--num_levels", type=int, default=3, help="码本层数")
    parser.add_argument("--codebook_size", type=int, default=256, help="每层码本大小")
    parser.add_argument("--uniform", action="store_true", help="enable Sinkhorn uniform mapping")
    parser.add_argument("--sk_iters", type=int, default=30, help="Sinkhorn iterations")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--output_root", default="../data")
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = f"../data/Amazon/index/{args.dataset}.emb-qwen-td.npy"

    out_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.dataset}.faiss-rq.index.json")
    out_faiss = out_json.replace(".json", ".faiss")

    print("loading:", data_path)
    emb = np.load(data_path)
    print("shape:", emb.shape)

    # 训练 rqkmeans
    rq = train_faiss_rq(emb, args.num_levels, args.codebook_size)
    # 推理获得 SID
    codes_raw = encode_with_rq(rq, emb, args.codebook_size, verbose=True)
    # 计算碰撞率
    analyze_codes(codes_raw, "Before balancing:")

    if args.uniform:
        codes_bal = sinkhorn_uniform_mapping(
            rq, emb, codes_raw,
            batch_size=args.batch_size,
            iters=args.sk_iters,
            verbose=True)
        # 再计算一下碰撞率
        analyze_codes(codes_bal, "After  balancing:")
        codes_final = codes_bal
    else:
        codes_final = codes_raw

    save_indices_json(codes_final, out_json, use_prefix=True)

    try:
        nbits_val = get_first_nbits(rq)     
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits_val)
        index.rq = rq
        index.is_trained = True
        faiss.write_index(index, out_faiss)
        print("Saved faiss quantizer:", out_faiss)
    except Exception as e:
        print("save faiss index failed:", e)

def get_first_nbits(rq):
    if isinstance(rq.nbits, int):
        return rq.nbits            
    return int(faiss.vector_to_array(rq.nbits).ravel()[0])

if __name__ == "__main__":
    main()
