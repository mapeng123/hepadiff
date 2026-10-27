#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实 vs 真实 AUC 定标对照 v2 (修正 indptr 索引 bug + 增加对照):
修正: 全程使用原始语料行号 + 原始 indptr 读 token (与 gen_panels_v5.py 同法).
三组对比:
 1) cross_dataset: 同层内 数据集A vs 数据集B (原始)
 2) cross_dataset_depth_matched: 同上, 但两侧按检出数分布匹配抽样
 3) gen_vs_single_dataset: 生成细胞(B配置, 同层) vs 单一数据集真实细胞
sanity: 同数据集随机二分 (~0.5).
输出: /root/autodl-tmp/panels_v5_unbiased/real_vs_real_v2.csv"""
import numpy as np, pandas as pd, json, time
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

MM = "/root/autodl-tmp/virtual-liver/processed/v4_mm_r1"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
GEN_NPZ = "/root/autodl-tmp/panels_v5_unbiased/gen100k_B_steps4_nobias.npz"
OUT_CSV = "/root/autodl-tmp/panels_v5_unbiased/real_vs_real_v2.csv"
edges = np.load(EDGES_F)
LUT = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)
t0 = time.time()
rng = np.random.default_rng(0)

meta = pd.read_csv(f"{MM}/meta_v7_v5.csv", low_memory=False,
                   usecols=["celltype_v5", "condition_v3", "species", "dataset"])
for c in meta.columns:
    meta[c] = meta[c].astype(str)
meta["species"] = meta["species"].str.lower()
indptr = np.load(f"{MM}/indptr.npy", mmap_mode="r")
nnz_all = np.diff(indptr)
keep_mask = (nnz_all >= 50) & (meta["condition_v3"] != "EXCLUDED").to_numpy()
keep_rows = np.where(keep_mask)[0]           # 原始语料行号
meta = meta.iloc[keep_rows]                  # 保留原始 index (= 语料行号)
mm_data = np.load(f"{MM}/data.npy", mmap_mode="r")
mm_index = np.load(f"{MM}/indices.npy", mmap_mode="r")
G = 16000
print(f"corpus {len(meta)} | {time.time()-t0:.0f}s", flush=True)

def read_tok(orig_rows):
    """orig_rows: 原始语料行号. 与 gen_panels_v5.py 完全同法."""
    tok = np.zeros((len(orig_rows), G), dtype=np.int8)
    for j, r in enumerate(orig_rows):
        s, e = indptr[r], indptr[r + 1]
        tok[j, mm_index[s:e]] = mm_data[s:e]
    return tok

def log_chunk(tok):
    X = LUT[tok.astype(np.int16)] * 1e4
    np.log1p(X, out=X)
    return X

def auc_pair(toka, tokb):
    X = np.vstack([log_chunk(toka), log_chunk(tokb)])
    y = np.r_[np.zeros(len(toka)), np.ones(len(tokb))]
    Z = PCA(n_components=50, random_state=0).fit_transform(X)
    sc = cross_val_predict(LogisticRegression(max_iter=1000), Z, y, cv=5,
                           method="predict_proba")[:, 1]
    auc_lr = float(roc_auc_score(y, sc))
    hv = np.argsort(X.var(0))[-2000:]
    sc2 = cross_val_predict(KNeighborsClassifier(n_neighbors=15, n_jobs=20),
                            X[:, hv], y, cv=5, method="predict_proba")[:, 1]
    return auc_lr, float(roc_auc_score(y, sc2))

def depth_matched_pick(idx_a_ref, idx_b):
    """按 a 侧检出数分布, 从 b 侧匹配抽样 (分箱最近邻)."""
    nnz_a = nnz_all[idx_a_ref]
    nnz_b = nnz_all[idx_b]
    order = np.argsort(nnz_b)
    pos = np.searchsorted(nnz_b[order], nnz_a)
    pos = np.clip(pos, 0, len(order) - 1)
    return idx_b[order[pos]]

# 生成细胞 (B 配置)
dg = np.load(GEN_NPZ, allow_pickle=True)
gen_tok = dg["gen"]
g_ct, g_cond, g_sp = dg["ct_gen"].astype(str), dg["cond_gen"].astype(str), dg["sp_gen"].astype(str)
print(f"gen loaded | {time.time()-t0:.0f}s", flush=True)

meta["stratum"] = meta["species"] + "|" + meta["condition_v3"] + "|" + meta["celltype_v5"]
rows = []
cands = []
for st, g in meta.groupby("stratum"):
    vc = g["dataset"].value_counts()
    big = vc[vc >= 3000]
    if len(big) >= 2:
        cands.append((st, len(g), big))
cands.sort(key=lambda x: -x[1])
# 优先选有注释细胞型 (非 unknown)
ann = [c for c in cands if not c[0].endswith("|unknown")]
cands = (ann + [c for c in cands if c[0].endswith("|unknown")])[:8]
print(f"{len(cands)} strata", flush=True)

N_SIDE = 4000
for st, n_st, big in cands:
    g = meta[meta["stratum"] == st]
    vc = g["dataset"].value_counts()
    ds_a, ds_b = list(big.index[:2])
    sp, cond, ct = st.split("|")
    ia_all = g.index[g["dataset"] == ds_a].to_numpy()
    ib_all = g.index[g["dataset"] == ds_b].to_numpy()
    ia = rng.choice(ia_all, min(N_SIDE, len(ia_all)), replace=False)
    ib = rng.choice(ib_all, min(N_SIDE, len(ib_all)), replace=False)
    # 1) 跨数据集
    ta, tb = read_tok(ia), read_tok(ib)
    a1, k1 = auc_pair(ta, tb)
    rows.append(dict(stratum=st, kind="cross_dataset", n=len(ta),
                     auc_lr=round(a1, 4), auc_knn=round(k1, 4)))
    print(json.dumps(rows[-1]), flush=True)
    # 2) 跨数据集 + 深度匹配
    ibm = depth_matched_pick(ia, ib_all)
    tbm = read_tok(ibm)
    a2, k2 = auc_pair(ta, tbm)
    rows.append(dict(stratum=st, kind="cross_dataset_depth_matched", n=len(ta),
                     auc_lr=round(a2, 4), auc_knn=round(k2, 4)))
    print(json.dumps(rows[-1]), flush=True)
    del tb, tbm
    # 3) sanity: 同数据集随机二分
    rest = np.setdiff1d(ia_all, ia)
    ic = rng.choice(rest, min(N_SIDE, len(rest)), replace=False)
    tc = read_tok(ic)
    a3, k3 = auc_pair(ta, tc)
    rows.append(dict(stratum=st, kind="within_dataset_random", n=len(ta),
                     auc_lr=round(a3, 4), auc_knn=round(k3, 4)))
    print(json.dumps(rows[-1]), flush=True)
    del ta, tc
    # 4) 生成 vs 单一数据集真实
    gi = np.where((g_sp == sp) & (g_cond == cond) & (g_ct == ct))[0]
    if len(gi) >= 2000:
        gi = rng.choice(gi, min(N_SIDE, len(gi)), replace=False)
        a4, k4 = auc_pair(gen_tok[gi], read_tok(ia))
        rows.append(dict(stratum=st, kind="gen_vs_single_dataset", n=len(gi),
                         auc_lr=round(a4, 4), auc_knn=round(k4, 4)))
        print(json.dumps(rows[-1]), flush=True)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("RVR2_DONE", flush=True)
