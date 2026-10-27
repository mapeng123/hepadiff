#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miller et al. Nat Biotechnol 2026 式校准评测: 零样本 KO 到底能不能打?
对每个留出基因, 四方对比 (全部在 delta vs 对照 尺度, 16k 基因 panel):
  模型预测   = eval_v5_full 已存 deltas_*.npz (v5-155k, 10 seeds 平均)
  阴性对照   = LOO 平均基线 (同数据集其他扰动的平均 delta; K562 剔除全部 holdout 基因)
  阳性对照   = 插值重复 (扰动细胞分半: alpha 来自 A 半 vs 对照的 DE 校正P, 与 WMSE 权重(B 半)不互通)
  真值 y     = B 半平均 - 对照平均
指标: MSE / Pearson(dctrl) / WMSE / 加权R2dctrl; 并给每把尺子算 DRF, 给模型算
  "理想差距达成率" rel = (m(model)-m(MB))/(m(y)-m(MB)) (0=平均基线, 1=完美)
用法: python eval_v5_calibrated.py [quick]
"""
import os, sys, json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

PERT_DIR = "/root/autodl-tmp/perturb_data"
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
DELTA_DIR = "/root/autodl-tmp/eval_v5_full"
MANIFEST = "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5/holdout_manifest.json"
OUT = "/root/autodl-tmp/eval_v5_calibrated"
os.makedirs(OUT, exist_ok=True)
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
RNG = np.random.default_rng(2026)

genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
edges = np.load(EDGES_F)
LUT = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)
manifest = json.load(open(MANIFEST))
HOLDOUT = set(manifest["tier1_40"]) | set(manifest["tier2_150"]) | set(manifest.get("tier5_demo_targets", []))

import anndata as ad

def tokenize_h5ad(path, obs_col, ctrl_label, max_cells=None, per_pert=300, seed=0):
    cache = path + ".tokcache.npz"
    if os.path.exists(cache) and max_cells in (None, 400000) and per_pert == 300 and seed == 0:
        z = np.load(cache, allow_pickle=True)
        print(f"  {os.path.basename(path)}: 缓存命中 {z['tok'].shape[0]} 细胞", flush=True)
        return z["tok"], z["pert"]
    a = ad.read_h5ad(path)
    common = [g for g in a.var.index if g in gidx]
    if len(common) < 1000:
        for cn in ("gene_name", "gene_symbol"):
            if cn in a.var.columns:
                a.var_names = a.var[cn].astype(str).to_numpy()
                a.var_names_make_unique()
                common = [g for g in a.var.index if g in gidx]
                if len(common) >= 1000:
                    break
    a = a[:, common]
    if max_cells and a.n_obs > max_cells:
        pv = a.obs[obs_col].astype(str).to_numpy()
        srs = pd.Series(np.arange(a.n_obs), index=pv)
        keep = srs.groupby(level=0, sort=False).apply(
            lambda g: g.to_numpy() if (g.name == ctrl_label or len(g) <= per_pert)
            else g.sample(per_pert, random_state=seed).to_numpy())
        keep = np.concatenate([np.atleast_1d(x) for x in keep])
        if len(keep) > max_cells:
            keep = np.random.default_rng(seed).choice(keep, max_cells, replace=False)
        a = a[keep, :]
    col = np.array([gidx[g] for g in common])
    tok = np.zeros((a.n_obs, G), np.int8)
    BLK = 50000
    for i0 in range(0, a.n_obs, BLK):
        i1 = min(i0 + BLK, a.n_obs)
        Xb = a.X[i0:i1]
        if not isinstance(Xb, np.ndarray):
            Xb = Xb.toarray()
        Xb = np.log1p(Xb / Xb.sum(1, keepdims=True).clip(1) * 1e4).astype(np.float32)
        tok[i0:i1, col] = np.searchsorted(edges, Xb).astype(np.int8)
        del Xb
    pert = a.obs[obs_col].astype(str).to_numpy()
    pert[pert == ctrl_label] = "control"
    return tok, pert

def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    return out

def lin(x):  # int8 tokens -> linearized expression
    return LUT[x.astype(int)]

def de_padj(cells_tok, ctrl_lin64):
    """cells vs ctrl(已线性化 float64) 的 Welch t 检验 + BH. cells_tok 为 token 矩阵."""
    a = lin(cells_tok).astype(np.float64)
    with np.errstate(all="ignore"):
        _, p = ttest_ind(a, ctrl_lin64, axis=0, equal_var=False)
    return bh(p)

def metrics(pred, y, w):
    """pred/y: (G,) delta 向量; w: WMSE 权重. 返回 mse, pearson, wmse, wr2"""
    mse = float(np.mean((pred - y) ** 2))
    pear = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 0 and pred.std() > 0 else np.nan
    sw = w.sum()
    if sw <= 1e-9:
        wmse, wr2 = np.nan, np.nan
    else:
        wmse = float(np.sum(w * (pred - y) ** 2) / sw)
        ybar = float(np.sum(w * y) / sw)
        sst = float(np.sum(w * (y - ybar) ** 2))
        wr2 = float(1 - np.sum(w * (pred - y) ** 2) / sst) if sst > 1e-12 else np.nan
    return mse, pear, wmse, wr2

EPS = 1e-8

def drf(m_pos, m_neg, higher_better):
    """DRF = (m_pos - m_neg) / (m_perf - m_neg). m_perf: MSE 类=0, 相关类=1."""
    m_perf = 1.0 if higher_better else 0.0
    den = (m_perf - m_neg)
    if abs(den) < EPS:
        return np.nan
    return (m_pos - m_neg) / den

rows = []

def run_context(ctx_name, tok, pert, eval_gene_list, tier_of, exclude_from_baseline):
    ctrl = tok[pert == "control"]
    if len(ctrl) < 50:
        print(f"  {ctx_name}: 对照不足, 跳过", flush=True); return
    ctrl_sub = ctrl[RNG.choice(len(ctrl), min(3000, len(ctrl)), replace=False)]
    ctrl_lin64 = lin(ctrl_sub).astype(np.float64)
    Ec = lin(ctrl).mean(0)
    # LOO 平均基线池
    vc = pd.Series(pert).value_counts()
    pool = [p for p in vc.index if p != "control" and vc[p] >= 30 and p not in exclude_from_baseline]
    pool_delta = {}
    print(f"  {ctx_name}: 基线池 {len(pool)} 扰动, 对照 {len(ctrl)} 细胞", flush=True)
    for g in eval_gene_list:
        idx = np.where(pert == g)[0]
        if len(idx) < 30:
            print(f"  {g}: 细胞不足 ({len(idx)}), 跳过", flush=True); continue
        # 分半
        h = RNG.permutation(idx)
        A, B = h[: len(h) // 2], h[len(h) // 2:]
        y = lin(tok[B]).mean(0) - Ec                      # 真值 (B 半)
        td = lin(tok[A]).mean(0) - Ec                     # 技术重复 (A 半)
        alpha = 1.0 - de_padj(tok[A], ctrl_lin64)         # 只用 A 半 (防泄漏)
        w = 1.0 - de_padj(tok[B], ctrl_lin64)             # WMSE 权重只用 B 半
        # LOO 平均基线
        mb = np.zeros(G, np.float64); cnt = 0
        for p in pool:
            if p == g: continue
            if p not in pool_delta:
                pool_delta[p] = lin(tok[pert == p]).mean(0) - Ec
            mb += pool_delta[p]; cnt += 1
        mb /= max(cnt, 1)
        idup = alpha * td + (1 - alpha) * mb              # 插值重复 (阳性对照)
        # 模型预测 (npz 已存)
        pred = PRED_DELTAS[ctx_name][g]
        n_deg = int((w > 0.5).sum())
        r = dict(context=ctx_name, tier=tier_of(g), gene=g, n_cells=len(idx), n_deg=n_deg)
        for tag, pr in [("model", pred), ("meanbase", mb), ("posdup", idup)]:
            m = metrics(pr, y, w)
            for mi, val in zip(["mse", "pearson", "wmse", "wr2"], m):
                r[f"{tag}_{mi}"] = val
        # DRF (尺子体检) 与模型理想差距达成率
        for mi, hb in [("mse", False), ("pearson", True), ("wmse", False), ("wr2", True)]:
            m_neg, m_pos, m_mod = r[f"meanbase_{mi}"], r[f"posdup_{mi}"], r[f"model_{mi}"]
            if np.isnan(m_neg) or np.isnan(m_pos):
                r[f"drf_{mi}"] = np.nan; r[f"rel_{mi}"] = np.nan; continue
            r[f"drf_{mi}"] = drf(m_pos, m_neg, hb)
            m_perf = 1.0 if hb else 0.0
            den = m_perf - m_neg
            r[f"rel_{mi}"] = (m_mod - m_neg) / den if abs(den) > EPS else np.nan
        rows.append(r)
        print(f"  {ctx_name} {g}: nDEG={n_deg} DRFmse={r['drf_mse']:.2f} DRFwmse={r['drf_wmse']:.2f} "
              f"relPear={r['rel_pearson']:.2f} relWMSE={r['rel_wmse']:.2f}", flush=True)

# ---- 载入模型预测 delta ----
PRED_DELTAS = {}
for ctx, f in [("k562", "deltas_tier1.npz"), ("k562_t2", "deltas_tier2.npz"),
               ("hepg2", "deltas_tier3_hepg2.npz"), ("jurkat", "deltas_tier3_jurkat.npz"),
               ("organoid_gi", "deltas_tier3_organoid_gi.npz")]:
    fp = f"{DELTA_DIR}/{f}"
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        PRED_DELTAS.setdefault("k562" if ctx.startswith("k562") else ctx, {}).update(
            {g: p for g, p in zip(z["genes"], z["pred"])})
        print(f"载入 {f}: {len(z['genes'])} 基因", flush=True)

# ---- K562: Tier1 + Tier2 ----
print("== K562-gwps ==", flush=True)
tok, pert = tokenize_h5ad(f"{PERT_DIR}/ReplogleWeissman2022_K562_gwps_400k.h5ad",
                          "perturbation", "control", max_cells=400000)
t1 = json.load(open(MANIFEST))["tier1_40"]
t1 = [g for g in t1 if g in PRED_DELTAS["k562"]]
t2 = [g for g in manifest["tier2_150"] if g in PRED_DELTAS["k562"]]
if QUICK:
    t1, t2 = t1[:5], []
tier_of_k = lambda g: "tier1" if g in set(t1) else "tier2"
run_context("k562", tok, pert, t1 + t2, tier_of_k, HOLDOUT)
del tok, pert

# ---- Tier3 LOCO 三体系 ----
if not QUICK:
    for ctx, hp, oc, cl in [
        ("hepg2", f"{PERT_DIR}/GSE264667_hepg2_raw_singlecell_01.h5ad", "gene", "non-targeting"),
        ("jurkat", f"{PERT_DIR}/GSE264667_jurkat_raw_singlecell_01.h5ad", "gene", "non-targeting"),
        ("organoid_gi", f"{PERT_DIR}/Lo2025_organoid_perturbseq_GSE280506.h5ad", "perturbation", "control")]:
        if ctx not in PRED_DELTAS or not os.path.exists(hp):
            print(f"!! {ctx} 缺预测或 h5ad, 跳过", flush=True); continue
        print(f"== {ctx} LOCO ==", flush=True)
        tok_h, pert_h = tokenize_h5ad(hp, oc, cl, max_cells=200000)
        gs = [g for g in PRED_DELTAS[ctx] if g in gidx]
        run_context(ctx, tok_h, pert_h, gs, lambda g: "tier3_loco", set())
        del tok_h, pert_h

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/calibrated_per_gene.csv", index=False)

# ---- 汇总: 每 tier x 指标: 尺子 DRF + 模型 vs 平均基线 Wilcoxon + 中位达成率 ----
summ = {}
for (ctx, tier), sub in df.groupby(["context", "tier"]):
    key = f"{ctx}:{tier}"
    ent = {"n": len(sub), "n_deg_median": float(sub.n_deg.median())}
    for mi, hb in [("mse", False), ("pearson", True), ("wmse", False), ("wr2", True)]:
        ent[f"drf_{mi}_median"] = float(sub[f"drf_{mi}"].median())
        ent[f"rel_{mi}_median"] = float(sub[f"rel_{mi}"].median())
        diff = (sub[f"meanbase_{mi}"] - sub[f"model_{mi}"]) if not hb else (sub[f"model_{mi}"] - sub[f"meanbase_{mi}"])
        diff = diff.dropna()
        if len(diff) >= 10:
            wstat, p = wilcoxon(diff)
            ent[f"model_vs_meanbase_{mi}"] = dict(frac_better=float((diff > 0).mean()), wilcoxon_p=float(p))
    summ[key] = ent
json.dump(summ, open(f"{OUT}/calibrated_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
print("CALIBRATED_DONE", flush=True)
