#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P-MDDM v4 全套评估 (回应 JHP R1-1/2/3, R2-1):
  Tier1: 40 VCC 基因 (训练全排除) — pearson delta / 方向一致 / PDS / ridge / mean-delta
  Tier2: 150 非必需基因 — 同上 + 真实delta互相关分布 (对照 Tier1 的 0.7-0.9) + PDS top-1
  Tier3: HepG2 跨上下文 (全系未见) — 零样本 KO vs 真实 delta; delta-transfer 基线对比
  对照: shuffled gene-identity embedding (R1-3 点名) — 打乱后机制恢复应崩塌
用法: python eval_v4_full.py [step]
"""
import os, re, glob, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = os.environ.get("EVAL_DEV", "cuda")
BONE_MM = "/root/autodl-tmp/virtual-liver/processed/v4_mm_r1"
PERT_DIR = "/root/autodl-tmp/perturb_data"
CKPT_DIR = os.environ.get("HPD_CKPT_DIR", "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5")
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
OUT = "/root/autodl-tmp/eval_v5_full"
os.makedirs(OUT, exist_ok=True)

D_MODEL, N_HEAD, N_LAYER = 512, 8, 12
CHUNK = 48
MASK_TOKEN = 51
GID_DIM = 256
ALPHA = -0.5
N_GEN = 500
DE_TOPN = 200
DIR_TOPN = 100
N_SEEDS = 10

TIER1_40 = ["KRR1","FCF1","TSR2","DDX47","TIMM23B","EIF2B2","QARS","CHMP6","AATF","EIF2B5",
            "NOP16","HSPA9","SAMM50","MED19","RPL24","GPN3","EXOSC2","KDM1A","RPL23","HARS",
            "MED22","MED12","MED11","WDR82","HSD17B12","FARSA","GTF2H1","EXOSC8","NOL11","POLR1A",
            "ERCC2","SUPT5H","RARS2","TAF2","TAF8","ZC3H13","NKX6-1","EIF3D","PSMD7","DBR1"]

genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}

# 预注册剔除: Tier1 中不在 16k 基因表的基因 (必需基因表达稳定, 未入 tokenized 基因表)
# 不重训不改表, 按预注册清单如实剔除并在 summary 中记录
TIER1_MISSING = [g for g in TIER1_40 if g not in gidx]
TIER1 = [g for g in TIER1_40 if g in gidx]
if TIER1_MISSING:
    print(f"!! Tier1 预注册剔除 {len(TIER1_MISSING)} 个不在基因表的基因: {TIER1_MISSING}", flush=True)
N_CHUNK = (G + CHUNK - 1) // CHUNK
G_PAD = N_CHUNK * CHUNK
edges = np.load(EDGES_F)
_lut_np = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

cks = sorted(glob.glob(f"{CKPT_DIR}/pmdm_v4_step*.pt"),
             key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
assert cks, "无 v4 检查点"
if len(sys.argv) > 1:
    want = int(sys.argv[1])
    cks = [c for c in cks if int(re.search(r"step(\d+)", c).group(1)) == want] or cks[-1:]
ck = torch.load(cks[-1], map_location=DEV, weights_only=False)
STEP = ck["step"]
ct_vocab, cond_vocab, pert_vocab = ck["ct_vocab"], ck["cond_vocab"], ck["pert_vocab"]
N_CT, N_COND, N_PERT = len(ct_vocab) + 1, len(cond_vocab) + 1, len(pert_vocab) + 1
NULL_CT, NULL_COND, NULL_PERT = len(ct_vocab), len(cond_vocab), len(pert_vocab)
SPV = ck.get("species_vocab", {"human": 0, "mouse": 1, "rat": 2, "cellline": 3})
N_SP, NULL_SP = len(SPV) + 1, len(SPV)
PTV = ck.get("ptype_vocab", {"LOF": 0, "GOF": 1, "compound": 2})
N_PTYPE = 4
manifest = json.load(open(f"{CKPT_DIR}/holdout_manifest.json"))
TIER2 = manifest["tier2_150"]
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
if QUICK:
    N_GEN = 300
print(f"评估 {cks[-1]} (step {STEP}) | Tier2 n={len(TIER2)} | quick={QUICK}", flush=True)

pert_gene_idx = np.full(N_PERT, -1, dtype=np.int64)
for p, i in pert_vocab.items():
    if p in gidx:
        pert_gene_idx[i] = gidx[p]

class PMDDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gene_emb = nn.Embedding(52, D_MODEL)
        self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, N_CHUNK + 5, D_MODEL) * 0.02)
        self.ct_emb = nn.Embedding(N_CT, D_MODEL)
        self.cond_emb = nn.Embedding(N_COND, D_MODEL)
        self.sp_emb = nn.Embedding(N_SP, D_MODEL)
        self.ptype_emb = nn.Embedding(N_PTYPE, GID_DIM)
        self.gene_id_emb = nn.Embedding(G, GID_DIM)
        self.pert_gene_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.pert_free_emb = nn.Embedding(N_PERT, GID_DIM)
        self.pert_free_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.delta_head = nn.Sequential(nn.Linear(D_MODEL * 3, 512), nn.GELU(),
                                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, G))
        self.z0_proj = nn.Linear(D_MODEL, D_MODEL)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, batch_first=True,
                                       norm_first=True, activation="gelu"), N_LAYER)
        self.decompress = nn.Linear(D_MODEL, CHUNK * 52)

    def z0_token(self, x0):
        e = self.gene_emb(x0.long()).view(x0.shape[0], N_CHUNK, CHUNK * D_MODEL)
        return self.z0_proj(self.compress(e).mean(1))

    def pert_vector(self, pert, pgid, ptype):
        v = torch.zeros(pert.shape[0], D_MODEL, device=pert.device)
        is_null = pert.eq(NULL_PERT)
        pt = self.ptype_emb(ptype.clamp(0, N_PTYPE - 1))
        gm = (pgid >= 0) & ~is_null
        if gm.any():
            v[gm] = self.pert_gene_mlp(self.gene_id_emb(pgid[gm]) + pt[gm])
        fm = (pgid < 0) & ~is_null
        if fm.any():
            v[fm] = self.pert_free_mlp(self.pert_free_emb(pert[fm]) + pt[fm])
        return v

    def forward(self, x, ct, cond, pert, z0, pgid, ptype, sp):
        B = x.shape[0]
        e = self.gene_emb(x.long()).view(B, N_CHUNK, CHUNK * D_MODEL)
        z = self.compress(e)
        pv = self.pert_vector(pert, pgid, ptype)
        head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0, self.sp_emb(sp)], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 5:]
        return self.decompress(z).view(B, G_PAD, 52)

model = PMDDM().to(DEV)
model.load_state_dict(ck["model"]); model.eval()
del ck

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
        for cn in ('gene_name', 'gene_symbol'):
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
    pert[pert == ctrl_label] = 'control'
    return tok, pert

def generate(model, ct_id, pert_id, pgid, z0, n=N_GEN, ptype_id=0, sp_id=SPV["cellline"]):
    outs = []
    for i0 in range(0, n, 64):
        b = min(250, n - i0)
        xm = torch.full((b, G_PAD), MASK_TOKEN, dtype=torch.int8, device=DEV)
        ct = torch.full((b,), ct_id, dtype=torch.long, device=DEV)
        cond = torch.full((b,), NULL_COND, dtype=torch.long, device=DEV)
        pt = torch.full((b,), pert_id, dtype=torch.long, device=DEV)
        pg = torch.full((b,), pgid, dtype=torch.long, device=DEV)
        pty = torch.full((b,), ptype_id, dtype=torch.long, device=DEV)
        sp = torch.full((b,), sp_id, dtype=torch.long, device=DEV)
        with torch.no_grad():
            logits = model(xm, ct, cond, pt, z0.expand(b, -1), pg, pty, sp).float()
        logits[..., 0] += ALPHA
        t = torch.distributions.Categorical(logits=logits.view(-1, 52)).sample().view(b, G_PAD)
        outs.append(t[:, :G].cpu().numpy().astype(np.int8))
    return np.concatenate(outs)

def generate_multi(model, ct_id, pert_id, pgid, z0, n=N_GEN, n_seeds=1, ptype_id=0, sp_id=SPV["cellline"]):
    """logits 与种子无关: 一次前向, 多次采样 (种子间唯一差异在采样)"""
    per_seed = [[] for _ in range(n_seeds)]
    for i0 in range(0, n, 64):
        b = min(64, n - i0)
        xm = torch.full((b, G_PAD), MASK_TOKEN, dtype=torch.int8, device=DEV)
        ct = torch.full((b,), ct_id, dtype=torch.long, device=DEV)
        cond = torch.full((b,), NULL_COND, dtype=torch.long, device=DEV)
        pt = torch.full((b,), pert_id, dtype=torch.long, device=DEV)
        pg = torch.full((b,), pgid, dtype=torch.long, device=DEV)
        pty = torch.full((b,), ptype_id, dtype=torch.long, device=DEV)
        sp = torch.full((b,), sp_id, dtype=torch.long, device=DEV)
        with torch.no_grad():
            logits = model(xm, ct, cond, pt, z0.expand(b, -1), pg, pty, sp).float()
        logits[..., 0] += ALPHA
        flat = logits.view(-1, 52)
        for s in range(n_seeds):
            t = torch.distributions.Categorical(logits=flat).sample().view(b, G_PAD)
            per_seed[s].append(t[:, :G].cpu().numpy().astype(np.int8))
    return [np.concatenate(o) for o in per_seed]

def eval_one_gene(g, d_true, Ec, ct_id, z0, n_seeds=N_SEEDS, shuffle=False, pgid_override=None):
    """生成并评估单基因 KO. shuffle=True 时打乱基因身份 (R1-3 对照)"""
    if g in pert_vocab:
        pid = pert_vocab[g]
    elif g in gidx:
        # 未见基因零样本: 必须给非 null 的 pert token — pert_vector 中 is_null 会清零 v.
        # pert id 本身不进 v (gm 路径只用 gene_id_emb(pgid)+ptype), 任意非 null id 等效.
        pid = 0
    else:
        pid = NULL_PERT
    pgid = gidx[g] if g in gidx else -1
    if pgid_override is not None:
        pgid = pgid_override
    rs, ds_, pds_ = [], [], []
    gens = generate_multi(model, ct_id, pid, pgid, z0, n=N_GEN, n_seeds=n_seeds)
    for sd in range(n_seeds):
        Eg = _lut_np[gens[sd].astype(int)].mean(0)
        d_pred = Eg - Ec
        pds_.append(d_pred)
        top = np.argsort(-np.abs(d_true))[:DE_TOPN]
        rs.append(np.corrcoef(d_pred[top], d_true[top])[0, 1])
        td = np.argsort(-np.abs(d_true))[:DIR_TOPN]
        ds_.append(float((np.sign(d_pred[td]) == np.sign(d_true[td])).mean()))
    return np.mean(rs), np.std(rs), np.mean(ds_), np.std(ds_), np.stack(pds_)

def pds_metrics(pred_all, true_all):
    """pred_all/true_all: (n_genes, G). 返回 top1, 归一化PDS"""
    C = np.corrcoef(np.concatenate([pred_all, true_all]))[:len(pred_all), len(pred_all):]
    ranks = []
    for i in range(len(pred_all)):
        order = np.argsort(-C[i])
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks = np.array(ranks)
    return float((ranks == 1).mean()), float(1 - (ranks.mean() - 1) / (len(pred_all) - 1)), ranks

summary = {}
summary["tier1_excluded_not_in_vocab"] = TIER1_MISSING
summary["tier1_n_preregistered"] = len(TIER1_40)
summary["tier1_n_evaluated"] = len(TIER1)

# ============ 加载 K562-gwps (Tier1+Tier2) ============
print("加载 K562-gwps ...", flush=True)
tok, pert = tokenize_h5ad(f"{PERT_DIR}/ReplogleWeissman2022_K562_gwps_400k.h5ad",
                          'perturbation', 'control', max_cells=400000)
ctrl = tok[pert == 'control']
Ec_k562 = _lut_np[ctrl.astype(int)].mean(0)
ct_k562 = ct_vocab.get('K562-gwps', NULL_CT)
with torch.no_grad():
    z0_k562 = model.z0_token(torch.from_numpy(
        np.pad(ctrl[:300], ((0, 0), (0, G_PAD - G)))).to(DEV)).mean(0, keepdim=True)

def true_delta(tok, pert, g, Ec):
    idx = tok[pert == g]
    if len(idx) < 10:
        return None
    return _lut_np[idx.astype(int)].mean(0) - Ec

_qs = 3 if QUICK else N_SEEDS
_tiers = [("tier2", TIER2)] if QUICK else [("tier1", TIER1), ("tier2", TIER2)]
for tier_name, tier_genes in _tiers:
    print(f"== {tier_name} ({len(tier_genes)} 基因) ==", flush=True)
    rows, pred_all, true_all, names = [], [], [], []
    for g in tier_genes:
        d_true = true_delta(tok, pert, g, Ec_k562)
        if d_true is None:
            print(f"  {g}: gwps 中细胞不足, 跳过", flush=True)
            continue
        r, rsd, dc, dcsd, pd_ = eval_one_gene(g, d_true, Ec_k562, ct_k562, z0_k562, n_seeds=_qs)
        rows.append(dict(tier=tier_name, gene=g, pearson=r, pearson_sd=rsd, dir=dc, dir_sd=dcsd,
                         effect=float(np.abs(d_true).mean())))
        pred_all.append(pd_.mean(0)); true_all.append(d_true); names.append(g)
        print(f"  {g:12s} pearson={r:.3f} dir={dc:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/eval_v4_{tier_name}.csv", index=False)
    top1, pdsn, _ = pds_metrics(np.stack(pred_all), np.stack(true_all))
    np.savez(f"{OUT}/deltas_{tier_name}.npz", pred=np.stack(pred_all), true=np.stack(true_all), genes=np.array(names))
    # 真实 delta 互相关 (评估集内在难度)
    TT = np.corrcoef(np.stack(true_all))
    iu = TT[np.triu_indices(len(TT), 1)]
    summary[tier_name] = dict(n=len(df), pearson=float(df.pearson.mean()), dir=float(df.dir.mean()),
                              pds_top1=top1, pds_norm=pdsn,
                              true_delta_intercorr_median=float(np.median(iu)),
                              true_delta_intercorr_q90=float(np.quantile(iu, 0.9)))
    print(f"== {tier_name} 汇总: pearson={df.pearson.mean():.3f} dir={df.dir.mean():.3f} "
          f"PDS top1={top1:.3f} norm={pdsn:.3f} | 真实delta互相关 med={np.median(iu):.2f}", flush=True)

if QUICK:
    json.dump(summary, open(f"{OUT}/quick_tier2_step{STEP}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("QUICK_TIER2_DONE", flush=True)
    sys.exit(0)

# ---- Tier1 shuffled gene-identity 对照 (R1-3) ----
print("== Tier1 shuffled embedding 对照 ==", flush=True)
rng = np.random.default_rng(7)
shuf_rows = []
perm = rng.permutation(len(TIER1))
for i, g in enumerate(TIER1):
    d_true = true_delta(tok, pert, g, Ec_k562)
    if d_true is None:
        continue
    wrong_gene = TIER1[perm[i]]
    if wrong_gene == g:
        wrong_gene = TIER1[perm[(i + 1) % len(TIER1)]]
    if wrong_gene not in gidx:
        print(f"  {g}: 配对基因 {wrong_gene} 不在基因表, shuffled 跳过", flush=True)
        continue
    r, _, dc, _, _ = eval_one_gene(g, d_true, Ec_k562, ct_k562, z0_k562,
                                   n_seeds=3, pgid_override=gidx[wrong_gene])
    shuf_rows.append(dict(gene=g, wrong_identity=wrong_gene, pearson_shuffled=r, dir_shuffled=dc))
dfs = pd.DataFrame(shuf_rows)
dfs.to_csv(f"{OUT}/eval_v4_tier1_shuffled.csv", index=False)
summary["tier1_shuffled"] = dict(pearson=float(dfs.pearson_shuffled.mean()),
                                 dir=float(dfs.dir_shuffled.mean()))
print(f"  shuffled: pearson={dfs.pearson_shuffled.mean():.3f} (原版 {summary['tier1']['pearson']:.3f})",
      flush=True)

# ---- mean-delta / ridge 基线 (同 v3 口径) ----
all_d = []
for p in pd.unique(pert[pert != 'control']):
    idx = tok[pert == p]
    if len(idx) >= 30:
        all_d.append(_lut_np[idx.astype(int)].mean(0) - Ec_k562)
mean_d = np.mean(all_d, axis=0)
for tier_name, tier_genes in [("tier1", TIER1), ("tier2", TIER2)]:
    rs = []
    for g in tier_genes:
        d_true = true_delta(tok, pert, g, Ec_k562)
        if d_true is None: continue
        top = np.argsort(-np.abs(d_true))[:DE_TOPN]
        rs.append(np.corrcoef(mean_d[top], d_true[top])[0, 1])
    summary[tier_name]["mean_delta_baseline"] = float(np.mean(rs))

# ============ Tier3: 跨上下文 LOCO (HepG2 + Jurkat) + Tier3b 类器官独立验证 ============
TIER3_SETS = [
    ("hepg2", f"{PERT_DIR}/GSE264667_hepg2_raw_singlecell_01.h5ad", "gene", "non-targeting"),
    ("jurkat", f"{PERT_DIR}/GSE264667_jurkat_raw_singlecell_01.h5ad", "gene", "non-targeting"),
    # Tier3b: Lo 2025 Nat Commun 人源 3D 胃类器官 iCRISPRi/a Perturb-seq (GSE280506)
    # 非细胞系、人源 3D 组织、从未进训练 — R1 "independent liver-relevant perturbation" 的最强可用替代
    ("organoid_gi", f"{PERT_DIR}/Lo2025_organoid_perturbseq_GSE280506.h5ad", "perturbation", "control"),
]
for tier3_name, hp, obs_col, ctrl_lab in TIER3_SETS:
    if not os.path.exists(hp):
        print(f"!! 缺 {hp}, {tier3_name} 跳过 — 需下载", flush=True)
        summary[f"tier3_{tier3_name}"] = "SKIPPED: h5ad missing"
        continue
    print(f"== Tier3: {tier3_name} 跨上下文 ==", flush=True)
    tok_h, pert_h = tokenize_h5ad(hp, obs_col, ctrl_lab, max_cells=200000)
    ctrl_h = tok_h[pert_h == 'control']
    if len(ctrl_h) < 50:
        print(f"  {tier3_name}: 对照细胞不足, 跳过", flush=True)
        summary[f"tier3_{tier3_name}"] = "SKIPPED: too few controls"
        continue
    Ec_h = _lut_np[ctrl_h.astype(int)].mean(0)
    with torch.no_grad():
        z0_h = model.z0_token(torch.from_numpy(
            np.pad(ctrl_h[:300], ((0, 0), (0, G_PAD - G)))).to(DEV)).mean(0, keepdim=True)
    # 上下文通道: ct=NULL + z0(该体系对照) — 跨上下文迁移走 z0
    hperts = [p for p in pd.unique(pert_h[pert_h != 'control'])
              if (pert_h == p).sum() >= 30 and p in gidx]
    rows3, pred3, true3 = [], [], []
    for g in hperts:
        d_true = true_delta(tok_h, pert_h, g, Ec_h)
        if d_true is None: continue
        r, rsd, dc, dcsd, pd_ = eval_one_gene(g, d_true, Ec_h, NULL_CT, z0_h, n_seeds=5)
        seen_elsewhere = g in pert_vocab
        # delta-transfer 基线: 同基因 K562-gwps delta 直接搬到该体系基线
        d_k = true_delta(tok, pert, g, Ec_k562) if seen_elsewhere else None
        r_transfer = np.nan
        if d_k is not None:
            top = np.argsort(-np.abs(d_true))[:DE_TOPN]
            r_transfer = float(np.corrcoef(d_k[top], d_true[top])[0, 1])
        rows3.append(dict(gene=g, pearson=r, pearson_sd=rsd, dir=dc,
                          seen_in_other_lines=seen_elsewhere, transfer_r=r_transfer))
        pred3.append(pd_.mean(0)); true3.append(d_true)
        print(f"  {g:12s} pearson={r:.3f} dir={dc:.3f} transfer={r_transfer:.3f}", flush=True)
    df3 = pd.DataFrame(rows3)
    df3.to_csv(f"{OUT}/eval_v4_tier3_{tier3_name}.csv", index=False)
    if len(df3) >= 5:
        top1_3, pdsn_3, _ = pds_metrics(np.stack(pred3), np.stack(true3))
        np.savez(f"{OUT}/deltas_tier3_{tier3_name}.npz", pred=np.stack(pred3), true=np.stack(true3), genes=np.array([r["gene"] for r in rows3]))
        summary[f"tier3_{tier3_name}"] = dict(
            n=len(df3), pearson=float(df3.pearson.mean()), dir=float(df3.dir.mean()),
            pearson_seen=float(df3[df3.seen_in_other_lines].pearson.mean()) if df3.seen_in_other_lines.any() else None,
            pearson_unseen=float(df3[~df3.seen_in_other_lines].pearson.mean()) if (~df3.seen_in_other_lines).any() else None,
            transfer_r=float(df3.transfer_r.dropna().mean()) if df3.transfer_r.notna().any() else None,
            pds_top1=top1_3, pds_norm=pdsn_3)
        print(f"== {tier3_name} 汇总: model={df3.pearson.mean():.3f} vs delta-transfer="
              f"{summary[f'tier3_{tier3_name}']['transfer_r']} (n={len(df3)}) ==", flush=True)
    else:
        summary[f"tier3_{tier3_name}"] = f"SKIPPED: only {len(df3)} evaluable genes"

# ============ Tier4: 跨细胞系一致性 (R1-1 K562 占比问题的直接证据) ============
# K562-essential × RPE1 共享基因: 模型预测的"系间 delta 差异" vs 真实"系间差异"
# 直接检验 perturbation-by-context 交互, 排除"模型只学上下文无关 delta"的备择解释
f_ess = f"{PERT_DIR}/ReplogleWeissman2022_K562_essential.h5ad"
f_rpe = f"{PERT_DIR}/ReplogleWeissman2022_rpe1.h5ad"
if os.path.exists(f_ess) and os.path.exists(f_rpe):
    print("== Tier4: 跨细胞系一致性 (K562-essential vs RPE1) ==", flush=True)
    tok_e, pert_e = tokenize_h5ad(f_ess, 'perturbation', 'control')
    tok_r, pert_r = tokenize_h5ad(f_rpe, 'perturbation', 'control')
    ctrl_e = tok_e[pert_e == 'control']; ctrl_r = tok_r[pert_r == 'control']
    Ec_e = _lut_np[ctrl_e.astype(int)].mean(0); Ec_r = _lut_np[ctrl_r.astype(int)].mean(0)
    ct_e = ct_vocab.get('K562-essential', NULL_CT); ct_r = ct_vocab.get('RPE1', NULL_CT)
    with torch.no_grad():
        z0_e = model.z0_token(torch.from_numpy(np.pad(ctrl_e[:300], ((0, 0), (0, G_PAD - G)))).to(DEV)).mean(0, keepdim=True)
        z0_r = model.z0_token(torch.from_numpy(np.pad(ctrl_r[:300], ((0, 0), (0, G_PAD - G)))).to(DEV)).mean(0, keepdim=True)
    pe = pd.Series((pert_e != 'control')).to_numpy()
    cnt_e = pd.Series(pert_e[pe]).value_counts(); cnt_r = pd.Series(pert_r[pert_r != 'control']).value_counts()
    TIER4_HO = set(manifest.get("tier4_100", []))
    if TIER4_HO:
        shared = sorted(set(cnt_e[cnt_e >= 30].index) & set(cnt_r[cnt_r >= 30].index)
                        & TIER4_HO & set(gidx.keys()))
    else:  # 旧 manifest 回退: 仅限 Tier1/Tier2 留出基因
        shared = sorted(set(cnt_e[cnt_e >= 30].index) & set(cnt_r[cnt_r >= 30].index)
                        & (set(TIER1_40) | set(TIER2)) & set(gidx.keys()))
    rng4 = np.random.default_rng(4)
    if len(shared) > 100:
        shared = sorted(rng4.choice(shared, 100, replace=False).tolist())
    print(f"  共享可评估基因 {len(shared)}", flush=True)
    rows4 = []
    for g in shared:
        d_te = true_delta(tok_e, pert_e, g, Ec_e); d_tr = true_delta(tok_r, pert_r, g, Ec_r)
        if d_te is None or d_tr is None: continue
        r_e, _, _, _, pd_e = eval_one_gene(g, d_te, Ec_e, ct_e, z0_e, n_seeds=3)
        r_r, _, _, _, pd_r = eval_one_gene(g, d_tr, Ec_r, ct_r, z0_r, n_seeds=3)
        # 系间差异: 预测差 vs 真实差 (top200 并集)
        top = np.union1d(np.argsort(-np.abs(d_te))[:DE_TOPN], np.argsort(-np.abs(d_tr))[:DE_TOPN])
        pred_diff = pd_e.mean(0) - pd_r.mean(0)
        true_diff = d_te - d_tr
        r_diff = float(np.corrcoef(pred_diff[top], true_diff[top])[0, 1])
        rows4.append(dict(gene=g, pearson_k562=r_e, pearson_rpe1=r_r, diff_r=r_diff,
                          tier1=g in TIER1_40, tier2=g in TIER2))
    df4 = pd.DataFrame(rows4)
    df4.to_csv(f"{OUT}/eval_v4_tier4_crossline.csv", index=False)
    df4_ho = df4[df4.tier1 | df4.tier2]  # 仅训练留出基因: 严格的泛化版头条
    summary["tier4_crossline"] = dict(
        n=len(df4), pearson_k562=float(df4.pearson_k562.mean()),
        pearson_rpe1=float(df4.pearson_rpe1.mean()),
        diff_r_mean=float(df4.diff_r.mean()),
        diff_r_positive_frac=float((df4.diff_r > 0).mean()),
        n_holdout=len(df4_ho),
        diff_r_holdout=float(df4_ho.diff_r.mean()) if len(df4_ho) else None,
        diff_r_holdout_positive_frac=float((df4_ho.diff_r > 0).mean()) if len(df4_ho) else None)
    print(f"== Tier4 汇总: K562={df4.pearson_k562.mean():.3f} RPE1={df4.pearson_rpe1.mean():.3f} "
          f"系间差异相关={df4.diff_r.mean():.3f} | 留出基因子集 n={len(df4_ho)} "
          f"diff_r={df4_ho.diff_r.mean():.3f} ==", flush=True)
else:
    print("!! 缺 K562-essential 或 RPE1 h5ad, Tier4 跳过", flush=True)
    summary["tier4_crossline"] = "SKIPPED: h5ad missing"

json.dump(summary, open(f"{OUT}/eval_v4_summary.json", "w"), indent=1)
print("\n===== v4 评估汇总 =====")
print(json.dumps(summary, indent=1))
print("EVAL_V4_FULL_DONE", flush=True)


