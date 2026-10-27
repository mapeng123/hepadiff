#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成对照实验: 分离解码步数与表达偏置对生成真实感的影响.
B组: STEPS=4,  ALPHA=0, BETA=0   (只去偏置)
C组: STEPS=50, ALPHA=0, BETA=0   (去偏置+多步数)
各生成 100k, 与 panels_v5/fig2_real400k.npz 对比 (同 fig2_strat_metrics 口径).
输出: /root/autodl-tmp/panels_v5_unbiased/"""
import numpy as np, pandas as pd, torch, torch.nn as nn, time, json, os

GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
CKPT = "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5/pmdm_v4_step155000.pt"
REAL_NPZ = "/root/autodl-tmp/panels_v5/fig2_real400k.npz"
OUT = "/root/autodl-tmp/panels_v5_unbiased"
os.makedirs(OUT, exist_ok=True)

DEV = "cuda"
D_MODEL, N_HEAD, N_LAYER, CHUNK, MASK_TOKEN, GID_DIM = 512, 8, 12, 48, 51, 256
N_GEN = 100000
t0 = time.time()

genes = [l.strip() for l in open(GENES_F)]
G = len(genes)
N_CHUNK = (G + CHUNK - 1) // CHUNK; G_PAD = N_CHUNK * CHUNK
edges = np.load(EDGES_F)
LUT = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

# ---------- 真实参考 (直接读缓存) ----------
d1 = np.load(REAL_NPZ, allow_pickle=True)
real_tok, ct_r = d1["real"], d1["ct"].astype(str)
cond_r, sp_r = d1["cond"].astype(str), d1["sp"].astype(str)
print(f"real {len(real_tok)} loaded | {time.time()-t0:.0f}s", flush=True)

# ---------- 模型 ----------
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
ct_vocab, cond_vocab, pert_vocab = ck["ct_vocab"], ck["cond_vocab"], ck["pert_vocab"]
SPV = ck.get("species_vocab", {"human": 0, "mouse": 1, "rat": 2, "cellline": 3})
N_CT, N_COND, N_PERT = len(ct_vocab) + 1, len(cond_vocab) + 1, len(pert_vocab) + 1
NULL_PERT = len(pert_vocab)
N_SP = len(SPV) + 1
N_PTYPE = 4

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
    def forward(self, x, ct, cond, pert, z0, pgid, ptype, sp):
        B = x.shape[0]
        e = self.gene_emb(x.long()).view(B, N_CHUNK, CHUNK * D_MODEL)
        z = self.compress(e)
        pv = torch.zeros(B, D_MODEL, device=x.device)
        pt = self.ptype_emb(ptype.clamp(0, N_PTYPE - 1))
        gm = (pgid >= 0)
        if gm.any():
            pv[gm] = self.pert_gene_mlp(self.gene_id_emb(pgid[gm]) + pt[gm])
        head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0, self.sp_emb(sp)], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 5:]
        return self.decompress(z).view(B, G_PAD, 52)

model = PMDDM().to(DEV); model.load_state_dict(ck["model"]); model.eval(); del ck
print(f"model loaded | {time.time()-t0:.0f}s", flush=True)

# ---------- 标签平铺 (与正式版同口径) ----------
ok = np.isin(ct_r, list(ct_vocab)) & np.isin(cond_r, list(cond_vocab)) & np.isin(sp_r, list(SPV))
rng = np.random.default_rng(0)
pool_ct, pool_cond, pool_sp = ct_r[ok], cond_r[ok], sp_r[ok]
rep = np.concatenate([rng.permutation(len(pool_ct))
                      for _ in range(N_GEN // len(pool_ct) + 1)])[:N_GEN]
gen_ct, gen_cond, gen_sp = pool_ct[rep], pool_cond[rep], pool_sp[rep]

Z0_ZERO = torch.zeros(1, D_MODEL, device=DEV)
HIBINS = torch.arange(52, device=DEV) >= 45

@torch.no_grad()
def gen_iterative(steps, alpha, beta, seed0=0):
    gen_tok = np.zeros((N_GEN, G), dtype=np.int8)
    B = 250
    for i in range(0, N_GEN, B):
        torch.manual_seed(seed0 + i)
        n = min(B, N_GEN - i)
        cts = torch.tensor([ct_vocab[c] for c in gen_ct[i:i+n]], device=DEV)
        cns = torch.tensor([cond_vocab[c] for c in gen_cond[i:i+n]], device=DEV)
        sps = torch.tensor([SPV[s] for s in gen_sp[i:i+n]], device=DEV)
        x = torch.full((n, G_PAD), MASK_TOKEN, dtype=torch.long, device=DEV)
        pt = torch.full((n,), NULL_PERT, dtype=torch.long, device=DEV)
        pg = torch.full((n,), -1, dtype=torch.long, device=DEV)
        pty = torch.zeros((n,), dtype=torch.long, device=DEV)
        z0b = Z0_ZERO.expand(n, -1)
        order = torch.argsort(torch.rand(n, G_PAD, device=DEV), dim=1)
        for s in range(steps):
            logits = model(x, cts, cns, pt, z0b, pg, pty, sps)
            logits[..., MASK_TOKEN] = -1e9
            logits[..., 0] += alpha
            logits[..., HIBINS] += beta
            samp = torch.distributions.Categorical(logits=logits).sample()
            a = int(s * G_PAD / steps); b = int((s + 1) * G_PAD / steps)
            reveal = order[:, a:b]
            x.scatter_(1, reveal, samp.gather(1, reveal))
        x[x == MASK_TOKEN] = 0
        gen_tok[i:i+n] = x[:, :G].cpu().numpy().astype(np.int8)
        if (i // B) % 200 == 0:
            print(f"  gen {i}/{N_GEN} steps={steps} | {time.time()-t0:.0f}s", flush=True)
    return gen_tok

# ---------- 评估 (fig2_strat_metrics 同口径) ----------
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def log_chunk(tok):
    X = LUT[tok.astype(np.int16)] * 1e4
    np.log1p(X, out=X)
    return X

real_sub_idx = rng.choice(len(real_tok), N_GEN, replace=False)
sub50 = rng.choice(len(real_tok), 50000, replace=False)
pca = PCA(n_components=50, random_state=0).fit(log_chunk(real_tok[sub50]))
ZR = np.vstack([pca.transform(log_chunk(real_tok[real_sub_idx[i:i+20000]]))
                for i in range(0, N_GEN, 20000)])
mr = log_chunk(real_tok[real_sub_idx[:50000]]).mean(0)
dr = (real_tok[real_sub_idx[:50000]] > 0).mean(0)
km = KMeans(n_clusters=20, random_state=0, n_init=5).fit(
    ZR[rng.choice(len(ZR), 20000, replace=False)])
cr = np.bincount(km.predict(ZR), minlength=20) / len(ZR)
compact_r = float(np.sqrt(((ZR - ZR.mean(0)) ** 2).sum(1)).mean())
print(f"real side ready | {time.time()-t0:.0f}s", flush=True)

def evaluate(gen_tok, tag):
    ZG = np.vstack([pca.transform(log_chunk(gen_tok[i:i+20000]))
                    for i in range(0, len(gen_tok), 20000)])
    s_r = rng.choice(len(ZR), 25000, replace=False)
    s_g = rng.choice(len(ZG), 25000, replace=False)
    Z = np.vstack([ZR[s_r], ZG[s_g]])
    y = np.r_[np.zeros(len(s_r)), np.ones(len(s_g))]
    sc = cross_val_predict(LogisticRegression(max_iter=1000), Z, y, cv=5,
                           method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, sc))
    compact_g = float(np.sqrt(((ZG - ZG.mean(0)) ** 2).sum(1)).mean())
    cg = np.bincount(km.predict(ZG), minlength=20) / len(ZG)
    occ_g = float((cg > 0.001).mean())
    mg = log_chunk(gen_tok[:50000]).mean(0)
    dg = (gen_tok[:50000] > 0).mean(0)
    rho_m = float(spearmanr(mr, mg).statistic)
    rho_d = float(spearmanr(dr, dg).statistic)
    res = {"tag": tag, "auc": round(auc, 4), "compact_real": round(compact_r, 2),
           "compact_gen": round(compact_g, 2),
           "occupancy_real": round(float((cr > 0.001).mean()), 3),
           "occupancy_gen": round(occ_g, 3),
           "marg_spearman": round(rho_m, 4), "det_spearman": round(rho_d, 4),
           "nnz_gen_mean": round(float((gen_tok > 0).sum(1).mean()), 1)}
    print(json.dumps(res), flush=True)
    return res

results = []
for tag, steps, alpha, beta in [("B_steps4_nobias", 4, 0.0, 0.0),
                                ("C_steps50_nobias", 50, 0.0, 0.0)]:
    tg = time.time()
    gt = gen_iterative(steps, alpha, beta)
    print(f"[{tag}] 生成完成 steps={steps} a={alpha} b={beta} | "
          f"{(time.time()-tg)/60:.1f}min", flush=True)
    np.savez_compressed(f"{OUT}/gen100k_{tag}.npz", gen=gt, ct_gen=gen_ct,
                        cond_gen=gen_cond, sp_gen=gen_sp,
                        meta=json.dumps({"steps": steps, "alpha": alpha, "beta": beta}))
    results.append(evaluate(gt, tag))
    del gt; torch.cuda.empty_cache()

json.dump(results, open(f"{OUT}/unbiased_results.json", "w"), indent=1)
print("UNBIASED_DONE", flush=True)
