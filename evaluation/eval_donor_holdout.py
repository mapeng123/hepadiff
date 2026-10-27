#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_sf9_donor.py — SF9 donor-level hold-out 评估
对照: donorho@60k (106 donor 整供体未见) vs 主模型@60k (同步数)
任务: 留出 donor 细胞的 50% 掩码重建, per-cell Pearson -> per-donor 中位
输出: /root/autodl-tmp/panels_v5_fig4/sf9_donorho_recon.csv + sf9_summary.json
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, time, os

BONE_MM = "/root/autodl-tmp/virtual-liver/processed/v4_mm_r1"
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
CK_MAIN = "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5/pmdm_v4_step60000.pt"
CK_HO = "/root/autodl-tmp/virtual-liver/checkpoints_pmdm_v5_donorho/pmdm_v4_step60000.pt"
OUT = "/root/autodl-tmp/panels_v5_fig4"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
D_MODEL, N_HEAD, N_LAYER, CHUNK, MASK_TOKEN, GID_DIM = 512, 8, 12, 48, 51, 256
MIN_NNZ, PER_DONOR, BATCH = 50, 300, 250
t0 = time.time()

genes = [l.strip() for l in open(GENES_F)]
G = len(genes)
N_CHUNK = (G + CHUNK - 1) // CHUNK; G_PAD = N_CHUNK * CHUNK
edges = np.load(EDGES_F)
LUT = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

class PMDDM(nn.Module):
    def __init__(self, n_ct, n_cond, n_pert, n_sp):
        super().__init__()
        self.gene_emb = nn.Embedding(52, D_MODEL)
        self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, N_CHUNK + 5, D_MODEL) * 0.02)
        self.ct_emb = nn.Embedding(n_ct, D_MODEL)
        self.cond_emb = nn.Embedding(n_cond, D_MODEL)
        self.sp_emb = nn.Embedding(n_sp, D_MODEL)
        self.ptype_emb = nn.Embedding(4, GID_DIM)
        self.gene_id_emb = nn.Embedding(G, GID_DIM)
        self.pert_gene_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.pert_free_emb = nn.Embedding(n_pert, GID_DIM)
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
        head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0, self.sp_emb(sp)], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 5:]
        return self.decompress(z).view(B, G_PAD, 52)

def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PMDDM(len(ck["ct_vocab"]) + 1, len(ck["cond_vocab"]) + 1,
              len(ck["pert_vocab"]) + 1, len(ck["species_vocab"]) + 1).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    voc = dict(ct=ck["ct_vocab"], cond=ck["cond_vocab"], sp=ck["species_vocab"])
    del ck
    return m, voc

print("load main@60k ...", flush=True)
m_main, voc = load_model(CK_MAIN)
print("load donorho@60k ...", flush=True)
m_ho, _ = load_model(CK_HO)
ct_vocab, cond_vocab, SPV = voc["ct"], voc["cond"], voc["sp"]

# ---------- meta 与留出 donor 细胞 ----------
print("load meta ...", flush=True)
meta = pd.read_csv(f"{BONE_MM}/meta_v6_r1.csv", index_col=0, low_memory=False)
_v4 = pd.read_csv(f"{BONE_MM}/condition_v4.csv")
meta["condition"] = _v4["condition_v4"].astype(str).to_numpy()
EXCL = _v4["exclude"].astype(bool).to_numpy() if "exclude" in _v4.columns else np.zeros(len(meta), bool)
mm_indptr = np.load(f"{BONE_MM}/indptr.npy", mmap_mode="r")
mm_index = np.load(f"{BONE_MM}/indices.npy", mmap_mode="r")
mm_data = np.load(f"{BONE_MM}/data.npy", mmap_mode="r")
nnz = np.diff(mm_indptr)
_sf9 = json.load(open(f"{BONE_MM}/sf9_holdout_donors.json"))
ho_pairs = set(tuple(x.split("||", 1)) for x in _sf9["donors"])
sp_arr = meta["species"].astype(str).str.lower().to_numpy()
sm_arr = meta["sample"].astype(str).to_numpy()
ho_mask = np.array([(s, d) in ho_pairs for s, d in zip(sp_arr, sm_arr)])
elig = np.where((nnz >= MIN_NNZ) & ~EXCL & ho_mask)[0]
print(f"held-out donor cells: {len(elig)}", flush=True)

ct_arr = meta["celltype_v5"].astype(str).to_numpy()
cd_arr = meta["condition"].to_numpy()

def vid(vocab, name):
    return vocab.get(name, len(vocab))

rng = np.random.default_rng(0)
donor_keys = np.array([f"{s}||{d}" for s, d in zip(sp_arr, sm_arr)])
sel_rows = []
for dk in sorted(_sf9["donors"]):
    idx = elig[donor_keys[elig] == dk]
    if len(idx) == 0:
        continue
    take = rng.choice(idx, min(PER_DONOR, len(idx)), replace=False)
    sel_rows.append((dk, take))
print(f"donors with cells: {len(sel_rows)}", flush=True)

# ---------- 重建评估 ----------
mrng = np.random.default_rng(97)
rows = []
for dk, take in sel_rows:
    sp_name, sample = dk.split("||", 1)
    r_ho, r_main = [], []
    for i in range(0, len(take), BATCH):
        bidx = take[i:i + BATCH]
        b = len(bidx)
        x = np.zeros((b, G), np.int8)
        for j, r in enumerate(bidx):
            s, e = int(mm_indptr[r]), int(mm_indptr[r + 1])
            if e > s:
                x[j, mm_index[s:e].astype(np.int64)] = mm_data[s:e]
        xt = torch.from_numpy(np.pad(x, ((0, 0), (0, G_PAD - G)))).to(DEV)
        mask = torch.from_numpy(mrng.random(xt.shape) < 0.5).to(DEV)
        xm = xt.clone(); xm[mask] = MASK_TOKEN
        ct = torch.tensor([vid(ct_vocab, c) for c in ct_arr[bidx]], device=DEV)
        cn = torch.tensor([vid(cond_vocab, c) for c in cd_arr[bidx]], device=DEV)
        sp = torch.tensor([vid(SPV, s) for s in sp_arr[bidx]], device=DEV)
        pt = torch.full((b,), 0, dtype=torch.long, device=DEV)   # NULL pert = len(pert_vocab)? 用0无妨——pv不用
        z0 = torch.zeros(b, D_MODEL, device=DEV)
        pg = torch.full((b,), -1, dtype=torch.long, device=DEV)
        pty = torch.zeros(b, dtype=torch.long, device=DEV)
        with torch.no_grad():
            pr_main = m_main(xm, ct, cn, pt, z0, pg, pty, sp).argmax(-1)[:, :G].cpu().numpy()
            pr_ho = m_ho(xm, ct, cn, pt, z0, pg, pty, sp).argmax(-1)[:, :G].cpu().numpy()
        m_np = mask.cpu().numpy()[:, :G]
        true_e = LUT[x.astype(np.int16)] * 1e4; np.log1p(true_e, out=true_e)
        for j in range(b):
            sel = m_np[j] & (x[j] > 0)
            if sel.sum() < 20:
                continue
            pe_m = LUT[pr_main[j].astype(np.int16)] * 1e4; np.log1p(pe_m, out=pe_m)
            pe_h = LUT[pr_ho[j].astype(np.int16)] * 1e4; np.log1p(pe_h, out=pe_h)
            t_ = true_e[j][sel]
            r_main.append(float(np.corrcoef(pe_m[sel], t_)[0, 1]))
            r_ho.append(float(np.corrcoef(pe_h[sel], t_)[0, 1]))
    if r_main:
        rows.append(dict(donor=dk, species=sp_name, n_cells=len(r_main),
                         r_main60k=float(np.median(r_main)), r_donorho60k=float(np.median(r_ho))))
    print(f"{dk} n={len(r_main)} | {(time.time()-t0)/60:.1f}min", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/sf9_donorho_recon.csv", index=False)
from scipy.stats import wilcoxon
w = wilcoxon(df.r_main60k, df.r_donorho60k)
summ = dict(n_donors=len(df), n_cells=int(df.n_cells.sum()),
            main_median=round(float(df.r_main60k.median()), 4),
            donorho_median=round(float(df.r_donorho60k.median()), 4),
            wilcoxon_p=float(w.pvalue))
json.dump(summ, open(f"{OUT}/sf9_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
print("SF9_DONOR_EVAL_DONE", flush=True)
