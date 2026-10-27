# lincs_spec_train.py
# 药筛专职模型: SMILES -> ChemBERTa-2(冻结) -> MLP -> 978 基因表达响应 (LINCS L1000)
# 数据: PerturbNet 处理版 LINCS-Drug (677k 样本, 978 landmark, 立体异构体清洗)
# 划分: 官方 holdout1 (isoform-clean) + 3 seeds
# 指标: 留出化合物 pearson / top100 方向一致率; Touchstone 子集单列
import json, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = "/root/autodl-tmp/drug_spec"
OUT = f"{DS}/lincs_spec_results"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---- 1. 聚合成 per-compound delta profile ----
PROF_F = f"{DS}/lincs_profiles.npz"
if os.path.exists(PROF_F):
    z = np.load(PROF_F, allow_pickle=True)
    PROF, PERTS, SPLIT, TOUCH = z["prof"], z["perts"], z["split"], z["touch"]
else:
    import anndata as ad
    a = ad.read_h5ad(f"{DS}/lincs_corrected_isoforms_splits.h5ad", backed="r")
    obs = a.obs[["pert_id", "pert_type", "if_remove", "is_touchstone",
                 "holdout1", "canonical_smiles", "inchi_key_prefix"]].copy()
    obs["row"] = np.arange(len(obs), dtype=np.int64)
    trt = obs[(obs.pert_type == "trt_cp") & (obs.if_remove == "save") &
              obs.canonical_smiles.notna()]
    ctl = obs[obs.pert_type == "ctl_vehicle"]
    print("trt rows", len(trt), "ctl rows", len(ctl), flush=True)

    def group_mean(rows, chunks=200000):
        rows = np.sort(rows)
        if len(rows) == 0:
            return np.zeros(978, np.float32)
        s = None; n = 0
        for b0 in range(0, len(rows), chunks):
            X = a.X[rows[b0:b0+chunks]].astype(np.float64)
            s = X.sum(0) if s is None else s + X.sum(0)
            n += X.shape[0]
        return (s / max(n, 1)).astype(np.float32)

    ctl_mean = group_mean(ctl.row.to_numpy())
    print("ctl mean done", time.time() - t0, flush=True)

    profs = {}; meta_rows = {}
    gb = trt.groupby("pert_id", observed=True)
    keys = list(gb.groups.keys())
    for i, pid in enumerate(keys):
        idx = trt.loc[list(gb.groups[pid]), "row"].to_numpy().astype(np.int64)
        m = group_mean(idx)
        profs[pid] = m - ctl_mean
        sub = trt[trt.pert_id == pid].iloc[0]
        meta_rows[pid] = (sub.canonical_smiles, sub.inchi_key_prefix,
                          sub.holdout1, sub.is_touchstone)
        if i % 500 == 499:
            print(i + 1, "perts", time.time() - t0, flush=True)
    PERTS = np.array(keys)
    PROF = np.stack([profs[p] for p in PERTS]).astype(np.float32)
    SPLIT = np.array([meta_rows[p][2] for p in PERTS])
    TOUCH = np.array([meta_rows[p][3] for p in PERTS], np.float32)
    SMILES = [meta_rows[p][0] for p in PERTS]
    np.savez(PROF_F, prof=PROF, perts=PERTS, split=SPLIT, touch=TOUCH,
             smiles=np.array(SMILES))
    a.file.close()
print("prof", PROF.shape, flush=True)

z = np.load(PROF_F, allow_pickle=True)
SMILES = list(z["smiles"])
NP_ = len(PERTS)

# ---- 2. ChemBERTa 嵌入 ----
EMB_F = f"{DS}/lincs_chemberta_emb.npy"
if os.path.exists(EMB_F):
    EMB = np.load(EMB_F)
else:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(f"{DS}/chemberta77m")
    cb = AutoModel.from_pretrained(f"{DS}/chemberta77m").to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, NP_, 64):
            t = tok(SMILES[i:i+64], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(DEV)
            h = cb(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            if i % 6400 == 0 and i:
                print("emb", i, time.time() - t0, flush=True)
    EMB = np.concatenate(embs).astype(np.float32)
    np.save(EMB_F, EMB)
print("emb", EMB.shape, flush=True)

# ---- 3. 训练 ----
G = PROF.shape[1]
class Net(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.15), nn.Linear(1024, G))
    def forward(self, x): return self.net(x)

tr = np.where(SPLIT == "train")[0]
te = np.where(SPLIT == "test")[0]
print("train", len(tr), "test", len(te), flush=True)

Xtr = torch.from_numpy(EMB[tr]).to(DEV); Ytr = torch.from_numpy(PROF[tr]).to(DEV)
rows = []
for seed in range(3):
    torch.manual_seed(seed)
    model = Net(EMB.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    n = len(Xtr)
    for ep in range(60):
        model.train(); perm = torch.randperm(n, device=DEV)
        for b0 in range(0, n, 256):
            b = perm[b0:b0+256]
            pred = model(Xtr[b])
            mse = ((pred - Ytr[b]) ** 2).mean()
            cos = 1 - torch.nn.functional.cosine_similarity(pred, Ytr[b], dim=-1).mean()
            (mse + 0.3 * cos).backward()
            opt.step(); opt.zero_grad()
        sched.step()
        if ep % 20 == 19:
            print(f"seed{seed} ep{ep+1} loss {mse.item():.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(te), 512):
            b = te[b0:b0+512]
            pred = model(torch.from_numpy(EMB[b]).to(DEV)).cpu().numpy()
            true = PROF[b]
            for j in range(len(b)):
                p, t = pred[j], true[j]
                pear = np.corrcoef(p, t)[0, 1]
                k = 100
                top = np.argsort(-np.abs(t))[:k]
                dirc = (np.sign(p[top]) == np.sign(t[top])).mean()
                rows.append(dict(seed=seed, pert=PERTS[b[j]], pearson=pear, dir100=dirc,
                                 touchstone=float(TOUCH[b[j]])))
    print(f"seed{seed} eval done {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/per_compound.csv", index=False)
summ = {
    "all": dict(pearson=float(res.pearson.mean()), pearson_med=float(res.pearson.median()),
                dir100=float(res.dir100.mean()), n=int(len(res))),
    "touchstone": dict(pearson=float(res[res.touchstone == 1].pearson.mean()),
                       dir100=float(res[res.touchstone == 1].dir100.mean()),
                       n=int((res.touchstone == 1).sum())),
}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
print("ALL DONE", flush=True)
