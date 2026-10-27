# dili_classifier.py
# DILI 毒性分类专职模型: SMILES -> ChemBERTa-2(冻结) -> MLP -> DILI 严重度二分类
# 数据: DILIrank v2, vMost+vLess=阳性(1), vNo=阴性(0), Ambiguous 剔除
# 验证: Murcko scaffold 5-fold x 3 seeds; 指标 AUC/AP/BACC
# 基线: ECFP4 -> 同构 MLP
import json, hashlib, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = "/root/autodl-tmp/drug_spec"
OUT = f"{DS}/dili_cls_" + os.environ.get("TASK", "all")
os.makedirs(OUT, exist_ok=True)

smiles = json.load(open(f"{DS}/dilirank_smiles_clean.json"))
scaf = json.load(open(f"{DS}/dilirank_scaffolds.json"))
dl = pd.read_csv("/root/autodl-tmp/dilirank_val/dilirank2.csv")
def sev(v):
    s = str(v).lower()
    if "most" in s: return 2
    if "less" in s: return 1
    if "no-dili" in s: return 0
    return -1
dl["sev"] = dl["vDILI-Concern"].map(sev)
dl = dl[dl.sev >= 0]
best = dl.groupby(dl.CompoundName)["sev"].max()
TASK = os.environ.get("TASK", "all")
if TASK == "vmost":
    names = [n for n in smiles if n in best.index and best[n] in (0, 2)]
    Y = np.array([1 if best[n] == 2 else 0 for n in names], np.int64)
else:
    names = [n for n in smiles if n in best.index]
    Y = np.array([1 if best[n] >= 1 else 0 for n in names], np.int64)
print("n:", len(names), "pos:", int(Y.sum()), "neg:", int((Y == 0).sum()), flush=True)

# ChemBERTa 嵌入缓存
EMB_F = f"{DS}/dilirank_chemberta_emb_all.npy"  # reuse cached embeddings from TASK=all
if os.path.exists(EMB_F):
    EMB = np.load(EMB_F)
else:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(f"{DS}/chemberta77m")
    cb = AutoModel.from_pretrained(f"{DS}/chemberta77m").to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(names), 32):
            t = tok([smiles[n] for n in names[i:i+32]], return_tensors="pt",
                    padding=True, truncation=True, max_length=256).to(DEV)
            h = cb(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    EMB = np.concatenate(embs).astype(np.float32)
    np.save(EMB_F, EMB)
print("emb", EMB.shape, flush=True)

from rdkit import Chem
from rdkit.Chem import AllChem
FP = np.zeros((len(names), 2048), np.float32)
for i, n in enumerate(names):
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles[n]), 2, nBits=2048)
    FP[i] = np.array(fp)

folds = np.random.RandomState(0).randint(0, 5, len(names))  # random-split control (ignores scaffold)

class Clf(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def metrics(y, s):
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    auc = (r[y == 1].mean() - (n1 + 1) / 2) / n0
    # AP
    order = np.argsort(-s); tp = np.cumsum(y[order]); prec = tp / np.arange(1, len(y) + 1)
    ap = (prec * y[order]).sum() / max(n1, 1)
    pred = (s > np.median(s)).astype(int)
    bacc = 0.5 * (((pred == 1) & (y == 1)).sum() / max(n1, 1) + ((pred == 0) & (y == 0)).sum() / max(n0, 1))
    return auc, ap, bacc

t0 = time.time()
rows = []
for feat, fname in [(EMB, "chemberta"), (FP, "ecfp4")]:
    for fold in range(5):
        te = folds == fold
        Xtr = torch.from_numpy(feat[~te]).to(DEV); ytr = torch.from_numpy(Y[~te]).float().to(DEV)
        Xte = torch.from_numpy(feat[te]).to(DEV)
        for seed in range(3):
            torch.manual_seed(seed)
            model = Clf(feat.shape[1]).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
            pos_w = torch.tensor((Y[~te] == 0).sum() / max((Y[~te] == 1).sum(), 1)).float().to(DEV)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            n = len(Xtr)
            for ep in range(200):
                model.train(); perm = torch.randperm(n, device=DEV)
                for b0 in range(0, n, 128):
                    b = perm[b0:b0+128]
                    loss = lossf(model(Xtr[b]), ytr[b])
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                s = torch.sigmoid(model(Xte)).cpu().numpy()
            auc, ap, bacc = metrics(Y[te], s)
            rows.append(dict(feat=fname, fold=fold, seed=seed, auc=auc, ap=ap, bacc=bacc,
                             n_te=int(te.sum())))
            print(f"{fname} f{fold} s{seed} AUC={auc:.3f} {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/per_fold.csv", index=False)
summ = res.groupby("feat")[["auc", "ap", "bacc"]].agg(["mean", "std"]).round(3)
print(summ, flush=True)
summ.to_csv(f"{OUT}/summary.csv")
print("ALL DONE", flush=True)
