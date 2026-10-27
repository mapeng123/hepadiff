# dili_spec_train.py
# DILI 专职模型 v1: SMILES -> ChemBERTa-2(冻结) -> adapter MLP -> 16000-d 表达 delta
# 监督: TG-GATEs 119 药 x 3 剂量 logFC (人源体外, 与主模型 16000 基因面板对齐)
# 验证: Murcko scaffold 5-fold 留出 (防立体异构/近缘骨架泄露)
# 基线: ECFP4 指纹 -> 同构 MLP
# 指标: 留出药 pearson / top200 方向一致率 / 损伤轴 AUC (vMost+vLess vs vNo)
import json, time, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = "/root/autodl-tmp/drug_spec"
DILI = "/root/autodl-tmp/dili_spec"
DLV = "/root/autodl-tmp/dilirank_val"
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
OUT = f"{DS}/dili_spec_results"
import os; os.makedirs(OUT, exist_ok=True)

genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}

# ---- 数据 ----
z = np.load(f"{DILI}/tggates_logfc.npz")
LOGFC = z["logfc"].astype(np.float32)          # (119, 3, 16000)
COMPS = list(z["compounds"])
meta = json.load(open(f"{DILI}/tggates_logfc_meta.json"))
DOSES = meta["dose_levels"]                     # e.g. ["low","middle","high"]
smiles = json.load(open(f"{DS}/tggates_smiles.json"))
scaf = json.load(open(f"{DS}/tggates_scaffolds.json"))

# DILIrank 标签
dl = pd.read_csv(f"{DLV}/dilirank2.csv")
dl["key"] = dl["CompoundName"].str.strip().str.lower()
lab = {}
for _, r in dl.iterrows():
    v = str(r["vDILI-Concern"])
    lab[r["key"]] = 1 if "MOST" in v or "LESS" in v else (0 if "No-DILI" in v else -1)

# 损伤轴
inj = pd.read_csv(f"{DLV}/injury_signature.csv")
inj_w = np.zeros(G, np.float32)
for _, r in inj.iterrows():
    if r["gene"] in gidx:
        inj_w[gidx[r["gene"]]] = r["weight"]
inj_w = torch.from_numpy(inj_w).to(DEV)

# ---- ChemBERTa 嵌入 (冻结, 缓存) ----
EMB_F = f"{DS}/tggates_chemberta_emb.npy"
import os.path
if os.path.exists(EMB_F):
    EMB = np.load(EMB_F)
else:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(f"{DS}/chemberta77m")
    cb = AutoModel.from_pretrained(f"{DS}/chemberta77m").to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(COMPS), 16):
            batch = [smiles[c] for c in COMPS[i:i+16]]
            t = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
            h = cb(**t).last_hidden_state            # (B, L, 768)
            m = t["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    EMB = np.concatenate(embs).astype(np.float32)     # (119, 768)
    np.save(EMB_F, EMB)
print("emb", EMB.shape, flush=True)

# ECFP4 基线指纹
from rdkit import Chem
from rdkit.Chem import AllChem
FP = np.zeros((len(COMPS), 2048), np.float32)
for i, c in enumerate(COMPS):
    mol = Chem.MolFromSmiles(smiles[c])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    FP[i] = np.array(fp)

# ---- scaffold 5-fold ----
uniq = sorted(set(scaf.values()))
fold_of_scaf = {}
for s in uniq:
    h = int(hashlib.md5(s.encode()).hexdigest(), 16)
    fold_of_scaf[s] = h % 5
drug_fold = np.array([fold_of_scaf[scaf[c]] for c in COMPS])
print("fold sizes:", [int((drug_fold == f).sum()) for f in range(5)], flush=True)

# dose 数值化
dnum = {d: i for i, d in enumerate(DOSES)}
ND = len(DOSES)

# ---- 样本 ----
# X: [emb(768) | dose_onehot(ND)] ; y: logfc(16000)
def build_xy(feat):
    X, Y, DIDX = [], [], []
    for ci in range(len(COMPS)):
        for di in range(ND):
            dh = np.zeros(ND, np.float32); dh[di] = 1.0
            X.append(np.concatenate([feat[ci], dh]))
            Y.append(LOGFC[ci, di])
            DIDX.append(ci)
    return np.stack(X), np.stack(Y), np.array(DIDX)

class Adapter(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.15), nn.Linear(1024, G))
    def forward(self, x): return self.net(x)

def run_fold(feat, feat_name, fold, seed):
    X, Y, DIDX = build_xy(feat)
    te_d = np.where(drug_fold == fold)[0]
    trm = ~np.isin(DIDX, te_d)
    Xt = torch.from_numpy(X[trm]).to(DEV); Yt = torch.from_numpy(Y[trm]).to(DEV)
    Xv = torch.from_numpy(X[~trm]).to(DEV); Yv = torch.from_numpy(Y[~trm]).to(DEV)
    torch.manual_seed(seed)
    model = Adapter(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    n = Xt.shape[0]
    for ep in range(300):
        model.train(); perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for b0 in range(0, n, 64):
            b = perm[b0:b0+64]
            pred = model(Xt[b])
            mse = ((pred - Yt[b]) ** 2).mean()
            cos = 1 - torch.nn.functional.cosine_similarity(pred, Yt[b], dim=-1).mean()
            loss = mse + 0.3 * cos
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
    # eval: 留出药, high dose
    model.eval()
    rows = []
    with torch.no_grad():
        for ci in np.unique(DIDX[~trm]):
            di = ND - 1  # high dose
            dh = np.zeros(ND, np.float32); dh[di] = 1.0
            x = torch.from_numpy(np.concatenate([feat[ci], dh])[None]).to(DEV)
            pred = model(x)[0]
            true = torch.from_numpy(LOGFC[ci, di]).to(DEV)
            pear = torch.corrcoef(torch.stack([pred, true]))[0, 1].item()
            k = 200
            top = torch.topk(true.abs(), k).indices
            dir_c = (torch.sign(pred[top]) == torch.sign(true[top])).float().mean().item()
            inj_s = (pred * inj_w).sum().item()
            cname = COMPS[ci]
            rows.append(dict(fold=fold, seed=seed, compound=cname,
                             pearson=pear, dir200=dir_c, injury=inj_s,
                             dilirank=lab.get(cname.lower(), -1)))
            if feat_name == "chemberta" and seed == 0:
                PRED_DELTA[cname] = pred.cpu().numpy().astype(np.float32)
    return rows

t0 = time.time()
all_rows = []
PRED_DELTA = {}
for feat, name in [(EMB, "chemberta"), (FP, "ecfp4")]:
    for fold in range(5):
        for seed in range(3):
            all_rows += run_fold(feat, name, fold, seed)
            print(f"{name} fold{fold} seed{seed} done {time.time()-t0:.0f}s", flush=True)
            # 给每行加 feat 标签
            for r in all_rows[-1 * len([1]):]:
                pass
    # 标注
    n_new = 5 * 0  # placeholder
# 重新打标签: run_fold 不带 feat, 这里按顺序补
np.savez(f"{OUT}/pred_deltas.npz", **{k: v for k, v in PRED_DELTA.items()})
res = pd.DataFrame(all_rows)
res["feat"] = np.repeat(["chemberta", "ecfp4"], [len(all_rows)//2, len(all_rows) - len(all_rows)//2])
res.to_csv(f"{OUT}/per_drug.csv", index=False)

# 汇总
def auc(sc, lb):
    pos = sc[lb == 1]; neg = sc[lb == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].mean() - (len(pos) + 1) / 2) / len(neg)

summ = {}
for name in ["chemberta", "ecfp4"]:
    d = res[res.feat == name]
    dd = d.groupby("compound").agg(pearson=("pearson", "mean"), dir200=("dir200", "mean"),
                                   injury=("injury", "mean"), dilirank=("dilirank", "first"))
    ok = dd[dd.dilirank >= 0]
    summ[name] = dict(
        pearson_mean=float(dd.pearson.mean()), pearson_med=float(dd.pearson.median()),
        dir200_mean=float(dd.dir200.mean()),
        injury_auc=float(auc(ok.injury.to_numpy(), ok.dilirank.to_numpy())),
        n_eval=int(len(dd)))
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
print("ALL DONE", flush=True)
