#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""program_screen.py — 程序坐标系药筛 + DILI 损伤轴终验 (无扰动集成) v2
v2 修复: LINCS h5ad var_names 为纯序号 -> 用 GSE92742 gene_info landmark 顺序映射基因名
        新增映射阳性对照 (硼替佐米->蛋白酶体亚基, 格尔德霉素->HSP)
A. 疾病程序轴: fig4 10种子生成谱 -> 肝细胞 疾病-健康 delta (16000 panel) [已有则加载]
B. 药筛 Tier A: LINCS 实测 profile (10694 化合物) 投影程序轴 -> 逆转排名; 已知药验证
C. DILI: 专职模型预测 delta (留出药) -> 损伤轴投影 -> DILIrank AUC
输出: /root/autodl-tmp/drug_spec/program_screen/{summary.json, screen_ranking.csv, disease_programs.npz}
"""
import numpy as np, pandas as pd, json, gzip, csv, os
from scipy.stats import rankdata

DS = "/root/autodl-tmp/drug_spec"
P4 = "/root/autodl-tmp/panels_v4"
DLV = "/root/autodl-tmp/dilirank_val"
OUT = f"{DS}/program_screen"
os.makedirs(OUT, exist_ok=True)

GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
EDGES_F = "/root/autodl-tmp/virtual-liver/processed/liver_bin_edges_v4.npy"
genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}
edges = np.load(EDGES_F)
LUT = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

def mean_log(tok):
    X = LUT[tok.astype(np.int16)] * 1e4
    np.log1p(X, out=X)
    return X.mean(0)

# ---------- A. 疾病程序轴 (已有则加载) ----------
CONDS = ["MASLD", "MASH", "fibrosis_mixed_etiology", "cirrhosis_mixed_etiology", "HCC"]
if os.path.exists(f"{OUT}/disease_programs.npz"):
    zz = np.load(f"{OUT}/disease_programs.npz")
    if all(c in zz.files for c in CONDS):
        programs = {c: zz[c] for c in CONDS}
        print("programs loaded from cache:", list(programs), flush=True)
    else:
        programs = None
else:
    programs = None

if programs is None:
    prog_seeds = {c: [] for c in CONDS}
    hep_cts = {"hepatocyte", "periportal region hepatocyte", "midzonal region hepatocyte",
               "centrilobular region hepatocyte"}
    for s in range(10):
        f = f"{P4}/fig4_spectrum_s{s}.npz"
        if not os.path.exists(f):
            print("missing", f, flush=True); continue
        z = np.load(f, allow_pickle=True)
        tok, ct, cond = z["tok"], z["ct"].astype(str), z["cond"].astype(str)
        hm = np.isin(ct, list(hep_cts))
        if (hm & (cond == "healthy")).sum() < 50:
            continue
        base = mean_log(tok[hm & (cond == "healthy")])
        for c in CONDS:
            m = hm & (cond == c)
            if m.sum() >= 50:
                prog_seeds[c].append(mean_log(tok[m]) - base)
        print(f"seed{s} done", flush=True)
    programs = {}
    for c in CONDS:
        if len(prog_seeds[c]) >= 5:
            programs[c] = np.mean(prog_seeds[c], 0)
    np.savez(f"{OUT}/disease_programs.npz", **programs)
print("programs:", {k: int((v != 0).sum()) for k, v in programs.items()}, flush=True)

# ---------- B. 药筛 Tier A: LINCS 实测谱 -> 逆转打分 ----------
# v2: landmark 基因名映射 (epsilon全集相关反解金标准, mapping_report.json 验证)
lm_genes = json.load(open(f"{DS}/lincs978_genes_final.json"))
assert len(lm_genes) == 978, f"landmark != 978: {len(lm_genes)}"
lm_idx_panel = np.array([gidx[g] for g in lm_genes if g in gidx])
lm_idx_lincs = np.array([i for i, g in enumerate(lm_genes) if g in gidx])
print("landmark overlap with 16000 panel:", len(lm_idx_panel), flush=True)

z = np.load(f"{DS}/lincs_profiles.npz", allow_pickle=True)
PROF, PERTS = z["prof"], z["perts"]            # (n, 978) delta vs vehicle, BRD pert_id
TOUCH = z["touch"] if "touch" in z.files else np.zeros(len(PERTS), bool)
P_sub = PROF[:, lm_idx_lincs]

# pert_id -> 化合物名 (h5ad obs 提取)
P2N = json.load(open(f"{DS}/pertid2name.json"))
PNAMES = np.array([P2N.get(str(p), str(p)).lower() for p in PERTS])

# ---- 阳性对照: 硼替佐米应上调蛋白毒性应激基因 (HSR/NRF2) ----
def top_genes(pert_name, n=30):
    idx = [i for i, nm in enumerate(PNAMES) if pert_name == nm]
    if not idx:
        return None
    prof = PROF[idx].mean(0)
    top = np.argsort(prof)[::-1][:n]
    return [lm_genes[i] for i in top]

sanity = {}
bort = top_genes("bortezomib")
if bort:
    stress = [g for g in bort if g in ("HSPA1A","DNAJB1","BAG3","HSPB1","HMOX1","SQSTM1","TXNRD1","DDIT4","HERPUD1")]
    sanity["bortezomib_top30_stress"] = dict(n=len(stress), genes=stress[:10])
geld = top_genes("geldanamycin")
if geld:
    hsp = [g for g in geld if g.startswith(("HSP", "DNAJ")) or g in ("HMOX1","HERPUD1")]
    sanity["geldanamycin_top30_hsr"] = dict(n=len(hsp), genes=hsp[:10])
print("MAPPING SANITY:", json.dumps(sanity, ensure_ascii=False), flush=True)

rank_rows = []
for cname, prog in programs.items():
    p = prog[lm_idx_panel]
    pn = p / (np.linalg.norm(p) + 1e-9)
    sc = -(P_sub @ pn) / (np.linalg.norm(P_sub, axis=1) + 1e-9)   # 逆转分
    rank_rows.append(pd.DataFrame(dict(program=cname, pert=PERTS.astype(str),
                                       pert_name=PNAMES,
                                       touchstone=TOUCH, reversal=sc)))
rank_df = pd.concat(rank_rows)
rank_df.to_csv(f"{OUT}/screen_ranking.csv", index=False)

# 已知 MASH/抗纤维化/肝病药验证
KNOWN = {
    "pioglitazone": "MASH获批(适应证外)/PPARg", "rosiglitazone": "PPARg",
    "obeticholic acid": "FXR激动剂(PBC获批,MASH III期)", "saroglitazar": "印度获批MASH",
    "elafibranor": "PPARa/d III期", "cenicriviroc": "抗纤维化III期",
    "selonsertib": "ASK1抑制剂III期", "vitamin e": "NASH指南", "alpha-tocopherol": "NASH指南",
    "metformin": "糖尿病/肝病常用", "ursodiol": "PBC获批", "ursodeoxycholic acid": "PBC获批",
    "troglitazone": "DILI退市(阴性对照)", "resmetirom": "MASH获批2024",
}
verif = {}
pert_lower = {nm: i for i, nm in enumerate(PNAMES)}
for drug, note in KNOWN.items():
    hit = None
    for pl, i in pert_lower.items():
        if drug in pl:
            hit = (pl, i); break
    if hit is None:
        verif[drug] = dict(in_lincs=False, note=note); continue
    pl, i = hit
    row = {"in_lincs": True, "name": pl, "note": note}
    for cname in programs:
        sub = rank_df[rank_df.program == cname].reset_index(drop=True)
        sc = sub.reversal.to_numpy()
        pct = 1 - rankdata(sc)[i] / len(sc)
        row[f"{cname}_pct"] = round(float(pct), 3)
    verif[drug] = row
print(json.dumps(verif, indent=1, ensure_ascii=False), flush=True)

# ---------- C. DILI 损伤轴 AUC (专职模型预测 delta) ----------
inj = pd.read_csv(f"{DLV}/injury_signature.csv")
inj_w = np.zeros(G, np.float32)
for _, r in inj.iterrows():
    if r["gene"] in gidx: inj_w[gidx[r["gene"]]] = r["weight"]
lab = json.load(open(f"{DS}/tggates_dilirank_labels.json"))
pdlt = np.load(f"{DS}/dili_spec_results/pred_deltas.npz")
names = [n for n in pdlt.files if lab.get(n) in (0, 1)]
y = np.array([lab[n] for n in names])
sc = np.array([float(pdlt[n] @ inj_w) for n in names])
r = rankdata(sc); n1 = (y == 1).sum(); n0 = (y == 0).sum()
auc = (r[y == 1].mean() - (n1 + 1) / 2) / n0 if n0 > 0 else float("nan")
dili_axis = dict(auc=round(float(auc), 3), n_pos=int(n1), n_neg=int(n0), note="n_neg小,仅供参照")
print("DILI axis AUC:", dili_axis, flush=True)
json.dump(dili_axis, open(f"{OUT}/dili_axis_auc.json", "w"), indent=1)

# 汇总
summ = dict(programs={k: dict(n_nonzero=int((v != 0).sum())) for k, v in programs.items()},
            landmark_overlap=int(len(lm_idx_panel)), screen_n=int(len(PERTS)),
            mapping_sanity=sanity,
            known_drug_verification=verif, dili_axis=dili_axis)
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False, default=str)
print("PROGRAM_SCREEN DONE", flush=True)
