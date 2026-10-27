#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lasso_ko_basis_v5_dump.py — 同 lasso_ko_basis_v5.py, 追加逐药 CSV 输出 (SF23 用)
输出: /root/autodl-tmp/drug_spec/lasso_ko_basis_v5_perdrug.csv
列: compound, r2_real, r2_null, n_selected
"""
import numpy as np, json
import pandas as pd
from sklearn.linear_model import LassoLarsIC

ko = np.load("/root/autodl-tmp/dilirank_val/screen_v5_deltas.npz")
kn = ko.files
lm = json.load(open("/root/autodl-tmp/drug_spec/lincs978_genes_final.json"))
panel = [l.strip() for l in open("/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt")]
pidx = {g: i for i, g in enumerate(panel)}
shared = [g for g in lm if g in pidx]
B = np.stack([ko[k] for k in kn])[:, [pidx[g] for g in shared]]
B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
tg = np.load("/root/autodl-tmp/dili_spec/tggates_logfc.npz", allow_pickle=True)
comps = [str(c).lower() for c in tg["compounds"]]
logfc = tg["logfc"][:, -1, :][:, [pidx[g] for g in shared]]
logfc = logfc / (np.linalg.norm(logfc, axis=1, keepdims=True) + 1e-9)

rng = np.random.default_rng(0)
r2_real, r2_null, nsel = [], [], []
for i in range(len(comps)):
    y = logfc[i]
    m = LassoLarsIC(criterion="bic").fit(B.T, y)
    pred = B.T @ m.coef_
    r2_real.append(1 - ((y - pred) ** 2).sum() / ((y ** 2).sum() + 1e-9))
    nsel.append(int((m.coef_ != 0).sum()))
    Rn = rng.standard_normal(B.shape)
    Rn /= np.linalg.norm(Rn, axis=1, keepdims=True)
    mn = LassoLarsIC(criterion="bic").fit(Rn.T, y)
    pn = Rn.T @ mn.coef_
    r2_null.append(1 - ((y - pn) ** 2).sum() / ((y ** 2).sum() + 1e-9))
r2_real, r2_null = np.array(r2_real), np.array(r2_null)
df = pd.DataFrame(dict(compound=comps, r2_real=r2_real, r2_null=r2_null, n_selected=nsel))
df.to_csv("/root/autodl-tmp/drug_spec/lasso_ko_basis_v5_perdrug.csv", index=False)
out = dict(real_med=round(float(np.median(r2_real)), 3), real_mean=round(float(r2_real.mean()), 3),
           null_med=round(float(np.median(r2_null)), 3), null_mean=round(float(r2_null.mean()), 3),
           frac_better=round(float((r2_real > r2_null).mean()), 3),
           n_selected_med=float(np.median(nsel)), n_drugs=len(comps))
json.dump(out, open("/root/autodl-tmp/drug_spec/lasso_ko_basis_v5_dump.json", "w"), indent=1)
print(json.dumps(out), flush=True)
print("LASSO_DUMP_DONE", flush=True)
