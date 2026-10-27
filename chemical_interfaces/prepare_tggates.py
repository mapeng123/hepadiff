#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DILI 专职模型数据准备: TG-GATEs 人源体外 → 化合物×剂量 logFC 矩阵
输出: /root/autodl-tmp/dili_spec/tggates_logfc.npz (compounds, dose_levels, genes)
"""
import numpy as np, pandas as pd, gzip, re, os, json, time

OUT = "/root/autodl-tmp/dili_spec"
os.makedirs(OUT, exist_ok=True)
GENES_F = "/root/autodl-tmp/virtual-liver/processed/liver_genes_v4.txt"
TG_RMA = "/root/autodl-tmp/tggates_rma"
TG_META = "/root/autodl-tmp/tggates_sample_meta.csv"
ANNOT = "/root/autodl-tmp/GPL570.annot.gz"

genes = [l.strip() for l in open(GENES_F)]
G = len(genes); gidx = {g: i for i, g in enumerate(genes)}

t0 = time.time()
# 探针注释
ann = {}
with gzip.open(ANNOT, "rt", errors="ignore") as f:
    for line in f:
        if line.startswith("#") or line.startswith("ID") or line.startswith("probeset"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        pid = parts[0].strip()
        sym = ""
        for p in parts[1:]:
            m = re.search(r"^([A-Z0-9][A-Z0-9\-\.]{1,15})$", p.strip())
            if m and p.strip().upper() == p.strip() and not p.strip().startswith("EN"):
                sym = p.strip(); break
        if pid and sym:
            ann[pid] = sym
print(f"annot {len(ann)} | {time.time()-t0:.0f}s", flush=True)

meta = pd.read_csv(TG_META)
meta["BARCODE"] = meta["BARCODE"].astype(str).str.zfill(12)
meta["COMPOUND_NAME"] = meta["COMPOUND_NAME"].astype(str)
meta["DOSE_LEVEL"] = meta["DOSE_LEVEL"].astype(str)
# 只用 24 hr 体外 (μM 单位) — 与之前 Fig5a 口径一致
meta = meta[(meta["DOSE_UNIT"] == "μM") & (meta["SACRIFICE_PERIOD"] == "24 hr")]
print(f"meta 24hr μM: {len(meta)} rows, {meta.COMPOUND_NAME.nunique()} compounds", flush=True)

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

avail = {norm(f[:-4]): f[:-4] for f in os.listdir(TG_RMA) if f.endswith(".csv")}
comp_rows = {}
dose_mats = {}
for k, fname in sorted(avail.items()):
    sub = meta[meta["COMPOUND_NAME"].map(norm) == k]
    if len(sub) < 4:
        continue
    bc = {lv: sub[sub["DOSE_LEVEL"] == lv]["BARCODE"].tolist() for lv in ["Control", "Low", "Middle", "High"]}
    if len(bc["Control"]) < 2:
        continue
    df = pd.read_csv(f"{TG_RMA}/{fname}.csv")
    df["gene"] = df["probeset"].map(ann)
    df = df.dropna(subset=["gene"])
    df = df[df["gene"].isin(gidx)]
    gm = df.groupby("gene").mean(numeric_only=True)
    ccols = [b for b in bc["Control"] if b in gm.columns]
    if len(ccols) < 2:
        continue
    c0 = gm[ccols].mean(1)
    comp_name = sub["COMPOUND_NAME"].iloc[0]
    mats = {}
    for lv in ["Low", "Middle", "High"]:
        cols = [b for b in bc[lv] if b in gm.columns]
        if len(cols) >= 2:
            d = (gm[cols].mean(1) - c0).reindex(genes).fillna(0).to_numpy().astype(np.float32)
            mats[lv] = d
    if mats:
        dose_mats[comp_name] = mats
        comp_rows[comp_name] = {"n_doses": len(mats)}
print(f"compounds with logFC: {len(dose_mats)} | {time.time()-t0:.0f}s", flush=True)

comps = sorted(dose_mats)
arr = np.zeros((len(comps), 3, G), np.float32)
for i, c in enumerate(comps):
    for j, lv in enumerate(["Low", "Middle", "High"]):
        if lv in dose_mats[c]:
            arr[i, j] = dose_mats[c][lv]
np.savez(f"{OUT}/tggates_logfc.npz", logfc=arr, compounds=np.array(comps))
json.dump({"compounds": comps, "dose_levels": ["Low", "Middle", "High"]},
          open(f"{OUT}/tggates_logfc_meta.json", "w"), indent=1)
print(f"TGGATES_LOGFC_DONE shape={arr.shape}", flush=True)
