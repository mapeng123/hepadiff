#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HepaDiff grounded virtual-liver agent backend.

Knockouts are restricted to the trained perturbation vocabulary. Unsupported
requests return ABSTAIN rather than an uncalibrated prediction. Generation uses
the manuscript-selected checkpoint and four-step unbiased-logit decoder.
"""
import argparse, hashlib, json, os, time
from importlib.resources import files
import numpy as np
import pandas as pd
import torch
from hepadiff.model import decode, load_model

BASE = os.environ.get("HEPADIFF_ROOT", "/root/autodl-tmp")
CKPT = os.environ.get("HEPADIFF_CKPT", "hepadiff_step155000.pt")
_PACKAGE_DATA = files("hepadiff").joinpath("data")
REF = os.environ.get("HEPADIFF_REF", str(_PACKAGE_DATA.joinpath("hepadiff_reference.npz")))
EDGES_F = os.environ.get("HEPADIFF_EDGES", str(_PACKAGE_DATA.joinpath("liver_bin_edges_v4.npy")))
HALLMARK = os.environ.get("HEPADIFF_HALLMARK", f"{BASE}/hallmark.gmt")
CHEMBL_MEC = os.environ.get("HEPADIFF_CHEMBL", f"{BASE}/dilirank_val/chembl_mechanisms.json")
DILI_TARGETS = os.environ.get("HEPADIFF_DILI_TARGETS", f"{BASE}/dilirank_val/dilirank_targets.csv")
DILIRANK = os.environ.get("HEPADIFF_DILIRANK", f"{BASE}/dilirank_val/dilirank2.csv")
LR_PAIRS = os.environ.get("HEPADIFF_LR_PAIRS", f"{BASE}/liverbench_v2/lr_pairs_cpdb_v5.csv")
OUTDIR = os.environ.get("HEPADIFF_AGENT_OUT", "hepadiff_agent_runs")
os.makedirs(OUTDIR, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_ref = np.load(REF, allow_pickle=True)
REF_CTS = [str(c) for c in _ref["celltypes"]]
REF_EC = {str(c): _ref[f"Ec__{c}"].astype(np.float32) for c in REF_CTS}
genes = [str(g) for g in _ref["genes"]]
G = len(genes)
gidx = {g: i for i, g in enumerate(genes)}
edges = np.load(EDGES_F)
_lut_np = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

model, ck = load_model(CKPT, G, DEV)
STEP = ck["step"]
ct_vocab, cond_vocab, pert_vocab = ck["ct_vocab"], ck["cond_vocab"], ck["pert_vocab"]
NULL_CT, NULL_COND, NULL_PERT = len(ct_vocab), len(cond_vocab), len(pert_vocab)
SPV = ck.get("species_vocab", {"human": 0, "mouse": 1, "rat": 2, "cellline": 3})
TRAINED_GENES = set(pert_vocab.keys()) & set(genes)
print(f"[HepaDiff agent] checkpoint step={STEP}; reference cell types={len(REF_CTS)}", flush=True)

# ---------------- external annotation tables ----------------
_dili = pd.read_csv(DILIRANK)
_lr = pd.read_csv(LR_PAIRS)
_lr_set = set(zip(_lr["ligand"], _lr["receptor"]))
_chembl = json.load(open(CHEMBL_MEC)) if os.path.exists(CHEMBL_MEC) else []

_pw = {}
if os.path.exists(HALLMARK):
    for line in open(HALLMARK):
        p = line.rstrip("\n").split("\t")
        if len(p) > 2:
            _pw[p[0]] = p[2:]

def _pathway_scan(delta, genes_order, top=5):
    d = dict(zip(genes_order, delta))
    rows = []
    for name, gs in _pw.items():
        vals = [d[g] for g in gs if g in d]
        if len(vals) >= 5:
            rows.append((name, float(np.mean(vals)), len(vals)))
    rows.sort(key=lambda x: x[1])
    return {"up_regulated_pathways": [{"pathway": n, "mean_delta": round(v, 4), "n_genes": k} for n, v, k in rows[-top:][::-1]],
            "down_regulated_pathways": [{"pathway": n, "mean_delta": round(v, 4), "n_genes": k} for n, v, k in rows[:top]]}

# ---------------- helpers ----------------
def _match_cell(cell_type):
    if cell_type in REF_CTS:
        return cell_type
    low = cell_type.lower()
    cand = [c for c in REF_CTS if low in c.lower() or c.lower() in low]
    if len(cand) == 1:
        return cand[0]
    alias = {"hepatocyte": "hepatocyte", "hsc": "hepatic stellate cell",
             "stellate": "hepatic stellate cell", "lsec": "endothelial cell of hepatic sinusoid",
             "kupffer": "kupffer cell", "macrophage": "kupffer cell",
             "zone3": "centrilobular region hepatocyte", "centrilobular": "centrilobular region hepatocyte",
             "periportal": "periportal region hepatocyte", "midzonal": "midzonal region hepatocyte",
             "cholangiocyte": "cholangiocyte", "endothelial": "endothelial cell of hepatic sinusoid"}
    for k, v in alias.items():
        if k in low:
            for c in REF_CTS:
                if v in c.lower():
                    return c
    raise ValueError(f"cell type '{cell_type}' not matched; call list_cell_types()")

def _generate(ct, cond_id, pert_id, pgid, ptype_id, n, seed):
    return decode(
        model, cell_type_id=ct_vocab.get(ct, NULL_CT), condition_id=cond_id,
        perturbation_id=pert_id, gene_id=pgid, perturbation_type_id=ptype_id,
        species_id=SPV.get("human", 0), n_cells=n, seed=seed, steps=4, device=DEV)

def _delta_report(tok, ct, tag, extra):
    expr = _lut_np[tok.astype(int)].mean(0)
    delta = expr - REF_EC[ct]
    df = pd.DataFrame({"gene": genes, "delta": delta})
    key = hashlib.md5(json.dumps([tag, extra, time.time()]).encode()).hexdigest()[:10]
    df.round(5).to_csv(f"{OUTDIR}/{tag}_{key}_delta.csv", index=False)
    out = {
        "status": "ok", "cell_type": ct, "n_cells": int(tok.shape[0]),
        "checkpoint_step": int(STEP), "decoder": "4-step-unbiased",
        "top_up_regulated": df.nlargest(15, "delta")[["gene", "delta"]].round(4).to_dict("records"),
        "top_down_regulated": df.nsmallest(15, "delta")[["gene", "delta"]].round(4).to_dict("records"),
        "pathways": _pathway_scan(delta, genes),
    }
    out.update(extra)
    return out

def _abstain(reason, suggestion=""):
    return {"status": "ABSTAIN", "reason": reason, "suggestion": suggestion,
            "capability_note": f"HepaDiff supports calibrated knockout prediction only for "
                               f"its {len(TRAINED_GENES)} trained panel genes. "
                               "Predictions outside this vocabulary are withheld by design."}

# ---------------- tool functions ----------------
def server_info() -> dict:
    return {
        "model": f"HepaDiff (masked discrete diffusion, checkpoint step {STEP})",
        "decoder": "4-step unbiased-logit",
        "corpus": "18.48 million liver cells from 482 datasets; 1.75 million perturbation cells",
        "trained_perturbation_genes": len(TRAINED_GENES),
        "reference_cell_types": REF_CTS,
        "conditions": sorted(cond_vocab.keys()),
        "capability_boundary": [
            "knockout predictions ONLY for genes in the trained vocabulary; others are refused (abstain)",
            "compound predictions are mechanism-level via annotated trained targets (inhibitors; agonists weaker)",
            "mechanism triage, not clinical severity classification",
            "multi-seed evaluation recommended for publication-grade claims",
        ],
    }

def list_cell_types() -> dict:
    return {"cell_types": REF_CTS, "n": len(REF_CTS)}

def list_conditions() -> dict:
    return {"conditions": sorted(cond_vocab.keys()), "n": len(cond_vocab)}

def check_gene(gene: str) -> dict:
    g = gene.upper().strip()
    return {"gene": g, "in_panel": g in gidx, "trained": g in TRAINED_GENES,
            "knockout_supported": g in TRAINED_GENES}

def knockout(gene: str, cell_type: str = "hepatocyte", n_cells: int = 500,
             seed: int = 0, k_seeds: int = 1) -> dict:
    g = gene.upper().strip()
    if g not in TRAINED_GENES:
        return _abstain(
            f"{g} is NOT in HepaDiff's trained perturbation vocabulary; "
            f"a knockout prediction would be uncalibrated fabrication.",
            "Choose a trained gene (check_gene) or use a trained paralog/pathway member.")
    ct = _match_cell(cell_type)
    tok = _generate(ct, NULL_COND, pert_vocab[g], gidx[g], 0, n_cells, seed)
    out = _delta_report(tok, ct, f"ko_{g}", {"perturbation": f"knockout:{g}", "seed": seed})
    if k_seeds > 1:
        toks = [tok] + [_generate(ct, NULL_COND, pert_vocab[g], gidx[g], 0, n_cells, seed + i)
                        for i in range(1, k_seeds)]
        deltas = np.stack([_lut_np[t.astype(int)].mean(0) - REF_EC[ct] for t in toks])
        disp = deltas.std(0)
        top = np.argsort(-np.abs(deltas.mean(0)))[:100]
        out["uncertainty"] = {"k_seeds": k_seeds,
                              "mean_dispersion_top100": round(float(disp[top].mean()), 4)}
    return out

def generate_condition(condition: str, cell_type: str = "hepatic stellate cell",
                       n_cells: int = 500, seed: int = 0) -> dict:
    if condition not in cond_vocab:
        low = condition.lower()
        cand = [c for c in cond_vocab if low in c.lower() or c.lower() in low]
        if len(cand) == 1:
            condition = cand[0]
        else:
            return _abstain(f"condition '{condition}' not in vocabulary",
                            f"call list_conditions(); close matches: {cand[:8]}")
    ct = _match_cell(cell_type)
    tok = _generate(ct, cond_vocab[condition], NULL_PERT, -1, 0, n_cells, seed)
    return _delta_report(tok, ct, f"cond_{condition}", {"condition": condition, "seed": seed})

def query_drug_target(drug_name: str) -> dict:
    low = drug_name.lower()
    hits = [rec for rec in _chembl if isinstance(rec, dict)
            and low in json.dumps(rec).lower()][:5]
    out = {"drug": drug_name, "chembl_mechanism_hits": hits}
    if os.path.exists(DILI_TARGETS):
        df = pd.read_csv(DILI_TARGETS)
        mask = df.apply(lambda c: c.astype(str).str.lower().str.contains(low, na=False)).any(axis=1)
        if mask.any():
            sub = df[mask].head(10).astype(object).where(pd.notna(df), None)
            out["target_table_rows"] = sub.to_dict("records")
    if not hits and "target_table_rows" not in out:
        out["note"] = "no annotation found; try the INN/generic name"
    return out

def query_drug_library(compound: str) -> dict:
    low = compound.lower().strip()
    exact = _dili[_dili["CompoundName"].str.lower() == low]
    sub = exact if len(exact) else _dili[_dili["CompoundName"].str.lower().str.contains(low, na=False)]
    if len(sub):
        r = sub.iloc[0]
        return {"compound": r["CompoundName"], "in_dilirank": True,
                "vDILI_concern": r["vDILI-Concern"], "severity_class": int(r["SeverityClass"]),
                "source": "DILIrank v2 (FDA/LTKB)"}
    return {"compound": compound, "in_dilirank": False,
            "source": "DILIrank v2 (FDA/LTKB); absence verified against 1,336-compound list"}

def query_ligand_receptor(ligand: str, receptor: str) -> dict:
    a, b = ligand.upper().strip(), receptor.upper().strip()
    if (a, b) in _lr_set:
        row = _lr[(_lr["ligand"] == a) & (_lr["receptor"] == b)].iloc[0]
        return {"pair": f"{a}-{b}", "curated": True,
                "classification": row["classification"], "source_db": "CellPhoneDB v5"}
    if (b, a) in _lr_set:
        row = _lr[(_lr["ligand"] == b) & (_lr["receptor"] == a)].iloc[0]
        return {"pair": f"{b}-{a}", "curated": True, "note": "direction reversed in database",
                "classification": row["classification"], "source_db": "CellPhoneDB v5"}
    return {"pair": f"{a}-{b}", "curated": False,
            "source_db": "CellPhoneDB v5; absence verified against curated interaction table"}

def compound_perturbation(compound: str, cell_type: str = "hepatocyte",
                          n_cells: int = 500, seed: int = 0) -> dict:
    info = query_drug_target(compound)
    tg = []
    for row in info.get("target_table_rows", []):
        for t in str(row.get("targets", "")).replace(";", ",").split(","):
            t = t.strip().upper()
            if t in TRAINED_GENES and t not in tg:
                tg.append(t)
    if not tg:
        dili = query_drug_library(compound)
        return _abstain(
            f"no trained-vocabulary target annotated for '{compound}'",
            "only on-target mechanism prediction via trained target genes is supported") | \
            {"dilirank": dili}
    ct = _match_cell(cell_type)
    per_target = []
    for t in tg[:3]:
        tok = _generate(ct, NULL_COND, pert_vocab[t], gidx[t], 0, n_cells, seed)
        per_target.append(_delta_report(tok, ct, f"cpd_{compound}_{t}",
                                        {"target_gene": t}))
    return {"status": "ok", "compound": compound, "mode": "on-target mechanism (target-gene knockout)",
            "trained_targets_used": tg[:3], "cell_type": ct,
            "per_target": per_target,
            "dilirank": query_drug_library(compound)}

TOOLS = {
    "server_info": server_info,
    "list_cell_types": list_cell_types,
    "list_conditions": list_conditions,
    "check_gene": check_gene,
    "knockout": knockout,
    "generate_condition": generate_condition,
    "query_drug_target": query_drug_target,
    "query_drug_library": query_drug_library,
    "query_ligand_receptor": query_ligand_receptor,
    "compound_perturbation": compound_perturbation,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6010)
    args = ap.parse_args()
    import uvicorn
    from fastapi import FastAPI, Request
    app = FastAPI(title="HepaDiff Virtual Liver", version="0.2.0")

    @app.get("/")
    def root():
        return server_info()

    @app.post("/tools/{name}")
    async def call_tool(name: str, request: Request):
        if name not in TOOLS:
            return {"error": f"unknown tool {name}"}
        kwargs = await request.json()
        try:
            return TOOLS[name](**kwargs)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    uvicorn.run(app, host=args.host, port=args.port)
