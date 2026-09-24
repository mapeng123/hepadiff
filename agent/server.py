#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HepaDiff MCP server — a virtual liver that LLM agents can perturb.

Exposes the HepaDiff masked-diffusion model (19.6M-cell liver corpus,
69M parameters) as MCP tools that any MCP-capable LLM client
(Kimi, ChatGPT, Claude, ...) can call in natural language.

Tools
-----
list_cell_types()                    cell types with a healthy reference
list_compounds()                     210 in-library compounds
knockout(gene, cell_type, ...)       zero-shot gene knockout (any of 16k panel genes)
compound_perturbation(name, ...)     in-library compound perturbation
generate_condition(condition, ...)   disease-state cell generation
query_drug_target(drug)              ChEMBL target / action-type / MoA lookup

Run (streamable HTTP, for remote MCP clients):
    python server.py --host 0.0.0.0 --port 6006

Run (stdio, for local MCP clients):
    python server.py --stdio
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

BASE = "/root/autodl-tmp"
PY = os.environ.get("HEPADIFF_PY", "/root/miniconda3/bin/python")
INFER = f"{BASE}/hepadiff_infer.py"
CKPT = f"{BASE}/virtual-liver/checkpoints_pmdm_v3/pmdm_v3_step200000.pt"
REF = f"{BASE}/hepadiff_reference.npz"
EDGES = f"{BASE}/toktest/liver_bin_edges_v4.npy"
HALLMARK = f"{BASE}/hallmark.gmt"
CHEMBL_MEC = f"{BASE}/dilirank_val/chembl_mechanisms.json"
DILI_TARGETS = f"{BASE}/dilirank_val/dilirank_targets.csv"
OUTDIR = f"{BASE}/hepadiff_agent/runs"
VOCAB_CACHE = f"{BASE}/hepadiff_agent/vocabs.json"
COMPOUND_LIST = f"{BASE}/st_tables/compound_vocab_210.txt"

os.makedirs(OUTDIR, exist_ok=True)

COMPOUNDS = [l.strip() for l in open(COMPOUND_LIST) if l.strip()] \
    if os.path.exists(COMPOUND_LIST) else []

# ---------------------------------------------------------------- vocab cache
def _load_vocabs():
    if os.path.exists(VOCAB_CACHE):
        return json.load(open(VOCAB_CACHE))
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref = np.load(REF, allow_pickle=True)
    genes = [str(g) for g in ref["genes"]]
    gidx = set(genes)
    vocabs = {
        "celltypes": [str(c) for c in ref["celltypes"]],
        "conditions": sorted(ck["cond_vocab"].keys()),
        "compounds": sorted(k for k in ck["pert_vocab"] if k not in gidx),
        "n_panel_genes": len(genes),
    }
    json.dump(vocabs, open(VOCAB_CACHE, "w"))
    return vocabs

VOCABS = _load_vocabs()

# ---------------------------------------------------------------- hallmark
def _load_hallmark():
    pw = {}
    if not os.path.exists(HALLMARK):
        return pw
    for line in open(HALLMARK):
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            pw[parts[0]] = parts[2:]
    return pw

HALLMARKS = _load_hallmark()

def _pathway_scan(delta_csv, top=5):
    """Rank Hallmark pathways by mean delta of member genes."""
    df = pd.read_csv(delta_csv)
    d = dict(zip(df["gene"], df["delta"]))
    rows = []
    for name, genes in HALLMARKS.items():
        vals = [d[g] for g in genes if g in d]
        if len(vals) >= 5:
            rows.append((name, float(np.mean(vals)), len(vals)))
    rows.sort(key=lambda x: x[1])
    down = [{"pathway": n, "mean_delta": round(v, 4), "n_genes": k} for n, v, k in rows[:top]]
    up = [{"pathway": n, "mean_delta": round(v, 4), "n_genes": k} for n, v, k in rows[-top:][::-1]]
    return {"up_regulated_pathways": up, "down_regulated_pathways": down}

# ---------------------------------------------------------------- inference
def _match_cell(cell_type):
    ct = VOCABS["celltypes"]
    if cell_type in ct:
        return cell_type
    low = cell_type.lower()
    cand = [c for c in ct if low in c.lower() or c.lower() in low]
    if len(cand) == 1:
        return cand[0]
    alias = {"hepatocyte": "centrilobular region hepatocyte",
             "hsc": "hepatic stellate cell",
             "stellate": "hepatic stellate cell",
             "lsec": "endothelial cell of hepatic sinusoid",
             "kupffer": "kupffer cell", "macrophage": "kupffer cell"}
    for k, v in alias.items():
        if k in low:
            for c in ct:
                if v in c.lower():
                    return c
    raise ValueError(f"cell type '{cell_type}' not matched. Call list_cell_types() for valid names.")

def _run_infer(extra_args, tag, n_cells, seed):
    key = hashlib.md5(json.dumps([extra_args, n_cells, seed]).encode()).hexdigest()[:12]
    prefix = f"{OUTDIR}/{tag}_{key}"
    delta_csv = f"{prefix}_delta.csv"
    if not os.path.exists(delta_csv):
        cmd = [PY, INFER, "--ckpt", CKPT, "--reference", REF, "--edges", EDGES,
               "--n", str(n_cells), "--seed", str(seed), "--batch", "250",
               "--out", prefix] + extra_args
        env = dict(os.environ, OMP_NUM_THREADS="16")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        if p.returncode != 0:
            raise RuntimeError(f"inference failed: {p.stderr[-800:]}")
    df = pd.read_csv(delta_csv)
    result = {
        "n_cells": n_cells, "seed": seed,
        "top_up_regulated": df.nlargest(15, "delta")[["gene", "delta"]].round(4).to_dict("records"),
        "top_down_regulated": df.nsmallest(15, "delta")[["gene", "delta"]].round(4).to_dict("records"),
        "pathways": _pathway_scan(delta_csv),
        "delta_csv": delta_csv,
    }
    return result

# ---------------------------------------------------------------- MCP tools
from fastmcp import FastMCP
mcp = FastMCP("HepaDiff Virtual Liver")

@mcp.tool
def list_cell_types() -> dict:
    """List the liver cell types that have a healthy reference state in this
    release (8 major types: hepatocyte zonation layers, stellate, Kupffer,
    cholangiocyte, sinusoidal endothelial) and can serve as the cellular
    context for perturbation experiments."""
    return {"cell_types": VOCABS["celltypes"], "n": len(VOCABS["celltypes"])}

@mcp.tool
def list_compounds() -> dict:
    """List the 210 compounds in HepaDiff's in-library perturbation vocabulary
    (seen during training). For compounds NOT in this list, use
    query_drug_target to find the target gene and then run a target-based
    knockout instead."""
    return {"compounds": COMPOUNDS, "n": len(COMPOUNDS)}

@mcp.tool
def knockout(gene: str, cell_type: str = "centrilobular region hepatocyte",
             n_cells: int = 500, seed: int = 0) -> dict:
    """Zero-shot knockout of ANY of the ~16,000 panel genes in a chosen liver
    cell type. Returns per-gene expression deltas vs healthy control and
    Hallmark pathway shifts. Typical runtime: 2-5 min on CPU for 500 cells.
    gene: HGNC symbol, e.g. PNPLA3, HMGCR, DRD2.
    cell_type: call list_cell_types() for valid names (fuzzy matching applied).
    """
    cell = _match_cell(cell_type)
    r = _run_infer(["--ko", gene.upper(), "--cell", cell], f"ko_{gene.upper()}", n_cells, seed)
    r.update({"perturbation": f"knockout:{gene.upper()}", "cell_type": cell})
    return r

@mcp.tool
def compound_perturbation(compound: str, cell_type: str = "centrilobular region hepatocyte",
                          n_cells: int = 500, seed: int = 0) -> dict:
    """Perturb liver cells with an in-library compound (see list_compounds).
    For compounds outside the library, find the target via query_drug_target
    and use knockout() on the target gene instead (inhibitors only; agonist
    predictions are currently weaker)."""
    cell = _match_cell(cell_type)
    r = _run_infer(["--compound", compound, "--cell", cell], f"cpd_{compound}", n_cells, seed)
    r.update({"perturbation": f"compound:{compound}", "cell_type": cell})
    return r

@mcp.tool
def generate_condition(condition: str, cell_type: str = "hepatic stellate cell",
                       n_cells: int = 500, seed: int = 0) -> dict:
    """Generate liver cells under a disease condition (e.g. NASH_MASH,
    fibrosis, HCC) without perturbation, and report expression shifts vs the
    healthy reference of the same cell type."""
    cell = _match_cell(cell_type)
    r = _run_infer(["--condition", condition, "--cell", cell], f"cond_{condition}", n_cells, seed)
    r.update({"condition": condition, "cell_type": cell})
    return r

@mcp.tool
def query_drug_target(drug_name: str) -> dict:
    """Look up a drug's annotated target gene(s), action type (inhibitor /
    agonist / ...) and mechanism of action from ChEMBL. Use this BEFORE
    perturbing a compound that is not in the HepaDiff library: if the target
    gene is in the panel and the action type is inhibition, a target-gene
    knockout approximates the on-target mechanism."""
    hits = []
    if os.path.exists(CHEMBL_MEC):
        data = json.load(open(CHEMBL_MEC))
        recs = data if isinstance(data, list) else list(data.values())
        low = drug_name.lower()
        for rec in recs:
            blob = json.dumps(rec).lower()
            if low in blob:
                hits.append(rec)
        hits = hits[:10]
    out = {"drug": drug_name, "chembl_mechanism_hits": hits}
    if os.path.exists(DILI_TARGETS):
        df = pd.read_csv(DILI_TARGETS)
        mask = df.apply(lambda c: c.astype(str).str.lower().str.contains(drug_name.lower(), na=False)).any(axis=1)
        if mask.any():
            sub = df[mask].head(10).astype(object).where(pd.notna(df), None)
            out["target_table_rows"] = sub.to_dict("records")
    if not hits and "target_table_rows" not in out:
        out["note"] = "no annotation found; try the INN/generic name"
    return out

@mcp.tool
def server_info() -> dict:
    """Describe this server: model version, corpus scale, capability boundary."""
    return {
        "model": "HepaDiff v3 (masked diffusion, 69M parameters, step 200000)",
        "corpus": "19,585,593 liver cells from 483 datasets; 181 cell types; 26 conditions",
        "reference_cell_types_this_release": len(VOCABS["celltypes"]),
        "panel_genes": VOCABS["n_panel_genes"],
        "in_library_compounds": len(COMPOUNDS),
        "capability_boundary": [
            "mechanism triage, not clinical severity classification",
            "target-annotation pathway recovers on-target mechanism only "
            "(no off-target / metabolite / host-dependent effects)",
            "agonist predictions weaker than inhibitors",
            "10-seed evaluation recommended for publication-grade claims",
        ],
        "citation": "Ma P, Guo Q, Sun Y, Ou Y. HepaDiff: a perturbable virtual liver "
                    "for mechanism-driven hepatology and drug safety. "
                    "https://github.com/mapeng123/hepadiff",
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--stdio", action="store_true", help="stdio transport (local MCP clients)")
    ap.add_argument("--mcp-only", action="store_true", help="serve MCP only, no REST/OpenAPI")
    args = ap.parse_args()
    if args.stdio:
        mcp.run()
    elif args.mcp_only:
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path="/mcp")
    else:
        # Combined server: MCP at /mcp + REST (OpenAPI) for ChatGPT GPT Actions
        import uvicorn
        from fastapi import Body, FastAPI

        mcp_app = mcp.http_app(path="/")
        app = FastAPI(
            title="HepaDiff Virtual Liver",
            version="0.1.0",
            description=("A perturbable virtual liver (19.6M-cell corpus, masked-diffusion "
                         "model). Run zero-shot gene knockouts, compound perturbations and "
                         "disease-state generation in liver cell types, and query drug "
                         "target annotations. Mechanism triage, not severity classification."),
            lifespan=mcp_app.lifespan,
        )

        @app.get("/", operation_id="health")
        def root():
            return server_info()

        @app.post("/tools/list_cell_types", operation_id="list_cell_types")
        def r_list_cell_types():
            return list_cell_types()

        @app.post("/tools/list_compounds", operation_id="list_compounds")
        def r_list_compounds():
            return list_compounds()

        @app.post("/tools/knockout", operation_id="knockout")
        def r_knockout(gene: str = Body(..., embed=True), cell_type: str = Body("centrilobular region hepatocyte", embed=True),
                       n_cells: int = Body(500, embed=True), seed: int = Body(0, embed=True)):
            return knockout(gene, cell_type, n_cells, seed)

        @app.post("/tools/compound_perturbation", operation_id="compound_perturbation")
        def r_compound(compound: str = Body(..., embed=True), cell_type: str = Body("centrilobular region hepatocyte", embed=True),
                       n_cells: int = Body(500, embed=True), seed: int = Body(0, embed=True)):
            return compound_perturbation(compound, cell_type, n_cells, seed)

        @app.post("/tools/generate_condition", operation_id="generate_condition")
        def r_condition(condition: str = Body(..., embed=True), cell_type: str = Body("hepatic stellate cell", embed=True),
                        n_cells: int = Body(500, embed=True), seed: int = Body(0, embed=True)):
            return generate_condition(condition, cell_type, n_cells, seed)

        @app.post("/tools/query_drug_target", operation_id="query_drug_target")
        def r_target(drug_name: str = Body(..., embed=True)):
            return query_drug_target(drug_name)

        app.mount("/mcp", mcp_app)
        uvicorn.run(app, host=args.host, port=args.port)
