#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HepaDiff command-line interface.

Usage
-----
# zero-shot gene knockout in a chosen cell type
hepadiff infer --ko CYP2E1 --cell centrilobular-hepatocyte --n 2000 --out cyp2e1_ko

# in-library compound perturbation
hepadiff infer --compound "(+)-JQ1" --cell hepatocyte --n 2000 --out jq1_hep

# disease-condition generation (no perturbation)
hepadiff infer --condition NASH_MASH --cell "hepatic stellate cell" --n 2000 --out nash_hsc

Outputs (prefix = --out)
------------------------
<prefix>_cells.npz   generated cells (token matrix, gene panel, metadata)
<prefix>_delta.csv   per-gene mean expression delta vs healthy control, sorted
"""
import argparse
import os

import numpy as np

D_MODEL, N_HEAD, N_LAYER = 512, 8, 12
CHUNK = 48
GID_DIM = 256
MASK_TOKEN = 51
ALPHA = -0.5  # zero-class bias calibrated on held-out data

DEFAULT_CKPT = "hepadiff_step200000.pt"


def _bundled(name):
    """Path to a data file bundled inside the installed package."""
    try:
        from importlib.resources import files
        return str(files("hepadiff").joinpath("data", name))
    except Exception:
        return os.path.join(os.path.dirname(__file__), "data", name)


def _build_model(n_ct, n_cond, n_pert, g, g_pad, n_chunk, dev):
    import torch
    import torch.nn as nn

    class HepaDiff(nn.Module):
        def __init__(self):
            super().__init__()
            self.g, self.g_pad, self.n_chunk = g, g_pad, n_chunk
            self.gene_emb = nn.Embedding(52, D_MODEL)
            self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
            self.pos_emb = nn.Parameter(torch.randn(1, n_chunk + 4, D_MODEL) * 0.02)
            self.ct_emb = nn.Embedding(n_ct, D_MODEL)
            self.cond_emb = nn.Embedding(n_cond, D_MODEL)
            self.gene_id_emb = nn.Embedding(g, GID_DIM)
            self.pert_gene_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
            self.pert_free_emb = nn.Embedding(n_pert, GID_DIM)
            self.pert_free_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
            self.delta_head = nn.Sequential(nn.Linear(D_MODEL * 3, 512), nn.GELU(),
                                            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, g))
            self.z0_proj = nn.Linear(D_MODEL, D_MODEL)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, batch_first=True,
                                           norm_first=True, activation="gelu"), N_LAYER)
            self.decompress = nn.Linear(D_MODEL, CHUNK * 52)

        def z0_token(self, x0):
            e = self.gene_emb(x0.long()).view(x0.shape[0], self.n_chunk, CHUNK * D_MODEL)
            return self.z0_proj(self.compress(e).mean(1))

        def pert_vector(self, pert, pgid, null_pert):
            v = torch.zeros(pert.shape[0], D_MODEL, device=pert.device)
            is_null = pert.eq(null_pert)
            gm = (pgid >= 0) & ~is_null
            if gm.any():
                v[gm] = self.pert_gene_mlp(self.gene_id_emb(pgid[gm]))
            fm = (pgid < 0) & ~is_null
            if fm.any():
                v[fm] = self.pert_free_mlp(self.pert_free_emb(pert[fm]))
            return v

        def forward(self, x, ct, cond, pert, z0, pgid, null_pert):
            B = x.shape[0]
            e = self.gene_emb(x.long()).view(B, self.n_chunk, CHUNK * D_MODEL)
            z = self.compress(e)
            pv = self.pert_vector(pert, pgid, null_pert)
            head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0], 1)
            z = torch.cat([head, z], 1) + self.pos_emb
            z = self.encoder(z)[:, 4:]
            return self.decompress(z).view(B, self.g_pad, 52)

    return HepaDiff().to(dev)


def run_infer(args):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = np.load(args.reference, allow_pickle=True)
    celltypes = [str(c) for c in ref["celltypes"]]
    genes = [str(g) for g in ref["genes"]] if args.genes is None else \
        [l.strip() for l in open(args.genes)]
    G = len(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    n_chunk = (G + CHUNK - 1) // CHUNK
    G_PAD = n_chunk * CHUNK
    edges = np.load(args.edges)
    lut = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)

    if args.list_cells:
        print("Available cell types (with healthy reference):")
        for c in celltypes:
            print(" ", c)
        return

    if not os.path.exists(args.ckpt):
        raise SystemExit(
            f"model checkpoint not found: {args.ckpt}\n"
            f"Download '{DEFAULT_CKPT}' (~276 MB) from the release link in the README "
            f"and place it in the current directory, or pass --ckpt <path>.")

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ct_vocab, cond_vocab, pert_vocab = ck["ct_vocab"], ck["cond_vocab"], ck["pert_vocab"]
    NULL_CT, NULL_COND, NULL_PERT = len(ct_vocab), len(cond_vocab), len(pert_vocab)

    if args.list_compounds:
        try:
            vocab = {l.strip() for l in open(_bundled("compound_vocab_210.txt"))}
        except Exception:
            vocab = None
        comp = sorted(k for k in pert_vocab if (k in vocab if vocab else k not in gidx))
        print(f"{len(comp)} in-library compounds:")
        for c in comp:
            print(" ", c)
        return

    model = _build_model(len(ct_vocab) + 1, len(cond_vocab) + 1, len(pert_vocab) + 1,
                         G, G_PAD, n_chunk, dev)
    model.load_state_dict(ck["model"])
    model.eval()

    # resolve cell type / reference
    cell = args.cell
    alias = {"centrilobular-hepatocyte": "centrilobular region hepatocyte",
             "periportal-hepatocyte": "periportal region hepatocyte",
             "midzonal-hepatocyte": "midzonal region hepatocyte",
             "HSC": "hepatic stellate cell",
             "LSEC": "endothelial cell of hepatic sinusoid"}
    cell = alias.get(cell, cell)
    if cell not in celltypes:
        raise SystemExit(f"cell type '{args.cell}' has no healthy reference; "
                         f"choose from: {celltypes}")
    z0 = torch.from_numpy(ref[f"z0__{cell}"]).to(dev)
    Ec = ref[f"Ec__{cell}"]

    # resolve perturbation
    if args.ko and args.compound:
        raise SystemExit("use either --ko or --compound, not both")
    if args.ko:
        gene = args.ko.upper()
        if gene not in gidx:
            raise SystemExit(f"{gene} not in gene panel")
        pert_id = pert_vocab.get(gene, NULL_PERT)
        pgid = gidx[gene]
        tag = f"KO:{gene}"
    elif args.compound:
        if args.compound not in pert_vocab:
            raise SystemExit(f"'{args.compound}' is not in the perturbation library; "
                             f"run 'hepadiff infer --cell hepatocyte --list-compounds' to see "
                             f"available compounds, or use --ko <target gene> for a "
                             f"target-based proxy")
        pert_id, pgid = pert_vocab[args.compound], -1
        tag = f"compound:{args.compound}"
    else:
        pert_id, pgid = NULL_PERT, -1
        tag = "none"

    cond_id = cond_vocab.get(args.condition, NULL_COND)
    if args.condition not in cond_vocab:
        print(f"[warn] condition '{args.condition}' unseen; falling back to null condition")
    ct_id = ct_vocab.get(cell, NULL_CT)

    # generate
    torch.manual_seed(args.seed)
    outs = []
    with torch.no_grad():
        for i0 in range(0, args.n, 250):
            b = min(250, args.n - i0)
            xm = torch.full((b, G_PAD), MASK_TOKEN, dtype=torch.int8, device=dev)
            ctt = torch.full((b,), ct_id, dtype=torch.long, device=dev)
            cn = torch.full((b,), cond_id, dtype=torch.long, device=dev)
            pt = torch.full((b,), pert_id, dtype=torch.long, device=dev)
            pg = torch.full((b,), pgid, dtype=torch.long, device=dev)
            logits = model(xm, ctt, cn, pt, z0.expand(b, -1), pg, NULL_PERT).float()
            logits[..., 0] += ALPHA
            t = torch.distributions.Categorical(logits=logits.view(-1, 52)).sample().view(b, G_PAD)
            outs.append(t[:, :G].cpu().numpy().astype(np.int8))
    cells = np.concatenate(outs)

    np.savez_compressed(f"{args.out}_cells.npz", tokens=cells, genes=np.array(genes),
                        cell=cell, perturbation=tag, condition=args.condition)
    expr = lut[cells.astype(int)].mean(0)
    delta = expr - Ec
    import pandas as pd
    rep = pd.DataFrame({"gene": genes, "mean_expr_generated": expr,
                        "mean_expr_control": Ec, "delta": delta})
    rep = rep.reindex(rep["delta"].abs().sort_values(ascending=False).index)
    rep.to_csv(f"{args.out}_delta.csv", index=False)

    print(f"generated {len(cells)} cells | cell={cell} | pert={tag} | cond={args.condition}")
    print("top up-regulated:")
    for _, r in rep.nlargest(10, "delta").iterrows():
        print(f"  {r['gene']:15s} delta={r['delta']:+.4f}")
    print("top down-regulated:")
    for _, r in rep.nsmallest(10, "delta").iterrows():
        print(f"  {r['gene']:15s} delta={r['delta']:+.4f}")
    print(f"saved {args.out}_cells.npz and {args.out}_delta.csv")


def main():
    ap = argparse.ArgumentParser(prog="hepadiff",
                                 description="HepaDiff — virtual liver tissue inference")
    sub = ap.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("infer", help="generate virtual cells under a perturbation/condition")
    inf.add_argument("--ckpt", default=DEFAULT_CKPT,
                     help=f"model checkpoint path (default: ./{DEFAULT_CKPT})")
    inf.add_argument("--reference", default=_bundled("hepadiff_reference.npz"),
                     help="hepadiff_reference.npz (default: bundled)")
    inf.add_argument("--genes", default=None,
                     help="gene panel txt (default: taken from reference)")
    inf.add_argument("--edges", default=_bundled("liver_bin_edges_v4.npy"),
                     help="liver_bin_edges_v4.npy (default: bundled)")
    inf.add_argument("--ko", default=None, help="gene symbol to knock out (zero-shot)")
    inf.add_argument("--compound", default=None, help="in-library compound name")
    inf.add_argument("--condition", default="healthy",
                     help="condition label (default: healthy)")
    inf.add_argument("--cell", default="hepatocyte",
                     help="cell type (default: hepatocyte; see --list-cells)")
    inf.add_argument("--n", type=int, default=2000, help="number of cells to generate")
    inf.add_argument("--seed", type=int, default=0)
    inf.add_argument("--out", default="hepadiff_out", help="output prefix")
    inf.add_argument("--list-cells", action="store_true",
                     help="list available cell types and exit")
    inf.add_argument("--list-compounds", action="store_true",
                     help="list in-library compounds and exit (requires checkpoint)")
    args = ap.parse_args()

    if args.command == "infer":
        if args.list_cells:
            # needs only the bundled reference; no torch required
            ref = np.load(args.reference, allow_pickle=True)
            print("Available cell types (with healthy reference):")
            for c in ref["celltypes"]:
                print(" ", str(c))
            return
        run_infer(args)


if __name__ == "__main__":
    main()
