#!/usr/bin/env python
"""Command-line inference for the HepaDiff virtual liver."""
from __future__ import annotations

import argparse
import os
from importlib.resources import files

import numpy as np
import pandas as pd

DEFAULT_CKPT = os.environ.get("HEPADIFF_CKPT", "hepadiff_step155000.pt")
CELL_ALIASES = {
    "centrilobular-hepatocyte": "centrilobular region hepatocyte",
    "periportal-hepatocyte": "periportal region hepatocyte",
    "midzonal-hepatocyte": "midzonal region hepatocyte",
    "HSC": "hepatic stellate cell",
    "LSEC": "endothelial cell of hepatic sinusoid",
}


def bundled(name: str) -> str:
    return str(files("hepadiff").joinpath("data", name))


def load_reference(path: str):
    ref = np.load(path, allow_pickle=True)
    return ref, [str(x) for x in ref["celltypes"]], [str(x) for x in ref["genes"]]


def run_infer(args) -> None:
    ref, reference_cells, genes = load_reference(args.reference)
    if args.list_cells:
        print(f"{len(reference_cells)} cell types with a healthy reference:")
        for cell in reference_cells:
            print(f"  {cell}")
        return

    import torch
    from .model import decode, load_model

    if not os.path.exists(args.ckpt):
        raise SystemExit(
            f"Model checkpoint not found: {args.ckpt}\n"
            "The step-155000 checkpoint is not stored in this git repository. "
            "Provide it with --ckpt or HEPADIFF_CKPT.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, checkpoint = load_model(args.ckpt, len(genes), device)
    ct_vocab = checkpoint["ct_vocab"]
    cond_vocab = checkpoint["cond_vocab"]
    pert_vocab = checkpoint["pert_vocab"]
    species_vocab = checkpoint.get(
        "species_vocab", {"human": 0, "mouse": 1, "rat": 2, "cellline": 3})
    null_condition = len(cond_vocab)
    null_perturbation = len(pert_vocab)
    gene_index = {gene: i for i, gene in enumerate(genes)}

    if args.list_conditions:
        print(f"{len(cond_vocab)} conditions:")
        for condition in sorted(cond_vocab):
            print(f"  {condition}")
        return
    if args.list_compounds:
        compounds = sorted(name for name in pert_vocab if name not in gene_index)
        print(f"{len(compounds)} in-library compounds:")
        for compound in compounds:
            print(f"  {compound}")
        return
    if args.check_gene:
        gene = args.check_gene.upper().strip()
        print(f"gene={gene}")
        print(f"in_panel={gene in gene_index}")
        print(f"trained_knockout={gene in gene_index and gene in pert_vocab}")
        return

    cell = CELL_ALIASES.get(args.cell, args.cell)
    if cell not in reference_cells or cell not in ct_vocab:
        raise SystemExit(
            f"Cell type '{args.cell}' is unavailable in the release reference/model. "
            "Run --list-cells for supported names.")
    if args.condition not in cond_vocab:
        raise SystemExit(
            f"Condition '{args.condition}' is not in the model vocabulary. "
            "Run --list-conditions; unseen conditions are not silently substituted.")
    if args.species not in species_vocab:
        raise SystemExit(
            f"Species '{args.species}' is unsupported; choose from {sorted(species_vocab)}.")
    if args.ko and args.compound:
        raise SystemExit("Use either --ko or --compound, not both.")

    condition_id = cond_vocab[args.condition] if args.condition else null_condition
    species_id = species_vocab[args.species]
    if args.ko:
        gene = args.ko.upper().strip()
        if gene not in gene_index or gene not in pert_vocab:
            raise SystemExit(
                f"ABSTAIN: {gene} is outside the trained knockout vocabulary. "
                "HepaDiff does not issue calibrated predictions for unseen genes. "
                "Use --check-gene GENE to test support.")
        perturbation_id = pert_vocab[gene]
        gene_id = gene_index[gene]
        perturbation_type = 0  # loss of function
        perturbation_label = f"knockout:{gene}"
    elif args.compound:
        if args.compound not in pert_vocab or args.compound in gene_index:
            raise SystemExit(
                f"ABSTAIN: '{args.compound}' is outside the in-library compound "
                "vocabulary. Run --list-compounds for supported names.")
        perturbation_id = pert_vocab[args.compound]
        gene_id = -1
        perturbation_type = 2
        perturbation_label = f"compound:{args.compound}"
    else:
        perturbation_id = null_perturbation
        gene_id = -1
        perturbation_type = 3
        perturbation_label = "none"

    tokens = decode(
        model,
        cell_type_id=ct_vocab[cell],
        condition_id=condition_id,
        perturbation_id=perturbation_id,
        gene_id=gene_id,
        perturbation_type_id=perturbation_type,
        species_id=species_id,
        n_cells=args.n,
        seed=args.seed,
        batch_size=args.batch,
        steps=args.steps,
        device=device,
    )
    np.savez_compressed(
        f"{args.out}_cells.npz", tokens=tokens, genes=np.array(genes), cell=cell,
        perturbation=perturbation_label, condition=args.condition,
        species=args.species, seed=args.seed, checkpoint_step=checkpoint["step"],
        decoder=f"{args.steps}-step-unbiased")

    edges = np.load(args.edges)
    lut = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)
    mean_expression = lut[tokens.astype(int)].mean(axis=0)
    control = ref[f"Ec__{cell}"]
    report = pd.DataFrame({
        "gene": genes,
        "mean_expr_generated": mean_expression,
        "mean_expr_control": control,
        "delta": mean_expression - control,
    })
    report = report.reindex(report["delta"].abs().sort_values(ascending=False).index)
    report.to_csv(f"{args.out}_delta.csv", index=False)

    print(f"Generated {len(tokens)} cells | checkpoint step={checkpoint['step']} | "
          f"cell={cell} | condition={args.condition or 'null'} | "
          f"species={args.species} | perturbation={perturbation_label}")
    print("Top up-regulated genes:")
    for _, row in report.nlargest(10, "delta").iterrows():
        print(f"  {row['gene']:15s} delta={row['delta']:+.4f}")
    print("Top down-regulated genes:")
    for _, row in report.nsmallest(10, "delta").iterrows():
        print(f"  {row['gene']:15s} delta={row['delta']:+.4f}")
    print(f"Saved {args.out}_cells.npz and {args.out}_delta.csv")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hepadiff", description="HepaDiff virtual-liver inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="generate virtual liver cells")
    infer.add_argument("--ckpt", default=DEFAULT_CKPT)
    infer.add_argument("--reference", default=bundled("hepadiff_reference.npz"))
    infer.add_argument("--edges", default=bundled("liver_bin_edges_v4.npy"))
    infer.add_argument("--ko", help="trained-vocabulary gene to knock out")
    infer.add_argument("--compound", help="in-library compound name")
    infer.add_argument("--condition", default="healthy")
    infer.add_argument("--cell", default="hepatocyte")
    infer.add_argument("--species", default="human", help="human, mouse, rat or cellline")
    infer.add_argument("--n", type=int, default=2000, help="number of cells")
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--batch", type=int, default=250)
    infer.add_argument("--steps", type=int, default=4, help="release default: 4")
    infer.add_argument("--cpu", action="store_true")
    infer.add_argument("--out", default="hepadiff_out")
    infer.add_argument("--list-cells", action="store_true")
    infer.add_argument("--list-conditions", action="store_true")
    infer.add_argument("--list-compounds", action="store_true")
    infer.add_argument("--check-gene", metavar="GENE")
    args = parser.parse_args()
    if args.n < 1 or args.batch < 1:
        parser.error("--n and --batch must be positive")
    run_infer(args)


if __name__ == "__main__":
    main()
