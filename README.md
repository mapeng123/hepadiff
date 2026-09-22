# HepaDiff — virtual liver tissue

HepaDiff is a perturbation-conditioned masked discrete diffusion model trained on
19.6 million single-cell liver transcriptomes (388 GEO series + 95 CELLxGENE Census
blocks) plus 1.76 M perturbation cells (Replogle-gwps, sci-Plex3).
It generates realistic single-cell expression profiles conditioned on cell type,
disease condition, gene knockout (zero-shot), or in-library compound treatment.

## Installation

```bash
pip install git+https://github.com/mapeng123/hepadiff.git
# or, after PyPI release:
pip install hepadiff
```

A conda environment specification (`environment.yml`) is also provided:

```bash
conda env create -f environment.yml
conda activate hepadiff
pip install -e .
```

## Model checkpoint

The trained checkpoint is too large for git hosting and is distributed separately:

| File | Size | Where to get |
|---|---|---|
| `hepadiff_step200000.pt` | ~276 MB | [Zenodo link — upon publication] |

Download it and place it in your working directory (the CLI looks for
`./hepadiff_step200000.pt` by default), or pass `--ckpt <path>`.

The healthy-reference atlas (`hepadiff_reference.npz`) and expression bin edges
(`liver_bin_edges_v4.npy`) are bundled with the package — no separate download needed.

## Quick start

List the 8 cell types with a healthy reference (hepatocyte, three zoned hepatocyte
subpopulations, hepatic stellate cell, Kupffer cell, cholangiocyte, and liver sinusoidal
endothelial cell) and the 210 in-library compounds:

```bash
hepadiff infer --list-cells
hepadiff infer --list-compounds        # requires the checkpoint
```

Zero-shot gene knockout (e.g. CYP2E1 in centrilobular hepatocytes):

```bash
hepadiff infer --ko CYP2E1 --cell centrilobular-hepatocyte --n 2000 --out cyp2e1_ko
```

In-library compound perturbation:

```bash
hepadiff infer --compound "(+)-JQ1" --cell hepatocyte --n 2000 --out jq1_hep
```

Disease-condition generation (26 conditions, e.g. NASH_MASH, cirrhosis, DILI_tox):

```bash
hepadiff infer --condition NASH_MASH --cell "hepatic stellate cell" --n 2000 --out nash_hsc
```

The same functionality is available as a plain script without installation —
see `python -m hepadiff.cli infer --help` from a source checkout.

## Output

- `<out>_cells.npz` — generated cells as a token matrix (`tokens`, shape n_cells × n_genes),
  with `genes`, `cell`, `perturbation`, `condition` metadata. Bin tokens can be mapped back
  to log-normalized expression via the bundled bin edges.
- `<out>_delta.csv` — per-gene mean expression of generated cells vs the healthy control
  reference of the same cell type, sorted by |delta|. Top up-/down-regulated genes are
  printed to stdout.

## Notes & scope

- Gene knockout is **zero-shot**: any gene in the panel can be knocked out via the
  gene-identity embedding, including genes never perturbed during training.
- Compound perturbation is available for the 210 compounds present in the training
  perturbation library. For a compound outside the library, use `--ko <target gene>`
  as a target-based proxy.
- Generation of 2,000 cells takes ~1 minute on a single GPU (also runs on CPU, slower).

## Citation

[Manuscript under review — citation placeholder]
