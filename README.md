# HepaDiff — a perturbable virtual liver

HepaDiff is a perturbation-conditioned masked discrete-diffusion model for mechanism-oriented hepatology research. The current release combines:

- a **69.2-million-parameter virtual-liver core**, trained on 18.48 million human and mouse liver cells from 482 public datasets and 1.75 million perturbation cells;
- a **DILI toxicity interface** using ECFP4 or frozen ChemBERTa-2 representations with Murcko-scaffold-held-out evaluation;
- a **chemical-spectrum interface** that maps molecular structures to 978 LINCS L1000 landmark-gene responses for disease-axis ranking and sparse programme interpretation; and
- a **grounded conversational agent** with vocabulary checks, curated database tools and explicit abstention.

The manuscript-selected core checkpoint is **step 155000**. Release generation uses **four-step, unbiased-logit decoding**.

> HepaDiff is a research tool for hypothesis generation and mechanistic triage. It does not provide clinical diagnosis, treatment recommendations or clinical DILI severity prediction.

## What is included

- `hepadiff/`: installable inference package and model definition
- `training/`: core training and reduced-step donor-hold-out control
- `chemical_interfaces/`: DILI, LINCS/TG-GATEs spectrum, disease-axis screening and sparse knockout-basis analyses
- `evaluation/`: perturbation, calibrated-control, decoder, real-versus-real and donor-hold-out evaluations
- `agent/`: tool server, LiverBench v2 runner and the 343-question benchmark
- `hepadiff/data/`: 22-cell-type healthy reference atlas and expression-bin edges

Large expression matrices, raw public datasets, model checkpoints, chemical-model weights, run logs and API credentials are **not included**.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/mapeng123/hepadiff.git
cd hepadiff
python -m pip install -e .
```

For research-script dependencies:

```bash
python -m pip install -r requirements.txt
```

A conda specification is also available:

```bash
conda env create -f environment.yml
conda activate hepadiff
python -m pip install -e .
```

## Checkpoint

The selected checkpoint is named `hepadiff_step155000.pt` in the commands below. **The checkpoint is not currently distributed through this repository.** Place a locally available copy in the working directory, or pass `--ckpt /path/to/checkpoint.pt`; the `HEPADIFF_CKPT` environment variable is also supported.

The previous step-200000 checkpoint and one-step decoder described by older releases are not the configuration used for the current manuscript.

## Quick start

List supported reference cell types (no checkpoint required):

```bash
hepadiff infer --list-cells
```

Inspect model vocabularies and check whether a knockout is supported:

```bash
hepadiff infer --ckpt hepadiff_step155000.pt --list-conditions
hepadiff infer --ckpt hepadiff_step155000.pt --list-compounds
hepadiff infer --ckpt hepadiff_step155000.pt --check-gene PNPLA3
```

Generate cells under a disease condition:

```bash
hepadiff infer \
  --ckpt hepadiff_step155000.pt \
  --cell "hepatic stellate cell" \
  --condition MASH --species human \
  --n 2000 --seed 0 --out mash_hsc
```

Run a trained-vocabulary knockout:

```bash
hepadiff infer \
  --ckpt hepadiff_step155000.pt \
  --cell hepatocyte --condition healthy --species human \
  --ko PNPLA3 --n 2000 --seed 0 --out pnpla3_ko
```

Run an in-library compound perturbation:

```bash
hepadiff infer \
  --ckpt hepadiff_step155000.pt \
  --cell hepatocyte --condition healthy --species cellline \
  --compound "(+)-JQ1" --n 2000 --out jq1_hep
```

Outputs:

- `<out>_cells.npz`: token matrix, gene panel, condition, species, perturbation, seed, checkpoint step and decoder metadata;
- `<out>_delta.csv`: generated mean expression and change relative to the matching healthy reference, sorted by absolute change.

A GPU is recommended for large generation runs. CPU inference is supported with `--cpu`, but is substantially slower.

## Capability boundaries

- Knockout generation is released **only for genes in the trained perturbation vocabulary**. Out-of-vocabulary requests return an explicit abstention; unrestricted zero-shot knockout is not claimed.
- The core training vocabulary contains 10,949 perturbations, including 8,870 panel genes and 210 compounds. These are not uniformly validated across every biological context.
- Chemical-spectrum projections and knockout-programme decompositions are research hypotheses, not proof of efficacy or mechanism.
- DILI toxicity classification, transcriptional mechanism generation and clinical severity prediction are separate tasks; performance on one must not be inferred from another.
- Generated populations support programme-level analysis but do not fully recover every rare substate.
- Publication-grade stochastic analyses should use repeated sampling; the manuscript generally used ten seeds.

## Reproducing research analyses

The scripts under `training/`, `chemical_interfaces/` and `evaluation/` are the research implementations used for the current analyses rather than a one-command pipeline. They require the corresponding public datasets and preprocessed caches. Their original compute layout uses `/root/autodl-tmp`; set or edit the path constants at the top of each script for a new environment. No private data or server credentials are required by the code.

The core training run used batch size 96, AdamW with learning rate `2e-4`, OneCycle scheduling and 200,000 steps. Eleven plateau-region checkpoints were compared, and step 155000 was selected using the prespecified generation criteria. `training/train_donor_holdout.py` is the separate 60,000-step donor-level control and must not be confused with the main training script.

## Conversational agent and LiverBench v2

See [`agent/README.md`](agent/README.md) for deployment. The benchmark contains **343 independently sourced questions** across 12 categories, including 40 out-of-vocabulary requests for which abstention is required. API keys are read from environment variables and are never stored in this repository.

## Citation

Ma P, Guo Q, Sun Y, Ou Y. *HepaDiff: a conversational virtual liver for mechanism-driven hepatology and drug-safety triage.* Manuscript under review.

## License

MIT; see [LICENSE](LICENSE). Public source datasets and third-party resources retain their original licences and citation requirements.
