# HepaDiff — a perturbable virtual liver

HepaDiff is a perturbation-conditioned masked discrete-diffusion system for mechanism-oriented hepatology research. Release v1.0.0 combines:

- a **69.2-million-parameter virtual-liver core**, trained on 18.48 million human and mouse liver cells and 1.75 million perturbation cells;
- **DILI toxicity classifiers** using 2,048-bit ECFP4 fingerprints;
- a **TG-GATEs spectrum interface** mapping chemical structure and dose to 16,000-gene responses;
- a **LINCS spectrum interface** mapping chemical structure to 978 L1000 landmark-gene responses;
- an **external PPI/GenePT zero-shot KO module** mapped to the step-155000 liver reference state; and
- a **grounded conversational agent** with vocabulary checks, curated tools and explicit abstention.

The manuscript-selected core checkpoint is **step 155000**. Core generation uses **four-step, unbiased-logit decoding**.

> HepaDiff is a research tool for hypothesis generation and mechanistic triage. It does not provide clinical diagnosis, treatment recommendations or clinical DILI severity prediction.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/mapeng123/hepadiff.git
cd hepadiff
python -m pip install -e .
```

Install the additional chemical-interface dependencies when needed:

```bash
python -m pip install -e '.[chemical]'
```

The LINCS and TG-GATEs interfaces download the frozen public encoder `DeepChem/ChemBERTa-77M-MTR` at pinned revision `66b895cab8adebea0cb59a8effa66b2020f204ca`. Use `--encoder-path` to supply a local copy instead.

## Model downloads

All production weights and external zero-shot resources are distributed through [GitHub Release v1.0.0](https://github.com/mapeng123/hepadiff/releases/tag/v1.0.0). The core checkpoint is unchanged from v0.2.0.

| Asset | Purpose | Size (bytes) | SHA256 |
|---|---|---:|---|
| `hepadiff_step155000.pt` | Virtual-liver core | 277,136,013 | `d2c9d62538361f42213fae93861a1d50284343a2005c02dff6578a71fe33e1ef` |
| `dili_all_ecfp4_final.pt` | Combined DILI concern versus no concern | 4,743,491 | `51048d2b20e6c87dd65784739efaa89d74e92e6edc571e136534fd11e945ba4d` |
| `dili_vmost_ecfp4_final.pt` | vMost versus no-DILI-concern | 4,743,583 | `65c77aac56910c3d096cabec27474e6cff0cf6e0a5cd7652a5790a31fdd66e9b` |
| `tggates_spectrum_chemberta_final.pt` | Dose-resolved 16,000-gene spectrum | 71,640,107 | `b010f3baaf3a9d40ac5a81366c1d27d1e8fca9ac9291c119fbbd143b65b1a638` |
| `lincs_model_final_alldata.pt` | 978-gene LINCS spectrum | 9,792,117 | `a783f6c5f35e011a30dd264d2224e5189f778237ce7ce88be9b606961534098e` |
| `hepadiff_zero_shot_assets.zip` | PPI/GenePT response profiles and support | 926,153,066 | `6d0dfdc7f75a6a569b45f71dddf7f43f9551376dc2b507004da13a6dd7d2bdb1` |

Example download:

```bash
curl -L -O https://github.com/mapeng123/hepadiff/releases/download/v1.0.0/hepadiff_step155000.pt
curl -L -O https://github.com/mapeng123/hepadiff/releases/download/v1.0.0/dili_vmost_ecfp4_final.pt
curl -L -O https://github.com/mapeng123/hepadiff/releases/download/v1.0.0/tggates_spectrum_chemberta_final.pt
curl -L -O https://github.com/mapeng123/hepadiff/releases/download/v1.0.0/lincs_model_final_alldata.pt
curl -L -O https://github.com/mapeng123/hepadiff/releases/download/v1.0.0/hepadiff_zero_shot_assets.zip
```

Verify checksums with `sha256sum FILE` on Linux/macOS or `Get-FileHash .\FILE -Algorithm SHA256` on Windows PowerShell.

The DILI and TG-GATEs files labelled `final` are **all-data deployment refits** trained after the manuscript analyses. They supply inference weights only; all reported performance estimates remain based on the original held-out cross-validation. The LINCS `final_alldata` file is the original all-data production refit. Three held-out LINCS seed weights are also included in the release for reproducibility; use `final_alldata` for new compounds.

## Core inference

List supported reference cell types (no checkpoint required):

```bash
hepadiff infer --list-cells
```

Inspect model vocabularies:

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

Outputs are `<out>_cells.npz` and `<out>_delta.csv`. A GPU is recommended for large generation runs; CPU inference is available with `--cpu`.

## Chemical interfaces

DILI-liability classification from a SMILES string:

```bash
hepadiff chemical --interface dili \
  --weights dili_vmost_ecfp4_final.pt \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --out aspirin_dili.json
```

Dose-resolved 16,000-gene TG-GATEs spectrum:

```bash
hepadiff chemical --interface tggates \
  --weights tggates_spectrum_chemberta_final.pt \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' --dose high \
  --out aspirin_tggates
```

LINCS 978-landmark-gene spectrum:

```bash
hepadiff chemical --interface lincs \
  --weights lincs_model_final_alldata.pt \
  --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \
  --out aspirin_lincs
```

Spectrum outputs contain a complete gene-level CSV and a metadata JSON. DILI scores are research-triage outputs, not calibrated clinical probabilities.

## External zero-shot KO module

Unzip `hepadiff_zero_shot_assets.zip` into a directory containing `manifest.json`, `profiles.npy` and `support.npy`. The archive is approximately 926 MB compressed and 1.02 GB uncompressed.

Check routing and support:

```bash
hepadiff zero-shot \
  --assets hepadiff_zero_shot_assets \
  --ckpt hepadiff_step155000.pt \
  --gene ABCB11 --check-only
```

Predict and map an externally supported response:

```bash
hepadiff zero-shot \
  --assets hepadiff_zero_shot_assets \
  --ckpt hepadiff_step155000.pt \
  --gene ABCB11 --cell hepatocyte --out abcb11_external
```

The module provides neighbour-supported predictions for 7,050 of 7,130 panel genes outside the native perturbation vocabulary. It maps a pooled external response to the selected liver reference state; cell type supplies baseline context, not a separately learned cell-type-specific KO interaction. Native trained targets are routed to `hepadiff infer --ko`, and unsupported targets abstain.

## Repository contents

- `hepadiff/`: installable core, chemical-interface and external zero-shot inference
- `training/`: core training, donor-hold-out control and deployment-refit scripts
- `chemical_interfaces/`: original research training and evaluation implementations
- `evaluation/`: perturbation, calibrated-control, decoder and donor-hold-out evaluations
- `agent/`: tool server, LiverBench v2 runner and the 343-question benchmark
- `hepadiff/data/`: 22-cell-type healthy reference atlas, expression-bin edges and LINCS gene list

Large raw public datasets and expression matrices are not stored in Git. Research scripts retain their original compute layout and may require path changes and public-data preprocessing.

## Capability boundaries

- Native core knockout generation is intended for genes in the trained perturbation vocabulary; external zero-shot predictions have limited effect sizes and require neighbour support.
- Chemical-spectrum projections and disease-axis rankings are research hypotheses, not proof of efficacy or mechanism.
- DILI liability, transcriptional response generation and clinical severity are separate tasks.
- Generated populations support programme-level analysis but do not fully recover every rare substate.
- Publication-grade stochastic analyses should use repeated sampling; the manuscript generally used ten generation seeds. Seed variability is not independent biological replication.

## Conversational agent and LiverBench v2

See [`agent/README.md`](agent/README.md). The released LiverBench v2 contains **343 independently sourced questions across 13 categories**. API keys are read from environment variables and are never stored in this repository.

## Citation

Ma P, Guo Q, Sun Y, Ou Y. *HepaDiff: a conversational virtual liver for mechanism-driven hepatology and drug-safety triage.* Manuscript under review.

## License

MIT; see [LICENSE](LICENSE). Public source datasets, GenePT/PPI resources, ChemBERTa and other third-party resources retain their original licences and citation requirements.
