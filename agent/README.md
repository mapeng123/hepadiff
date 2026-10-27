# HepaDiff grounded agent and LiverBench v2

This directory contains the REST tool server, evaluation runner and 343-question LiverBench v2 benchmark. The agent uses the same step-155000 core model and four-step unbiased decoder as the command-line release.

## Capability boundary

`knockout` first checks the trained perturbation vocabulary. A gene outside that vocabulary returns `status=ABSTAIN`; the server does not silently substitute a zero-shot prediction. Compound mechanism calls use annotated, trained target genes and are explicitly labelled as on-target proxies. Curated lookup results are kept distinct from model-generated evidence.

The server provides mechanism-oriented research outputs, not clinical severity grades or treatment recommendations.

## Tools

| Tool | Purpose |
|---|---|
| `server_info` | Checkpoint, decoder, corpus scale and capability boundaries |
| `list_cell_types` | 22 cell types with a healthy reference |
| `list_conditions` | Model condition vocabulary |
| `check_gene` | Panel membership and trained-knockout support |
| `knockout` | Trained-gene knockout with optional multi-seed dispersion |
| `generate_condition` | Disease/condition-conditioned generation |
| `compound_perturbation` | On-target mechanism via annotated trained targets |
| `query_drug_target` | ChEMBL/target-table annotation lookup |
| `query_drug_library` | DILIrank v2 lookup |
| `query_ligand_receptor` | CellPhoneDB v5 interaction lookup |

## Start the server

Install the package and dependencies from the repository root, then provide the checkpoint and external annotation resources:

```bash
export HEPADIFF_CKPT=/path/to/hepadiff_step155000.pt
export HEPADIFF_ROOT=/path/to/hepadiff_resources
python agent/server.py --host 127.0.0.1 --port 6010
```

The reference atlas and expression-bin edges are bundled. Their paths can be overridden with `HEPADIFF_REF` and `HEPADIFF_EDGES`. Generated reports are written to `hepadiff_agent_runs/` by default; override with `HEPADIFF_AGENT_OUT`.

Expected external resources under `HEPADIFF_ROOT` (or the corresponding individual environment variables):

```text
hallmark.gmt
dilirank_val/chembl_mechanisms.json
dilirank_val/dilirank_targets.csv
dilirank_val/dilirank2.csv
liverbench_v2/lr_pairs_cpdb_v5.csv
```

These third-party annotation tables are not redistributed in this repository. The server requires DILIrank and CellPhoneDB tables at startup; Hallmark and ChEMBL enrichment/annotation outputs are optional when their files are absent.

Example REST call:

```bash
curl -X POST http://127.0.0.1:6010/tools/check_gene \
  -H 'Content-Type: application/json' \
  -d '{"gene":"PNPLA3"}'
```

The service exposes `GET /`, `POST /tools/{name}` and FastAPI OpenAPI documentation at `/docs`.

## LiverBench v2

- `liverbench_v2.json`: 343 questions with independently sourced gold answers and source annotations
- `LiverBench_v2_bank.xlsx`: spreadsheet version of the benchmark
- `bench_runner.py`: grounded-agent and bare-LLM runs, independent judging, coverage@k and McNemar statistics

Set credentials through the environment; never place API keys in the JSON benchmark or source files:

```bash
export GLM_KEY='your-key'
export GLM_MODEL='glm-4.6'
export HEPADIFF_SERVER='http://127.0.0.1:6010'
python agent/bench_runner.py agent
python agent/bench_runner.py bare
python agent/bench_runner.py judge
python agent/bench_runner.py report
```

`GLM_API_URL`, `GLM_JUDGE_MODEL` and `LIVERBENCH_OUT` are optional overrides. Run records contain question IDs, answers, tool traces and judge rationales; they are excluded from git by the repository `.gitignore`.
