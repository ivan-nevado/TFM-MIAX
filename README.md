# TFM-MIAX: relative valuation with machine learning

Replication of **Geertsema & Lu (2023), "Relative Valuation with Machine Learning"**, *Journal of
Accounting Research* 61(1), on LSEG data for 1990–2019, organised as a monorepo: the ML package
`relval`, a FastAPI backend, a React frontend and Terraform for AWS.

The replication is the base of a master's thesis: a model that estimates a company's valuation
multiple from its financial statements, sector and (later) LLM-based qualitative scores.

## Where things are

| Path | What |
|---|---|
| `ml/` | Package `relval`: data adapters, panel, the 97 inputs, models, evaluation, SHAP, peer weights, reports |
| `backend/` | Read-only API over the replication outputs |
| `frontend/` | Replication dashboard and firm explorer |
| `terraform/` | AWS modules and `dev` / `prod` environments (validated, never applied from here) |
| `docs/replication/` | `REPLICATION_REPORT.md`, `PROGRESS.md`, `DECISIONS.md`, `DEVIATIONS.md`, `BLOCKERS.md`, `paper_targets/` |
| `docs/paper/` | The paper and the team's plan |
| `docs/adr/` | Architecture decision records |
| `data/` | Licensed data and every output. Git-ignored; never commit or upload it |
| `scripts/` | Paper-target extraction, AWS Batch entry point |

Read `docs/replication/PROGRESS.md` first: it says what has been run and what is next.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10. The environment pins LightGBM 3.2.1,
the version of the paper.

```bash
uv venv .venv --python 3.10
uv pip install --python .venv -e "ml[dev]" -e "backend[dev]"
```

Put the LSEG download zips in `data/raw/lseg/` (they are read in place, never modified).

## Run the replication

`make` targets wrap the `relval` CLI. Without `make` (Windows), call the CLI directly:
`.venv/Scripts/relval <stage>` (Linux/macOS: `.venv/bin/relval`).

```bash
relval replicate            # everything from data/raw: panel, models, tables, figures, report
relval status               # which model runs are complete
```

Or stage by stage (each is resumable and skips finished work):

```bash
relval external             # download the public Ken French files
relval ingest               # vendor files -> standardised tables
relval panel                # firm-month panel, 97 inputs, targets, Table 1 filters
relval train --group base   # base GBMs; other groups: benchmarks, recon, robust, sensitivity
relval peers                # peer weights and the validation of the decomposition
relval report               # all tables (CSV + Markdown), figures (PNG), REPLICATION_REPORT.md
```

`relval replicate --config smoke.yaml` runs every stage on six months in a few minutes. It checks
the plumbing; its numbers are not results.

The full run is heavy for a laptop (about 60 model variants over 360 months). It is designed to
run as an AWS Batch array job, one child per year; see `terraform/modules/training_batch`.

## Read the results

- `docs/replication/REPLICATION_REPORT.md`: every table next to the paper's values, the six
  qualitative findings and the working thresholds. Generated; do not edit.
- `reports/replication/tables/`: every table as CSV and Markdown.
- `reports/replication/figures/`: Figures 1–5 as PNG.
- `data/predictions/replication/`: the master predictions table (one row per model, firm, month).

## Backend and frontend

```bash
# API on http://localhost:8000 (docs at /docs)
cd backend && RELVAL_DATA_ROOT=../data RELVAL_REPORTS_ROOT=../reports ../.venv/bin/uvicorn app.main:app --reload

# UI on http://localhost:5173
cd frontend && pnpm install && pnpm dev
```

Or both in containers: `docker compose up --build` (UI on http://localhost:8080).

## Tests and checks

```bash
.venv/bin/python -m pytest ml/tests           # method-fidelity tests, synthetic data only
cd backend && ../.venv/bin/python -m pytest tests
.venv/bin/ruff check ml backend scripts
cd frontend && pnpm build                     # type-check and build
make tf-validate                              # terraform fmt + validate, nothing is applied
```

CI (`.github/workflows/ci.yml`) runs all of these without the data.

## Rules of the project

- The licensed data never leave the machine or the owner's private raw-data bucket.
- No result is typed by hand: every number in a report comes from code that can be rerun.
- The method is fixed by the paper. No seed, filter or hyperparameter is tuned to move a number
  closer to the paper's.
- The test block never influences anything: not early stopping, model selection, filters or
  imputation statistics.
- No cloud resource is created from this repository: Terraform is formatted and validated only.
