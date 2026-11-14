# Every target is a thin wrapper over the `relval` CLI, so the same commands work without make
# (for example on Windows: `.venv/Scripts/relval replicate`).
#
#   make setup       create .venv with the pinned environment (LightGBM 3.2.1, NumPy < 2)
#   make replicate   rebuild the panel, every model, every table and figure from data/raw
#   make smoke       fast end-to-end check on six months (not a result)

PYTHON_VERSION ?= 3.10
CONFIG ?= replication.yaml
ifeq ($(OS),Windows_NT)
BIN := .venv/Scripts
else
BIN := .venv/bin
endif
RELVAL := $(BIN)/relval --config $(CONFIG)

.PHONY: setup external ingest panel train benchmarks evaluate shap peers report replicate smoke status \
        paper-targets test lint format backend frontend tf-validate docker clean-outputs

setup:
	uv venv .venv --python $(PYTHON_VERSION)
	uv pip install --python .venv -e "ml[dev]" -e "backend[dev]"

external:
	$(RELVAL) external

ingest:
	$(RELVAL) ingest

panel:
	$(RELVAL) panel

train:              ## base GBMs (all seeds)
	$(RELVAL) train --group base

benchmarks:         ## IND, BL, RRV, BG, HP + reconciliation + robustness + sensitivity runs
	$(RELVAL) train --group benchmarks,recon,robust,sensitivity

peers:              ## peer weights of the configured months, with the validation of the identity
	$(RELVAL) peers

# Metrics, SHAP tables, figures and the report are all produced by the report stage from the
# master predictions table and the stored SHAP values.
evaluate shap report:
	$(RELVAL) report

replicate:
	$(RELVAL) replicate

smoke:
	$(BIN)/relval replicate --config smoke.yaml

status:
	$(RELVAL) status

paper-targets:      ## re-extract the paper's tables and check every number against the PDF
	$(BIN)/python scripts/paper_targets.py extract
	$(BIN)/python scripts/paper_targets.py verify

test:
	$(BIN)/python -m pytest ml/tests
	cd backend && ../$(BIN)/python -m pytest tests

lint:
	$(BIN)/ruff check ml backend scripts
	$(BIN)/ruff format --check ml backend scripts

format:
	$(BIN)/ruff check --fix ml backend scripts
	$(BIN)/ruff format ml backend scripts

backend:            ## API on http://localhost:8000 (reads data/ and reports/)
	cd backend && RELVAL_DATA_ROOT=../data RELVAL_REPORTS_ROOT=../reports ../$(BIN)/uvicorn app.main:app --reload

frontend:           ## UI on http://localhost:5173 (proxies /api to :8000)
	cd frontend && pnpm install && pnpm dev

tf-validate:        ## formatting and validation only: nothing is ever applied from here
	terraform fmt -check -recursive terraform
	cd terraform/environments/dev && terraform init -backend=false -input=false && terraform validate
	cd terraform/environments/prod && terraform init -backend=false -input=false && terraform validate

docker:
	docker build -f ml/Dockerfile -t relval-training .
	docker build -f backend/Dockerfile -t relval-api .
	docker build -t relval-frontend frontend
