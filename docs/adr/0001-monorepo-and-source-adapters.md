# ADR 0001: one repository, and a source adapter between vendor files and the pipeline

**Status:** accepted

## Context

The thesis needs a research pipeline (the replication), a service and a UI over its outputs, and
cloud infrastructure. The data are LSEG today; the paper used CRSP/Compustat, which the project
may get later.

## Decision

- One repository with `ml/`, `backend/`, `frontend/` and `terraform/`. The backend imports the ML
  package through a single adapter module (`backend/app/ml/adapter.py`).
- The pipeline never reads vendor files. A source adapter (`relval.data.adapter.SourceAdapter`)
  returns four standardised tables (quarterly and annual fundamentals, market data, statics) in
  fixed units and with Compustat item names. Everything downstream is source-agnostic.
- Vendor-specific conventions (blank means zero, depreciation inside cost of revenue) are config
  switches under `panel.definitions`, not code branches.

## Consequences

- Running the paper's own data needs only the `wrds` adapter and three config changes.
- Universe membership and NYSE status are columns of the market table, so a source with
  historical exchange codes (CRSP) supplies them month by month with no change elsewhere.
- The adapter must reproduce Compustat semantics for every item; each mapping is documented in
  `docs/replication/DECISIONS.md`.
