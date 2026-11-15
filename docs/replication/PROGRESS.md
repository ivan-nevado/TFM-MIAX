# Progress

Read this first when starting or resuming. Then `relval status` for the exact state of the runs.

## State (2026-11-14)

| Phase | Status |
|---|---|
| 0. Set-up | Done. Pinned environment (LightGBM 3.2.1), originals in `data/raw/lseg`, paper targets extracted and verified against the PDF, Ken French files and the paper's online appendix downloaded |
| 1. Panel | Done. 547,642 → 494,076 → 421,392 firm-months (2,716 → 2,523 → 2,220 firms) |
| 2. GBM | Code done. First seed of the three base models run on 1990–2019 (see the caveat below). Seeds 2–5 not run |
| 3. Benchmarks | Code done. IND, BL, RRV, BG run on 1990–2019 (same caveat). HP: blocker |
| 4. Reconciliation and robustness | Code done and smoke-tested. Not run on 1990–2019 |
| 5. Portfolios | Code done; produced by `relval report` |
| 6. Explainability | Code done. SHAP stored with the base run; peer weights for December 2019 validated |
| 7. Report | Code done; `relval report` writes tables, figures and `REPLICATION_REPORT.md` |
| 8. Backend, frontend, Terraform | Done. API tests pass, frontend builds, both Terraform environments validate. `dev` is partly applied (see "Running on AWS") |

Every stage has also been run end to end on the smoke configuration (six months of 2019, all 55
model variants).

## First full-period numbers (first seed, before the fix below)

From `relval report` on 2026-11-14. Paper values in brackets.

| Model | med(\|ε\|) | med(ε) | R²oos(ŷ) | R²oos(M̂) |
|---|---|---|---|---|
| ML_lnm2b | 0.276 (0.29) | −0.003 | 0.53 (0.54) | 0.81 (0.78) |
| ML_lnv2a | 0.291 (0.30) | 0.010 | 0.76 (0.75) | 0.81 (0.77) |
| ML_lnv2s | 0.308 (0.32) | 0.005 | 0.79 (0.80) | 0.78 (0.76) |

- All working thresholds pass. The GBM is below IND and BL in 99.7–100% of months.
- IND median absolute error 0.40 / 0.45 / 0.58 (paper 0.42 / 0.47 / 0.64).
- Peer weights reproduce every December 2019 prediction (max error 8e-09) and sum to one.
- SHAP: Profitability leads m2b and v2a; Efficiency leads v2s with `opleverage_an` first and
  negative rho. For v2a the second category is Solvency, not Growth as in the paper.
- Long-short portfolios on GBM errors: positive mean returns (0.14–0.17% per month), not
  statistically significant on this survivor sample (paper 0.35–0.42%).
- RRV has a lower median error (0.288) than the GBM for v2a and v2s on its own, much smaller
  sample (38% of firm-months, industries with at least 30 firms). Compare on the common sample.

## Caveat: these outputs must be regenerated

A bug was fixed after that run: a lagged four-quarter sum that is zero in the accounts could be a
floating-point residue (1e-16), giving ratios such as sales growth of 2e16. `div` now treats
denominators below 1e-9 as zero. The GBM numbers above will barely move (trees split on
thresholds), but the BL dollar metrics were destroyed by it (R² of −1e27) and are meaningless
in that run. The panel has been rebuilt with the fix; predictions on disk have not.

RRV also shows a handful of exploding predictions (11 firm-months). That is not this bug: it is
extrapolation of a five-variable regression fitted on as few as 30 firms. It is kept as the method
gives it and is documented in `DEVIATIONS.md`.

## Next

1. Rerun with the fixed panel (locally or on AWS Batch; `--force` overwrites the stale years):
   `relval train --group base,benchmarks --force`
2. `relval train --group recon,robust,sensitivity` and seeds 2–5 (`relval train --group base`).
3. `relval peers && relval report`, then replace the table above with the new numbers.

## Running on AWS

State of the `dev` environment (account 364694604800, eu-west-1, 2026-11-15):

- Applied: the two ECR repositories and the raw, processed and artifacts buckets
  (`terraform apply -target=module.ecr -target=module.storage`).
- The LSEG files are in `s3://relval-dev-raw-364694604800/raw/lseg/`.
- Images `relval-dev/training:20261115` and `relval-dev/api:20261115` are pushed. The
  repositories are immutable: a new build needs a new tag in `terraform.tfvars`.
- Applied in a second targeted apply: network, IAM roles, Batch (Spot), log groups, the
  failed-job e-mail alert and the budget.
- Not applied: load balancer, API service, frontend hosting (the only parts with a fixed cost,
  about 0.85 USD a day). A plain `terraform apply` would create them: agree first.
- Submitted on 2026-11-15 to the queue `relval-dev-training`: `replication-panel`, then the
  `replication-train` array (30 children, one year each, `group=all`), then
  `replication-report`. Each waits for the previous one. Spot interruptions are retried (3
  attempts) and a retried child skips finished years.
- A Spot quota increase to 32 vCPUs was requested on 2026-11-15.
- Terraform state: `s3://relval-tfstate-364694604800/relval/dev/terraform.tfstate`.

`dev` has no interface VPC endpoints and no NAT gateway: the API tasks and the Batch instances
run in the public subnets with a public IP, and their security groups accept nothing from the
internet. The API service defaults to 0 tasks. The account's Spot quota is 5 vCPUs, so
`terraform.tfvars` sizes Batch for one 4-vCPU instance and the array children run one after
another; raise `batch_max_vcpus` after a quota increase.

Local timing test (2026-11-15): the three base models, first seed, 2019 only, 8 workers:
142 seconds. Median absolute error 0.295 / 0.307 / 0.328.

Progress of the jobs:

```bash
aws batch list-jobs --job-queue relval-dev-training --filters name=JOB_NAME,values='replication-*' \
  --query 'jobSummaryList[].[jobName,status,arrayProperties.statusSummary]' --profile relval
aws logs tail /relval-dev/training --follow --profile relval
```

Outputs land in `s3://relval-dev-artifacts-364694604800/` (`predictions/`, `artifacts/`,
`reports/`). See `terraform/modules/training_batch/README.md` and `scripts/batch_entrypoint.sh`.

## Commands

```bash
.venv/Scripts/relval status                 # state of every (model, seed)
.venv/Scripts/relval replicate              # everything, resumable
.venv/Scripts/python -m pytest ml/tests     # method-fidelity tests
```
