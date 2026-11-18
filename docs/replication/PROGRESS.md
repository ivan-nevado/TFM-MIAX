# Progress

Read this first when starting or resuming. Then `relval status` for the exact state of the runs.

## State (2026-11-16)

| Phase | Status |
|---|---|
| 0. Set-up | Done. Pinned environment (LightGBM 3.2.1), originals in `data/raw/lseg`, paper targets extracted and verified against the PDF, Ken French files and the paper's online appendix downloaded |
| 1. Panel | Done (with the `div` fix). 547,642 → 494,076 → 421,392 firm-months (2,716 → 2,523 → 2,220 firms) |
| 2. GBM | Done. Three base models, five seeds, 1990–2019 |
| 3. Benchmarks | Done: IND, BL, RRV, BG. HP: blocker |
| 4. Reconciliation and robustness | Done: Table 4, CART, 80:20, alternative targets, sensitivity runs |
| 5. Portfolios | Done (Table 3, Table IA.7) |
| 6. Explainability | Done. SHAP for the first seed; peer weights for December 2019 validated |
| 7. Report | Done. `REPLICATION_REPORT.md` regenerated locally on 2026-11-16 from the AWS outputs |
| 8. Backend, frontend, Terraform | Done. API tests pass, frontend builds, both Terraform environments validate. `dev` is partly applied (see "Running on AWS") |

Every model of the registry was run on AWS Batch (2026-11-15/16, ≈18 h on one 4-vCPU Spot
instance, no failed child). Outputs are in `s3://relval-dev-artifacts-364694604800/` and were
synced to `data/predictions`, `data/artifacts` and `reports/`.

## Full-period numbers (fixed panel, 2026-11-16)

From `relval report`. Headline seed; paper values in brackets.

| Model | med(\|ε\|) | med(ε) | R²oos(ŷ) | R²oos(M̂) | ρ |
|---|---|---|---|---|---|
| ML_lnm2b | 0.276 (0.29) | −0.003 (−0.00) | 0.53 (0.54) | 0.81 (0.78) | 0.91 (0.90) |
| ML_lnv2a | 0.291 (0.30) | 0.010 (0.01) | 0.76 (0.75) | 0.82 (0.77) | 0.91 (0.89) |
| ML_lnv2s | 0.308 (0.32) | 0.005 (0.01) | 0.79 (0.80) | 0.78 (0.76) | 0.90 (0.89) |

- Seed spread of med(|ε|) across the five seeds is at most 0.0005 (`table2_seed_spread`).
- All 15 working thresholds pass. The GBM is below IND and BL in 99.7–100% of months and below
  BG in every month.
- IND med(|ε|) 0.397 / 0.454 / 0.577 (paper 0.42 / 0.47 / 0.64). BL 0.562 / 0.533 / 0.582
  (paper 0.52 / 0.67 / 0.68). BG 1.018 (paper 0.86).
- Table 4 holds: logs, the GBM and the 97 variables each lower the error, and the GBM beats
  LASSO. GBM on the BL variables: 0.329 / 0.345 / 0.408 (paper 0.33 / 0.34 / 0.39).
- Robustness: CART 0.339 / 0.373 / 0.423 (paper 0.35 / 0.39 / 0.44), 80:20 split 0.282 / 0.317 /
  0.338 (paper 0.30 / 0.34 / 0.36). Sensitivity runs (numeric industry, listed-today firms, beta
  of 24–48 months) move med(|ε|) by at most 0.001.
- Peer weights reproduce every December 2019 prediction (max error 7.9e-09) and sum to one.
- SHAP: Profitability then Growth lead m2b, as in the paper. v2a: Profitability then Solvency
  (paper: Growth second); `debt_capital_wr` is first. v2s: Efficiency leads with
  `opleverage_an` first and rho −0.83 (paper −0.82). Industry ranks far lower than in the paper
  (1.6–2.1 against 5.9–8.9), consistent with 73% industry coverage.
- Long-short portfolios on GBM errors: mean 0.12 / 0.14 / 0.18% per month, positive but not
  significant (paper 0.37 / 0.42 / 0.35%). Equal-weighted: 0.30 / 0.35 / 0.31%, FF6 alpha
  significant (0.35 / 0.35 / 0.27%).
- Finding 1 of the report ("GBM has the lowest error") fails only against RRV (0.288), which
  values 38% of firm-months. On the common sample the GBM wins: 0.287 / 0.288 / 0.301 against
  RRV 0.357.

## Open issue: BL and OLS-on-BL dollar metrics

The `div` fix did not repair the BL dollar metrics. med(|ε|) of BL is sensible, but R²oos and
RMSE are not (BL_m2b R²oos(ŷ) −95, RMSE 91.8 against the paper's 6.47). The cause is a handful of
firm-months with exploding OLS predictions, not the earlier bug: 695 of 197,268 BL_m2b
predictions miss by more than 50, and the worst are TSSI.OQ (2007-10, 2007-11; ŷ ≈ −29,000),
BRO.N (1998-12, 1999-02) and EIX.N (2004-05). Removing the 20 worst errors brings the BL_m2b
RMSE to 10.4. The same firm-months explode in `OLSLOG_BL_*`, so some of the ten BL inputs take
extreme values there. The next step is to inspect those inputs (`adjopmad`, `adjsalesgrowth`,
`d2e`, the industry multiples) for those rows and check the definitions against Table IA.3. Do
not winsorise or drop rows to move the number. RRV keeps its known exploding predictions (see
`DEVIATIONS.md`).

## Next

1. Diagnose the BL extreme predictions (above). `relval visual` (2026-11-18) traces the eight worst to
   inputs thousands of times outside the month's range (`adj_sgr`, `pretret_noa_wr`, `lev`); proposal:
   winsorise the BL inputs at p1-p99 of the fitting sample, cut chosen on validation.
2. When the team wants to show results: apply the load balancer, API and frontend hosting.

## Running on AWS

State of the `dev` environment (account 364694604800, eu-west-1, 2026-11-15):

- Applied: the two ECR repositories and the raw, processed and artifacts buckets
  (`terraform apply -target=module.ecr -target=module.storage`).
- The LSEG files are in `s3://relval-dev-raw-364694604800/raw/lseg/`.
- Images `relval-dev/training:20261116` and `relval-dev/api:20261115` are pushed (the job
  definition uses training `20261116`, whose report stage also uploads
  `REPLICATION_REPORT.md` to `reports/`; the 2026-11-15 run used `20261115`). The
  repositories are immutable: a new build needs a new tag in `terraform.tfvars`.
- Applied in a second targeted apply: network, IAM roles, Batch (Spot), log groups, the
  failed-job e-mail alert and the budget.
- Not applied: load balancer, API service, frontend hosting (the only parts with a fixed cost,
  about 0.85 USD a day). A plain `terraform apply` would create them: agree first.
- Submitted on 2026-11-15 to the queue `relval-dev-training`: `replication-panel`, then the
  `replication-train` array (30 children, one year each, `group=all`), then
  `replication-report`. Each waits for the previous one. Spot interruptions are retried (3
  attempts) and a retried child skips finished years.
  All three finished on 2026-11-16 (30 of 30 train children succeeded, none retried).
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
