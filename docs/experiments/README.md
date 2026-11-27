# Experiments (extensions of the replication)

The replication (`ml/configs/replication.yaml`, run `replication`) is the fixed baseline. Every
extension is an experiment: one YAML file that changes one idea, trained on AWS Batch exactly like
the replication, written under its own run name and compared with the baseline on the same
firm-months.

## Where things are

| Path | What |
|---|---|
| `ml/configs/experiments/_common.yaml` | Settings shared by all experiments (headline seed only, SHAP on, no model strings) |
| `ml/configs/experiments/<id>.yaml` | One experiment: `experiment` block (id, family, title, hypothesis, baseline, models) + the keys it changes |
| `ml/src/relval/models/window.py` | E2: training window and leakage guards |
| `ml/src/relval/models/gbm.py` | E1: the loss is `gbm.params.objective`; row weights for E2 |
| `ml/src/relval/splits/blocks.py` | `firm_blocks`: blocks assigned per firm (E2 guard) |
| `ml/src/relval/experiments/` | `registry.py` (finds and validates experiment configs), `compare.py` (`relval compare`) |
| `ml/tests/test_experiments.py` | Synthetic tests: no leak, defaults equal the paper, loss reaches LightGBM, pairing |
| `scripts/submit_experiments.ps1` | Submits train → report → compare to AWS Batch |
| `data/predictions/experiments/<id>/` | Predictions of an experiment (S3: `predictions/experiments/<id>/`) |
| `reports/experiments/<id>/` | Full report of one experiment (tables, figures, `REPLICATION_REPORT.md`) |
| `reports/experiments/comparison/` | `leaderboard`, `by_decade`, `by_size`, `monthly`: every experiment against its baseline |
| `reports/experiments/HISTORY.md` | Catalogue: hypothesis, every training run (date, commit, image, Batch job), result |
| `reports/experiments/history/<utc>/` | Snapshot of the leaderboard at every `relval compare` (never overwritten) |

Nothing of the replication is overwritten: experiments only write under `experiments/`.

## The experiments

| id | Family | Changes | Compared with |
|---|---|---|---|
| `e1_l1` | E1 robust loss | `objective: regression_l1`, early stopping on L1 | replication |
| `e1_huber` | E1 robust loss | `objective: huber`, delta (`alpha`) 0.5 | replication |
| `e2_w12` | E2 training window | months t-11..t; past rows of test **and validation** firms excluded | replication |
| `e2_w12_aligned` | E2 training window | as `e2_w12`, each past month re-levelled to month t's median | replication |
| `e2_firmblocks_w1` | E2 control | blocks per firm, no window | replication |
| `e2_w12_firmblocks` | E2 training window | window with blocks per firm | `e2_firmblocks_w1` |
| `e2_w12_leaky` | E2 **diagnostic** | window with no guard: measures the leak. Never a result | replication |
| `e12_l1_w12` | E1+E2 | L1 and the window together (run after reading E1 and E2) | replication |

Notes on the method:

- **E1.** Only the loss changes; early stopping uses the same loss on the validation block. Peer
  weights (paper, appendix A.3) assume L2 and are not computed for these runs.
- **E2.** Month t keeps the paper's five blocks; only month t's test block is predicted and early
  stopping uses month t's validation block. Past months add training rows only. A firm in month
  t's test or validation block never contributes past rows (`leakage_guard`), so the model cannot
  recognise the firm it values. `n_train` in the predictions counts every training row; the
  R²oos reference mean stays month t's training mean, so R²oos is comparable with the baseline.
  The first 11 months of 1990 have a shorter window (the panel starts in 1990-01).
- Decisions fixed before seeing any result: Huber delta 0.5, window 12 months, no lag decay.
  Change them only as a new experiment file, never by editing one that has run.

## Reading rule (fixed in advance)

For each experiment and model, `relval compare` pairs the predictions with the baseline's on the
same firm-months (headline seed) and reports the Table 2 metrics of both, the difference of
med(|ε|), the share of months the experiment wins and a bootstrap 95% CI of the mean monthly
difference. **An experiment improves on its baseline when the CI is below zero and the pooled
difference is larger than the baseline's seed spread** (column `improves`). Look also at
`by_decade` (E2 targets the thin 1990s), `by_size`, `med_eps` (bias must stay near zero) and
`rmse_y`/`r2oos_M` (L1 may trade the tail for the median).

## Add an experiment

1. Copy the closest YAML in `ml/configs/experiments/`, give it a new id (file name = `experiment.id`,
   `run_name: experiments/<id>`), write the hypothesis **before** running it.
2. `relval train --config experiments/<id>.yaml --years 2019` locally if you want a quick check
   (it writes under `data/predictions/experiments/<id>/`).
3. `.venv/Scripts/python -m pytest ml/tests` (the test checks every experiment config is valid).
4. Submit it to AWS (below). Never edit a YAML that has already run: create a new one.

## Run on AWS Batch

Same infrastructure and flow as the replication (Spot compute environment, queue
`relval-dev-training`, job definition `relval-dev-pipeline`, one array child per year). The job
definition has a new parameter `config` (default `replication.yaml`), so one image serves every run.

**1. Build and push an image with this code** (repositories are immutable: new tag each time).
In PowerShell, from the repository root, with Docker running:

```powershell
aws sso login --profile relval; $env:AWS_PROFILE = "relval"
$TAG = "20261127"   # new date tag
$REPO = "364694604800.dkr.ecr.eu-west-1.amazonaws.com/relval-dev/training"
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REPO.Split("/")[0]
docker build -f ml/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg IMAGE_TAG=$TAG -t "${REPO}:$TAG" .
docker push "${REPO}:$TAG"
```

Commit the code first, so the commit baked into the image (and written to every manifest) exists.

**2. Point the job definition to the image** (only the Batch module; nothing with a fixed cost):

```powershell
# terraform/environments/dev/terraform.tfvars: training_image_tag = "<TAG>"
cd terraform/environments/dev
terraform plan -target=module.training_batch    # expect: job definition replaced, nothing else
terraform apply -target=module.training_batch
```

**3. Quick check on two years** (a few minutes, no comparison):

```powershell
./scripts/submit_experiments.ps1 -Configs experiments/e2_w12.yaml -FirstYear 2018 -LastYear 2019 -NoCompare
```

**4. Full runs** (`-DryRun` prints the commands without submitting). Suggested order: E1 first
(cheapest, independent), then E2 with its control and diagnostic, then the combination.

```powershell
./scripts/submit_experiments.ps1 -Configs experiments/e1_l1.yaml,experiments/e1_huber.yaml
./scripts/submit_experiments.ps1 -Configs experiments/e2_w12.yaml,experiments/e2_w12_aligned.yaml,experiments/e2_firmblocks_w1.yaml,experiments/e2_w12_firmblocks.yaml,experiments/e2_w12_leaky.yaml
```

Each config gets a `train` array (30 children) and a `report` job; one `compare` job waits for all
reports. `compare` always re-reads everything in S3, so it can be resubmitted alone at any time:

```powershell
aws batch submit-job --job-name exp-compare --job-queue relval-dev-training --job-definition relval-dev-pipeline --parameters stage=compare,group=base,first_year=1990,chunk=1,config=replication.yaml
```

**5. Follow and download.**

```powershell
aws batch list-jobs --job-queue relval-dev-training --filters name=JOB_NAME,values='exp-*' --query 'jobSummaryList[].[jobName,status,arrayProperties.statusSummary]'
aws logs tail /relval-dev/training --follow
aws s3 sync s3://relval-dev-artifacts-364694604800/reports/experiments reports/experiments
aws s3 sync s3://relval-dev-artifacts-364694604800/predictions/experiments data/predictions/experiments
```

Spot interruptions are retried and a retried child skips finished years, as in the replication.

### Time and cost (estimate)

Measured locally (8 workers, `ML_lnm2b`, year 2019): baseline set-up ≈ 50 s, `e2_w12` ≈ 57 s
(~14,000 training rows per model instead of ~1,200), `e1_l1` ≈ 81 s (about 4× more trees than
L2). Earlier years have fewer firms and are faster. With the current Spot quota (one 4-vCPU
instance, children one after another) expect roughly 2–5 hours per experiment for the three base
models, at about 0.10 USD per hour: a few dollars for the whole set, covered by the credits. After
the 32-vCPU quota increase, raise `batch_max_vcpus` and the children run in parallel.
