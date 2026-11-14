# ADR 0002: every analysis reads one master predictions table

**Status:** accepted

## Context

The paper computes every result from the out-of-sample predictions of its models. Metrics,
figures, portfolios, the reconciliation and the comparisons between models must use exactly the
same predictions, and about 60 model variants are trained month by month.

## Decision

- Every model, GBM or benchmark, writes the same schema: `model, target, id, date, block, seed,
  y, yhat, M, Mhat, eps, n_train, train_mean_y, train_mean_M, best_iteration`.
- Storage is parquet partitioned as `data/predictions/<run>/model=<m>/seed=<s>/year=<y>.parquet`.
  A year file is written last and atomically, so its presence means that year is finished.
- Training is a pure function of (model spec, month sample, seed). Blocks are derived from the
  seed and the month, never stored, so every model sees identical blocks.
- Raw SHAP values and model text are written by the same pass, for the first seed only.

## Consequences

- Any stage can be rerun or resumed; an interrupted AWS Batch child redoes at most one year.
- Array jobs can split the work by year with no coordination.
- Reports are cheap to regenerate and can never mix predictions from different runs.
- Explainability stages rebuild a model deterministically instead of depending on stored models.
