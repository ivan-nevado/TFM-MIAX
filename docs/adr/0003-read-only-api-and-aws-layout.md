# ADR 0003: a read-only API over files, Batch for training, Fargate for serving

**Status:** accepted

## Context

The first product need is to show the replication: tables next to the paper's, valuations, SHAP
explanations and peers of a firm. Later use cases (private companies, what-if scenarios) will
need trained models at request time. Training is a batch workload of many small models.

## Decision

- **Backend:** FastAPI, read-only. Routers hold no logic, services hold no I/O details, and one
  repository class hides whether outputs are on local disk or S3. No database: the outputs are
  parquet and CSV files written by the pipeline.
- **Scenarios:** `POST /api/v1/scenarios` returns 501 with the planned contract, so clients can
  integrate before the use case exists.
- **Training:** AWS Batch on Spot instances, submitted as an array job with one child per chunk
  of years. Children synchronise inputs from S3, run `relval train`, and write their year
  partitions back.
- **Serving:** ECS Fargate behind an application load balancer; the frontend is static files on
  S3 behind CloudFront, which also forwards `/api/*` so the browser uses one origin.
- **Data boundaries:** three buckets. The API role cannot read the raw (licensed) bucket.
- **Terraform:** small modules composed by `dev` and `prod`. Remote state is configured with a
  partial backend block, so no account detail is in the code. Nothing is applied from this
  repository.

## Consequences

- The API is stateless and cheap; scaling is a task-count change.
- Peer weights are served only for months the pipeline stored; serving any month on demand will
  need the model-loading path through `backend/app/ml/adapter.py`.
- VPC endpoints replace a NAT gateway, trading a small fixed cost per endpoint for no NAT cost.
