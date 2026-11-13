# training_batch

AWS Batch on Spot instances: a managed compute environment that scales from zero, a job queue and
the job definition of the `relval` pipeline. Spot interruptions are retried, and the pipeline
skips years that already have a predictions file, so a retried child resumes where it stopped.

The pipeline runs as an array job, one child per chunk of years. Example (30 children, one year
each; submit only after the owner has uploaded the raw data and pushed the image):

```bash
aws batch submit-job --job-name replication-train \
  --job-queue <job_queue_name> --job-definition <job_definition_name> \
  --array-properties size=30 \
  --parameters stage=train,group=all,first_year=1990,chunk=1
```

Then one job with `stage=report` builds the peer weights, tables, figures and the report.

Outputs: `job_queue_arn`, `job_queue_name`, `job_definition_arn`, `job_definition_name`.
