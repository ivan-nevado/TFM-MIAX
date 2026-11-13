# iam

Least-privilege roles:

| Role | Can do |
|---|---|
| training job | read the raw bucket; read and write processed data and artifacts |
| API task | read processed data and artifacts (never the raw, licensed data) |
| task execution | pull images, write logs, read the environment's parameters and listed secrets |
| Batch instance | the managed ECS container-instance policy |
| CI (optional) | push images to the two repositories from `main` of one GitHub repository (OIDC) |

Outputs: `training_job_role_arn`, `api_task_role_arn`, `execution_role_arn`,
`batch_instance_profile_arn`, `ci_role_arn`.
