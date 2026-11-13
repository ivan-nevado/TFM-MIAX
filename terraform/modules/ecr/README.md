# ecr

Two image repositories, `training` (the `relval` pipeline, `ml/Dockerfile`) and `api`
(`backend/Dockerfile`). Tags are immutable, images are scanned on push and only the most recent
`keep_images` are kept.

Outputs: `repository_urls`, `repository_arns` (maps keyed by repository name).
