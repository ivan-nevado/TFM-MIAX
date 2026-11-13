# frontend_hosting

The built React application in a private S3 bucket, served by CloudFront through origin access
control. When `api_origin_domain` is set, `/api/*` and `/health` are forwarded uncached to the
API load balancer, so the browser uses a single origin and the frontend needs no API URL.

Deploy (by the owner): `aws s3 sync frontend/dist s3://<bucket_name> --delete`, then invalidate
the distribution.

Outputs: `bucket_name`, `distribution_id`, `domain_name`.
