# api_service

The FastAPI backend as an ECS Fargate service behind an application load balancer in the public
subnets. The tasks run in `task_subnet_ids`: private subnets, or public subnets with
`assign_public_ip` when the VPC has no interface endpoints. Tasks accept traffic only from the load balancer and read the
replication outputs from S3 with a read-only role. With `certificate_arn` the listener serves
HTTPS and redirects HTTP; without it (development) it serves HTTP.

Outputs: `alb_dns_name`, `alb_arn_suffix`, `cluster_name`, `service_name`.
