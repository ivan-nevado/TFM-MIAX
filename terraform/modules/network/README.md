# network

VPC with public and private subnets in two availability zones, a gateway endpoint for S3 and
interface endpoints for ECR, CloudWatch Logs, SSM, Secrets Manager and ECS. The training jobs and
the API run in the private subnets and reach AWS services through the endpoints, so the NAT
gateway is optional (`enable_nat_gateway`, off by default to avoid its fixed cost).

Outputs: `vpc_id`, `vpc_cidr`, `public_subnet_ids`, `private_subnet_ids`.
