# network

VPC with public and private subnets in two availability zones and a gateway endpoint for S3
(free). The NAT gateway is optional (`enable_nat_gateway`, off by default). Workloads reach ECR,
CloudWatch Logs and ECS in one of two ways:

- **Private subnets and interface endpoints** (`interface_endpoints`, the default list). Each
  endpoint is billed per hour and per availability zone, about 4 USD a day for the eight.
- **Public subnets, no interface endpoints** (`interface_endpoints = []`,
  `map_public_ip_on_launch = true`). No fixed cost. Tasks and instances have a public IP for
  outbound calls; their security groups accept nothing from the internet.

Outputs: `vpc_id`, `vpc_cidr`, `public_subnet_ids`, `private_subnet_ids`.
