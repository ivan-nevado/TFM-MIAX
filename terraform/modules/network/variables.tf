variable "name" {
  description = "Prefix of every resource name (project and environment)."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones (one public and one private subnet in each)."
  type        = number
  default     = 2
}

variable "enable_nat_gateway" {
  description = "Create one NAT gateway for the private subnets. Not needed when the endpoints cover every AWS call."
  type        = bool
  default     = false
}

variable "interface_endpoints" {
  description = "Interface endpoint services (suffix after com.amazonaws.<region>.)."
  type        = list(string)
  default     = ["ecr.api", "ecr.dkr", "logs", "ssm", "secretsmanager", "ecs", "ecs-agent", "ecs-telemetry"]
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
