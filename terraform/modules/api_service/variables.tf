variable "name" {
  description = "Prefix of every resource name (project and environment)."
  type        = string
}

variable "region" {
  description = "AWS region (for the log configuration)."
  type        = string
}

variable "vpc_id" {
  description = "VPC of the service."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnets of the load balancer."
  type        = list(string)
}

variable "task_subnet_ids" {
  description = "Subnets of the tasks: private ones, or public ones together with assign_public_ip."
  type        = list(string)
}

variable "assign_public_ip" {
  description = "Give the tasks a public IP. Needed in public subnets without interface endpoints; inbound stays limited to the load balancer."
  type        = bool
  default     = false
}

variable "image" {
  description = "API image, for example <repository-url>:<tag>."
  type        = string
}

variable "execution_role_arn" {
  description = "Task execution role."
  type        = string
}

variable "task_role_arn" {
  description = "Role assumed by the API container (read-only access to the outputs)."
  type        = string
}

variable "artifacts_bucket" {
  description = "Bucket with the replication outputs (processed/, predictions/, artifacts/, reports/)."
  type        = string
}

variable "log_group_name" {
  description = "CloudWatch log group of the API."
  type        = string
}

variable "run_name" {
  description = "Replication run served by the API."
  type        = string
  default     = "replication"
}

variable "container_port" {
  description = "Port of the API container."
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "Fargate CPU units of one task."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate memory of one task, in MiB."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of tasks."
  type        = number
  default     = 1
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Null serves HTTP only (development)."
  type        = string
  default     = null
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to reach the load balancer."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "container_insights" {
  description = "Enable ECS Container Insights."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
