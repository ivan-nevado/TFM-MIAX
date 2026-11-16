variable "project" {
  description = "Project name, used in resource names and in the Project tag."
  type        = string
  default     = "relval"
}

variable "environment" {
  description = "Environment name."
  type        = string
  default     = "dev"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "eu-west-1"
}

variable "extra_tags" {
  description = "Additional tags for every resource."
  type        = map(string)
  default     = {}
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Create a NAT gateway. Not needed with interface endpoints or with workloads in the public subnets."
  type        = bool
  default     = false
}

variable "interface_endpoints" {
  description = "Interface VPC endpoints (billed per hour and availability zone). Empty needs workloads_in_public_subnets."
  type        = list(string)
  default     = []
}

variable "workloads_in_public_subnets" {
  description = "Run the API tasks and the training instances in the public subnets with a public IP. Their security groups accept no traffic from the internet."
  type        = bool
  default     = true
}

variable "artifacts_ia_days" {
  description = "Days after which artifacts move to Standard-IA (0 disables)."
  type        = number
  default     = 60
}

variable "force_destroy_buckets" {
  description = "Allow Terraform to delete non-empty buckets."
  type        = bool
  default     = true
}

variable "training_image_tag" {
  description = "Tag of the training image."
  type        = string
  default     = "latest"
}

variable "api_image_tag" {
  description = "Tag of the API image."
  type        = string
  default     = "latest"
}

variable "batch_max_vcpus" {
  description = "Upper bound of the Spot compute environment."
  type        = number
  default     = 32
}

variable "batch_instance_types" {
  description = "Instance types or families of the Spot compute environment. One instance must hold a job (batch_job_vcpus, batch_job_memory_mib) and fit in the account's Spot vCPU quota."
  type        = list(string)
  default     = ["c6i", "c6a", "m6i"]
}

variable "batch_job_vcpus" {
  description = "vCPUs of one array child."
  type        = number
  default     = 8
}

variable "batch_job_memory_mib" {
  description = "Memory of one array child, in MiB."
  type        = number
  default     = 30000
}

variable "api_cpu" {
  description = "Fargate CPU units of one API task."
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "Fargate memory of one API task, in MiB."
  type        = number
  default     = 2048
}

variable "api_desired_count" {
  description = "Number of API tasks. 0 keeps the service defined but stopped; raise it for a demo."
  type        = number
  default     = 0
}

variable "api_certificate_arn" {
  description = "ACM certificate of the API load balancer. Null serves HTTP behind CloudFront."
  type        = string
  default     = null
}

variable "container_insights" {
  description = "Enable ECS Container Insights."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Retention of the CloudWatch log groups."
  type        = number
  default     = 14
}

variable "alarm_emails" {
  description = "E-mail addresses that receive alarms and budget notifications."
  type        = list(string)
  default     = []
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget, in USD."
  type        = number
  default     = 50
}

variable "github_oidc_provider_arn" {
  description = "ARN of the account's GitHub OIDC provider. Null skips the CI role."
  type        = string
  default     = null
}

variable "github_repository" {
  description = "GitHub repository (owner/name) allowed to push images."
  type        = string
  default     = ""
}
