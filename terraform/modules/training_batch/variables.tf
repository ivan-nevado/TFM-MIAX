variable "name" {
  description = "Prefix of every resource name (project and environment)."
  type        = string
}

variable "region" {
  description = "AWS region (for the log configuration)."
  type        = string
}

variable "vpc_id" {
  description = "VPC of the compute environment."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets of the compute environment: private ones, or public ones that assign a public IP on launch."
  type        = list(string)
}

variable "image" {
  description = "Training image, for example <repository-url>:<tag>."
  type        = string
}

variable "job_role_arn" {
  description = "Role assumed by the pipeline container (S3 access)."
  type        = string
}

variable "execution_role_arn" {
  description = "Task execution role (image pull, logs)."
  type        = string
}

variable "instance_profile_arn" {
  description = "Instance profile of the container instances."
  type        = string
}

variable "bucket_names" {
  description = "Bucket name by role: raw, processed, artifacts."
  type        = map(string)
}

variable "log_group_name" {
  description = "CloudWatch log group of the training jobs."
  type        = string
}

variable "pipeline_config" {
  description = "Config file of the run inside ml/configs."
  type        = string
  default     = "replication.yaml"
}

variable "instance_types" {
  description = "Instance types or families of the Spot compute environment."
  type        = list(string)
  default     = ["c6i", "c6a", "m6i"]
}

variable "max_vcpus" {
  description = "Upper bound of the compute environment."
  type        = number
  default     = 64
}

variable "spot_bid_percentage" {
  description = "Maximum Spot price as a percentage of the On-Demand price."
  type        = number
  default     = 70
}

variable "job_vcpus" {
  description = "vCPUs of one array child."
  type        = number
  default     = 8
}

variable "job_memory_mib" {
  description = "Memory of one array child, in MiB."
  type        = number
  default     = 30000
}

variable "job_timeout_seconds" {
  description = "Timeout of one attempt."
  type        = number
  default     = 21600
}

variable "retry_attempts" {
  description = "Attempts per child (Spot interruptions are retried; the pipeline resumes finished years)."
  type        = number
  default     = 3
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
