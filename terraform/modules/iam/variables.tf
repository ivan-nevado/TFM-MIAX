variable "name" {
  description = "Prefix of every role name (project and environment)."
  type        = string
}

variable "bucket_arns" {
  description = "Bucket ARN by role: raw, processed, artifacts (output of the storage module)."
  type        = map(string)
}

variable "ecr_repository_arns" {
  description = "Repositories the CI role may push to."
  type        = list(string)
  default     = []
}

variable "ssm_prefix" {
  description = "Parameter Store path prefix the tasks may read, for example /relval/dev."
  type        = string
}

variable "secret_arns" {
  description = "Secrets Manager secrets the tasks may read."
  type        = list(string)
  default     = []
}

variable "github_oidc_provider_arn" {
  description = "ARN of the account's GitHub OIDC provider. Null skips the CI role."
  type        = string
  default     = null
}

variable "github_repository" {
  description = "GitHub repository (owner/name) allowed to assume the CI role."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
