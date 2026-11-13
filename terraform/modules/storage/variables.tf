variable "name" {
  description = "Prefix of the bucket names (the account id is appended)."
  type        = string
}

variable "kms_key_arn" {
  description = "KMS key for bucket encryption. Null uses S3-managed keys (SSE-S3)."
  type        = string
  default     = null
}

variable "artifacts_ia_days" {
  description = "Days after which artifacts move to Standard-IA (0 disables the transition)."
  type        = number
  default     = 60
}

variable "force_destroy" {
  description = "Allow Terraform to delete non-empty buckets. Keep false outside throwaway environments."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
