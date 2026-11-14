variable "name" {
  description = "Prefix of every resource name (project and environment)."
  type        = string
}

variable "api_origin_domain" {
  description = "DNS name of the API load balancer. Null serves the static site only."
  type        = string
  default     = null
}

variable "api_origin_https" {
  description = "Whether CloudFront talks HTTPS to the API origin (requires a certificate on the load balancer)."
  type        = bool
  default     = false
}

variable "price_class" {
  description = "CloudFront price class."
  type        = string
  default     = "PriceClass_100"
}

variable "force_destroy" {
  description = "Allow Terraform to delete the non-empty site bucket."
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
