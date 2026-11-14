variable "ssm_prefix" {
  description = "Path prefix of the environment, for example /relval/dev."
  type        = string
}

variable "parameters" {
  description = "Parameter Store entries: name (under the prefix) => description. Values are set outside Terraform."
  type        = map(string)
  default = {
    "api/cors_origins" = "Origins allowed to call the API"
    "api/run_name"     = "Replication run served by the API"
  }
}

variable "secrets" {
  description = "Secrets Manager entries: name => description. Created without a value."
  type        = map(string)
  default = {
    "llm/api_key" = "API key of the LLM provider (later extension: qualitative scores)"
  }
}

variable "recovery_window_in_days" {
  description = "Days a deleted secret can be recovered."
  type        = number
  default     = 7
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
