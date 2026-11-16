variable "name" {
  description = "Prefix of every resource name (project and environment)."
  type        = string
}

variable "project_tag" {
  description = "Value of the Project tag used to filter the cost budget."
  type        = string
}

variable "log_groups" {
  description = "Log groups to create, as /<name>/<group>."
  type        = list(string)
  default     = ["training", "api"]
}

variable "log_retention_days" {
  description = "Retention of the log groups."
  type        = number
  default     = 30
}

variable "alarm_emails" {
  description = "E-mail addresses subscribed to the alarm topic (each must confirm the subscription)."
  type        = list(string)
  default     = []
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget of the environment, in USD."
  type        = number
  default     = 50
}

variable "enable_api_alarms" {
  description = "Create the API 5xx and CPU alarms. A literal bool, so count is known at plan time."
  type        = bool
  default     = false
}

variable "alb_arn_suffix" {
  description = "ARN suffix of the API load balancer (required when enable_api_alarms is true)."
  type        = string
  default     = null
}

variable "ecs_cluster_name" {
  description = "ECS cluster of the API (required when enable_api_alarms is true)."
  type        = string
  default     = null
}

variable "ecs_service_name" {
  description = "ECS service of the API."
  type        = string
  default     = null
}

variable "api_5xx_threshold" {
  description = "Server errors in five minutes that trigger the alarm."
  type        = number
  default     = 5
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
