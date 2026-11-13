variable "name" {
  description = "Namespace of the repositories (project and environment)."
  type        = string
}

variable "repositories" {
  description = "Repository names inside the namespace."
  type        = list(string)
  default     = ["training", "api"]
}

variable "keep_images" {
  description = "Number of images kept per repository."
  type        = number
  default     = 20
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
