output "repository_urls" {
  description = "Repository URL by name (training, api)."
  value       = { for k, r in aws_ecr_repository.this : k => r.repository_url }
}

output "repository_arns" {
  description = "Repository ARN by name (training, api)."
  value       = { for k, r in aws_ecr_repository.this : k => r.arn }
}
