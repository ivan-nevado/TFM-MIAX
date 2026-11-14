output "bucket_names" {
  description = "Buckets by role. The owner uploads the licensed data to the raw bucket."
  value       = module.storage.bucket_names
}

output "ecr_repository_urls" {
  description = "Image repositories (training, api)."
  value       = module.ecr.repository_urls
}

output "batch_job_queue" {
  description = "Queue to submit the pipeline to."
  value       = module.training_batch.job_queue_name
}

output "batch_job_definition" {
  description = "Job definition of the pipeline."
  value       = module.training_batch.job_definition_name
}

output "api_url" {
  description = "Load balancer of the API."
  value       = "http://${module.api_service.alb_dns_name}"
}

output "frontend_url" {
  description = "Public URL of the frontend."
  value       = "https://${module.frontend_hosting.domain_name}"
}

output "frontend_bucket" {
  description = "Bucket to sync frontend/dist to."
  value       = module.frontend_hosting.bucket_name
}

output "ci_role_arn" {
  description = "Role for GitHub Actions (null without an OIDC provider)."
  value       = module.iam.ci_role_arn
}
