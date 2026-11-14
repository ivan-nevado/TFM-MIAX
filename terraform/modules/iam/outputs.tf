output "training_job_role_arn" {
  description = "Role assumed by the training containers."
  value       = aws_iam_role.training.arn
}

output "api_task_role_arn" {
  description = "Role assumed by the API containers."
  value       = aws_iam_role.api.arn
}

output "execution_role_arn" {
  description = "Task execution role (image pull, logs, configuration)."
  value       = aws_iam_role.execution.arn
}

output "batch_instance_profile_arn" {
  description = "Instance profile of the Batch compute environment."
  value       = aws_iam_instance_profile.batch_instance.arn
}

output "ci_role_arn" {
  description = "Role for GitHub Actions (null when the OIDC provider is not configured)."
  value       = try(aws_iam_role.ci[0].arn, null)
}
