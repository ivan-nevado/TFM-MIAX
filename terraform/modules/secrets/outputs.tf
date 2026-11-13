output "parameter_arns" {
  description = "Parameter ARN by name."
  value       = { for k, p in aws_ssm_parameter.this : k => p.arn }
}

output "secret_arns" {
  description = "Secret ARN by name."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}
