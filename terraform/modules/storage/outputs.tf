output "bucket_names" {
  description = "Bucket name by role (raw, processed, artifacts)."
  value       = { for k, b in aws_s3_bucket.this : k => b.id }
}

output "bucket_arns" {
  description = "Bucket ARN by role (raw, processed, artifacts)."
  value       = { for k, b in aws_s3_bucket.this : k => b.arn }
}
