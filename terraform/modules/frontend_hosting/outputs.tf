output "bucket_name" {
  description = "Bucket to sync the built frontend (frontend/dist) to."
  value       = aws_s3_bucket.site.id
}

output "distribution_id" {
  description = "CloudFront distribution (invalidate after a deploy)."
  value       = aws_cloudfront_distribution.site.id
}

output "domain_name" {
  description = "Public URL host of the frontend."
  value       = aws_cloudfront_distribution.site.domain_name
}
