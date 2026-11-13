output "alb_dns_name" {
  description = "DNS name of the API load balancer."
  value       = aws_lb.this.dns_name
}

output "alb_arn_suffix" {
  description = "ARN suffix of the load balancer (CloudWatch dimension)."
  value       = aws_lb.this.arn_suffix
}

output "cluster_name" {
  description = "ECS cluster of the API."
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "ECS service of the API."
  value       = aws_ecs_service.this.name
}
