output "log_group_names" {
  description = "Log group name by workload (training, api)."
  value       = { for k, g in aws_cloudwatch_log_group.this : k => g.name }
}

output "log_group_arns" {
  description = "Log group ARN by workload (training, api)."
  value       = { for k, g in aws_cloudwatch_log_group.this : k => g.arn }
}

output "alarm_topic_arn" {
  description = "SNS topic that receives alarms and budget notifications."
  value       = aws_sns_topic.alarms.arn
}
