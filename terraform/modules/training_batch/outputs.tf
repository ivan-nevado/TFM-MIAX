output "job_queue_arn" {
  description = "Queue to submit the pipeline to."
  value       = aws_batch_job_queue.this.arn
}

output "job_queue_name" {
  description = "Name of the job queue."
  value       = aws_batch_job_queue.this.name
}

output "job_definition_arn" {
  description = "Job definition of the pipeline."
  value       = aws_batch_job_definition.pipeline.arn
}

output "job_definition_name" {
  description = "Name of the job definition."
  value       = aws_batch_job_definition.pipeline.name
}
