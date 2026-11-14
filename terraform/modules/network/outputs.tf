output "vpc_id" {
  description = "Identifier of the VPC."
  value       = aws_vpc.this.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC."
  value       = aws_vpc.this.cidr_block
}

output "public_subnet_ids" {
  description = "Public subnets (load balancer)."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnets (training jobs and API tasks)."
  value       = aws_subnet.private[*].id
}
