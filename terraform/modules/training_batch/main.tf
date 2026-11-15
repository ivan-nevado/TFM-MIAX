# AWS Batch on Spot instances for the replication pipeline. The pipeline is submitted as an array
# job: child i runs the models for one chunk of years (see scripts/batch_entrypoint.sh), reading
# the panel from S3 and writing its year partitions of the predictions table back to S3.

data "aws_partition" "current" {}

resource "aws_security_group" "batch" {
  name_prefix = "${var.name}-batch-"
  description = "Training instances: outbound only"
  vpc_id      = var.vpc_id

  egress {
    description = "HTTPS to AWS services and the public Ken French files"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.name}-batch" })

  lifecycle {
    create_before_destroy = true
  }
}

data "aws_iam_policy_document" "batch_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  name               = "${var.name}-batch-service"
  assume_role_policy = data.aws_iam_policy_document.batch_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_batch_compute_environment" "spot" {
  compute_environment_name_prefix = "${var.name}-spot-"
  type                            = "MANAGED"
  state                           = "ENABLED"
  service_role                    = aws_iam_role.batch_service.arn
  tags                            = var.tags

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    bid_percentage      = var.spot_bid_percentage
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    instance_type       = var.instance_types
    instance_role       = var.instance_profile_arn
    subnets             = var.subnet_ids
    security_group_ids  = [aws_security_group.batch.id]
    tags                = merge(var.tags, { Name = "${var.name}-training" })
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "this" {
  name     = "${var.name}-training"
  state    = "ENABLED"
  priority = 1
  tags     = var.tags

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_definition" "pipeline" {
  name                  = "${var.name}-pipeline"
  type                  = "container"
  platform_capabilities = ["EC2"]
  tags                  = var.tags

  # Overridable at submission: stage, model group and the first year of the array.
  parameters = {
    stage      = "train"
    group      = "all"
    first_year = "1990"
    chunk      = "1"
  }

  retry_strategy {
    attempts = var.retry_attempts

    evaluate_on_exit {
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }

    evaluate_on_exit {
      on_reason = "*"
      action    = "EXIT"
    }
  }

  timeout {
    attempt_duration_seconds = var.job_timeout_seconds
  }

  container_properties = jsonencode({
    image            = var.image
    command          = ["batch_entrypoint.sh", "Ref::stage", "Ref::group", "Ref::first_year", "Ref::chunk"]
    jobRoleArn       = var.job_role_arn
    executionRoleArn = var.execution_role_arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory_mib) },
    ]
    environment = [
      { name = "RELVAL_RAW_BUCKET", value = var.bucket_names["raw"] },
      { name = "RELVAL_PROCESSED_BUCKET", value = var.bucket_names["processed"] },
      { name = "RELVAL_ARTIFACTS_BUCKET", value = var.bucket_names["artifacts"] },
      { name = "RELVAL_N_JOBS", value = tostring(var.job_vcpus) },
      { name = "RELVAL_CONFIG", value = var.pipeline_config },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = var.log_group_name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "pipeline"
      }
    }
  })
}
