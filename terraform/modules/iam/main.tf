# Least-privilege roles: training job, API task, shared task-execution role, Batch instances, CI.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_partition" "current" {}

locals {
  ssm_prefix_arn = "arn:${data.aws_partition.current.partition}:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter${var.ssm_prefix}/*"
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# ---- training job: reads raw data, reads and writes processed data and artifacts -----------------
data "aws_iam_policy_document" "training" {
  statement {
    sid       = "ListBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = values(var.bucket_arns)
  }

  statement {
    sid       = "ReadRaw"
    actions   = ["s3:GetObject"]
    resources = ["${var.bucket_arns["raw"]}/*"]
  }

  statement {
    sid       = "ReadWriteDerived"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${var.bucket_arns["processed"]}/*", "${var.bucket_arns["artifacts"]}/*"]
  }
}

resource "aws_iam_role" "training" {
  name               = "${var.name}-training-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy" "training" {
  name   = "data-access"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}

# ---- API task: read-only access to derived outputs -----------------------------------------------
data "aws_iam_policy_document" "api" {
  statement {
    sid       = "ListDerivedBuckets"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [var.bucket_arns["processed"], var.bucket_arns["artifacts"]]
  }

  statement {
    sid       = "ReadDerived"
    actions   = ["s3:GetObject"]
    resources = ["${var.bucket_arns["processed"]}/*", "${var.bucket_arns["artifacts"]}/*"]
  }
}

resource "aws_iam_role" "api" {
  name               = "${var.name}-api-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy" "api" {
  name   = "read-derived-outputs"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# ---- task execution role: pull images, write logs, read configuration parameters -----------------
data "aws_iam_policy_document" "execution_extra" {
  statement {
    sid       = "ReadConfiguration"
    actions   = ["ssm:GetParameters", "ssm:GetParameter"]
    resources = [local.ssm_prefix_arn]
  }

  dynamic "statement" {
    for_each = length(var.secret_arns) > 0 ? [1] : []

    content {
      sid       = "ReadSecrets"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = var.secret_arns
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_extra" {
  name   = "read-configuration"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_extra.json
}

# ---- Batch container instances --------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_instance" {
  name               = "${var.name}-batch-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "batch_instance" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.name}-batch-instance"
  role = aws_iam_role.batch_instance.name
  tags = var.tags
}

# ---- CI: push images from GitHub Actions through OIDC (optional) ---------------------------------
data "aws_iam_policy_document" "ci_assume" {
  count = var.github_oidc_provider_arn == null ? 0 : 1

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [var.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/main"]
    }
  }
}

data "aws_iam_policy_document" "ci" {
  count = var.github_oidc_provider_arn == null ? 0 : 1

  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PushImages"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
    ]
    resources = var.ecr_repository_arns
  }
}

resource "aws_iam_role" "ci" {
  count              = var.github_oidc_provider_arn == null ? 0 : 1
  name               = "${var.name}-ci"
  assume_role_policy = data.aws_iam_policy_document.ci_assume[0].json
  tags               = var.tags
}

resource "aws_iam_role_policy" "ci" {
  count  = var.github_oidc_provider_arn == null ? 0 : 1
  name   = "push-images"
  role   = aws_iam_role.ci[0].id
  policy = data.aws_iam_policy_document.ci[0].json
}
