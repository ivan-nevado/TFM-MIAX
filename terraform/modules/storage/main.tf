# Private S3 buckets for raw data (licensed), processed data and artifacts.
# Every bucket: versioned, encrypted, all public access blocked, TLS-only, lifecycle rules.

data "aws_caller_identity" "current" {}

locals {
  buckets = {
    raw       = { noncurrent_days = 365, ia_days = 0 }
    processed = { noncurrent_days = 90, ia_days = 0 }
    artifacts = { noncurrent_days = 90, ia_days = var.artifacts_ia_days }
  }
}

resource "aws_s3_bucket" "this" {
  for_each      = local.buckets
  bucket        = "${var.name}-${each.key}-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.force_destroy
  tags          = merge(var.tags, { Name = "${var.name}-${each.key}", DataClass = each.key == "raw" ? "licensed" : "derived" })
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    bucket_key_enabled = true

    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_arn == null ? "AES256" : "aws:kms"
      kms_master_key_id = var.kms_key_arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = local.buckets[each.key].noncurrent_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  dynamic "rule" {
    for_each = local.buckets[each.key].ia_days > 0 ? [1] : []

    content {
      id     = "infrequent-access"
      status = "Enabled"

      filter {}

      transition {
        days          = local.buckets[each.key].ia_days
        storage_class = "STANDARD_IA"
      }
    }
  }

  depends_on = [aws_s3_bucket_versioning.this]
}

data "aws_iam_policy_document" "tls_only" {
  for_each = aws_s3_bucket.this

  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [each.value.arn, "${each.value.arn}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "tls_only" {
  for_each   = aws_s3_bucket.this
  bucket     = each.value.id
  policy     = data.aws_iam_policy_document.tls_only[each.key].json
  depends_on = [aws_s3_bucket_public_access_block.this]
}
