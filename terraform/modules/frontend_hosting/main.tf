# Static hosting of the built frontend: a private S3 bucket served by CloudFront (origin access
# control). Requests to /api/* and /health are forwarded to the API load balancer when given.

data "aws_caller_identity" "current" {}

locals {
  managed_caching_optimized = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  managed_caching_disabled  = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
  managed_all_viewer        = "216adef6-5c7f-47e4-b989-5492eafa07d3"
  has_api                   = var.api_origin_domain != null
}

resource "aws_s3_bucket" "site" {
  bucket        = "${var.name}-frontend-${data.aws_caller_identity.current.account_id}"
  force_destroy = var.force_destroy
  tags          = var.tags
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.name}-frontend"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  comment             = "${var.name} frontend"
  default_root_object = "index.html"
  price_class         = var.price_class
  tags                = var.tags

  origin {
    origin_id                = "site"
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  dynamic "origin" {
    for_each = local.has_api ? [1] : []

    content {
      origin_id   = "api"
      domain_name = var.api_origin_domain

      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = var.api_origin_https ? "https-only" : "http-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }
    }
  }

  default_cache_behavior {
    target_origin_id       = "site"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    cache_policy_id        = local.managed_caching_optimized
    compress               = true
  }

  dynamic "ordered_cache_behavior" {
    for_each = local.has_api ? ["/api/*", "/health"] : []

    content {
      path_pattern             = ordered_cache_behavior.value
      target_origin_id         = "api"
      viewer_protocol_policy   = "redirect-to-https"
      allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods           = ["GET", "HEAD"]
      cache_policy_id          = local.managed_caching_disabled
      origin_request_policy_id = local.managed_all_viewer
    }
  }

  # Single-page application: unknown paths return the app shell.
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 10
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

data "aws_iam_policy_document" "site" {
  statement {
    sid       = "AllowCloudFront"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.site.arn}/*"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.site.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket     = aws_s3_bucket.site.id
  policy     = data.aws_iam_policy_document.site.json
  depends_on = [aws_s3_bucket_public_access_block.site]
}
