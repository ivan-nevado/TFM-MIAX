# CloudWatch log groups, an alarm topic, alarms for the API and failed training jobs, and a
# monthly cost budget.

resource "aws_cloudwatch_log_group" "this" {
  for_each          = toset(var.log_groups)
  name              = "/${var.name}/${each.value}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

resource "aws_sns_topic" "alarms" {
  name = "${var.name}-alarms"
  tags = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  for_each  = toset(var.alarm_emails)
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = each.value
}

# ---- API alarms (count uses a plan-time flag: ALB/ECS attributes are unknown until apply) ---------
resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  count               = var.enable_api_alarms ? 1 : 0
  alarm_name          = "${var.name}-api-5xx"
  alarm_description   = "The API returned server errors"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.api_5xx_threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags

  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "api_cpu" {
  count               = var.enable_api_alarms ? 1 : 0
  alarm_name          = "${var.name}-api-cpu"
  alarm_description   = "API tasks are close to their CPU limit"
  namespace           = "AWS/ECS"
  metric_name         = "CPUUtilization"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  tags                = var.tags

  dimensions = {
    ClusterName = var.ecs_cluster_name
    ServiceName = var.ecs_service_name
  }
}

# ---- failed training jobs --------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "batch_failed" {
  name        = "${var.name}-batch-job-failed"
  description = "A training job failed"
  tags        = var.tags

  event_pattern = jsonencode({
    source      = ["aws.batch"]
    detail-type = ["Batch Job State Change"]
    detail      = { status = ["FAILED"] }
  })
}

resource "aws_cloudwatch_event_target" "batch_failed" {
  rule = aws_cloudwatch_event_rule.batch_failed.name
  arn  = aws_sns_topic.alarms.arn
}

data "aws_iam_policy_document" "sns_events" {
  statement {
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alarms.arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com", "cloudwatch.amazonaws.com", "budgets.amazonaws.com"]
    }
  }
}

resource "aws_sns_topic_policy" "alarms" {
  arn    = aws_sns_topic.alarms.arn
  policy = data.aws_iam_policy_document.sns_events.json
}

# ---- monthly budget ----------------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$${var.project_tag}"]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.alarms.arn]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_sns_topic_arns = [aws_sns_topic.alarms.arn]
  }
}
