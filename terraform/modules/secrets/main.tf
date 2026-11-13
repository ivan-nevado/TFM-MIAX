# Names of configuration parameters and secrets. No real value lives in the code: parameters are
# created with a placeholder that Terraform ignores afterwards, and secrets are created empty.

resource "aws_ssm_parameter" "this" {
  for_each    = var.parameters
  name        = "${var.ssm_prefix}/${each.key}"
  description = each.value
  type        = "String"
  value       = "CHANGE_ME"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_secretsmanager_secret" "this" {
  for_each                = var.secrets
  name                    = "${trimprefix(var.ssm_prefix, "/")}/${each.key}"
  description             = each.value
  recovery_window_in_days = var.recovery_window_in_days
  tags                    = var.tags
}
