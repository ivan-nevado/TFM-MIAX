# Remote state in S3 with native locking. Configured but not initialised: the bucket, key and
# region are supplied by the owner at init time, so no account detail lives in the code.
#
#   terraform init \
#     -backend-config="bucket=<state-bucket>" \
#     -backend-config="key=relval/dev/terraform.tfstate" \
#     -backend-config="region=<region>"
#
# For validation without any backend or credentials: terraform init -backend=false

terraform {
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }
}
