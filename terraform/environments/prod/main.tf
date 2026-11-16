# Composition of the modules for one environment. dev and prod share this file; they differ only
# in variables.tf defaults and terraform.tfvars (sizes, names, budget).

locals {
  name = "${var.project}-${var.environment}"
  # Where the API tasks and the training instances run (see var.workloads_in_public_subnets).
  workload_subnet_ids = var.workloads_in_public_subnets ? module.network.public_subnet_ids : module.network.private_subnet_ids
  ssm_prefix          = "/${var.project}/${var.environment}"
  tags = merge(var.extra_tags, {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  })
}

module "network" {
  source                  = "../../modules/network"
  name                    = local.name
  vpc_cidr                = var.vpc_cidr
  enable_nat_gateway      = var.enable_nat_gateway
  interface_endpoints     = var.interface_endpoints
  map_public_ip_on_launch = var.workloads_in_public_subnets
  tags                    = local.tags
}

module "storage" {
  source            = "../../modules/storage"
  name              = local.name
  artifacts_ia_days = var.artifacts_ia_days
  force_destroy     = var.force_destroy_buckets
  tags              = local.tags
}

module "ecr" {
  source = "../../modules/ecr"
  name   = local.name
  tags   = local.tags
}

module "secrets" {
  source     = "../../modules/secrets"
  ssm_prefix = local.ssm_prefix
  tags       = local.tags
}

module "iam" {
  source                   = "../../modules/iam"
  name                     = local.name
  bucket_arns              = module.storage.bucket_arns
  ecr_repository_arns      = values(module.ecr.repository_arns)
  ssm_prefix               = local.ssm_prefix
  secret_arns              = values(module.secrets.secret_arns)
  github_oidc_provider_arn = var.github_oidc_provider_arn
  github_repository        = var.github_repository
  tags                     = local.tags
}

module "observability" {
  source             = "../../modules/observability"
  name               = local.name
  project_tag        = var.project
  log_retention_days = var.log_retention_days
  alarm_emails       = var.alarm_emails
  monthly_budget_usd = var.monthly_budget_usd
  enable_api_alarms  = true
  alb_arn_suffix     = module.api_service.alb_arn_suffix
  ecs_cluster_name   = module.api_service.cluster_name
  ecs_service_name   = module.api_service.service_name
  tags               = local.tags
}

module "training_batch" {
  source               = "../../modules/training_batch"
  name                 = local.name
  region               = var.region
  vpc_id               = module.network.vpc_id
  subnet_ids           = local.workload_subnet_ids
  image                = "${module.ecr.repository_urls["training"]}:${var.training_image_tag}"
  job_role_arn         = module.iam.training_job_role_arn
  execution_role_arn   = module.iam.execution_role_arn
  instance_profile_arn = module.iam.batch_instance_profile_arn
  bucket_names         = module.storage.bucket_names
  log_group_name       = module.observability.log_group_names["training"]
  max_vcpus            = var.batch_max_vcpus
  instance_types       = var.batch_instance_types
  job_vcpus            = var.batch_job_vcpus
  job_memory_mib       = var.batch_job_memory_mib
  tags                 = local.tags
}

module "api_service" {
  source             = "../../modules/api_service"
  name               = local.name
  region             = var.region
  vpc_id             = module.network.vpc_id
  vpc_cidr           = module.network.vpc_cidr
  public_subnet_ids  = module.network.public_subnet_ids
  task_subnet_ids    = local.workload_subnet_ids
  assign_public_ip   = var.workloads_in_public_subnets
  image              = "${module.ecr.repository_urls["api"]}:${var.api_image_tag}"
  execution_role_arn = module.iam.execution_role_arn
  task_role_arn      = module.iam.api_task_role_arn
  artifacts_bucket   = module.storage.bucket_names["artifacts"]
  log_group_name     = module.observability.log_group_names["api"]
  cpu                = var.api_cpu
  memory             = var.api_memory
  desired_count      = var.api_desired_count
  certificate_arn    = var.api_certificate_arn
  container_insights = var.container_insights
  tags               = local.tags
}

module "frontend_hosting" {
  source            = "../../modules/frontend_hosting"
  name              = local.name
  api_origin_domain = module.api_service.alb_dns_name
  api_origin_https  = var.api_certificate_arn != null
  force_destroy     = var.force_destroy_buckets
  tags              = local.tags
}
