# Scalingo Terraform Provider

The official Scalingo Terraform provider (v2.8.0 as of 2026-09) manages apps (and their environment), addons, container types, autoscalers, alerts, domains, SCM integrations, collaborators, log drains, projects, firewall rules, SSH keys, and Dedicated Resources databases declaratively.

- **Provider**: `Scalingo/scalingo`
- **Registry**: https://registry.terraform.io/providers/Scalingo/scalingo/latest/docs
- **Opinionated module**: `scalingo-community/app/scalingo` (wraps common patterns)

## Setup

```hcl
terraform {
  required_providers {
    scalingo = {
      source  = "Scalingo/scalingo"
      version = "~> 2.0"   # use "~> 2.8" for scalingo_app_firewall_rule
    }
  }
}

provider "scalingo" {
  region = "osc-fr1"
  # api_token is picked up from SCALINGO_API_TOKEN env var
}
```

Authentication — prefer env vars over hardcoded tokens:

```bash
export SCALINGO_API_TOKEN="tk-us-..."
export SCALINGO_REGION="osc-fr1"     # or osc-secnum-fr1
```

Generate the token in the Scalingo dashboard: Account Settings → API Tokens.

Never commit a token into Terraform files that hit version control.

## Core resources

### `scalingo_app`

```hcl
data "scalingo_stack" "s26" {
  name = "scalingo-26"
}

resource "scalingo_app" "api" {
  name        = "my-api"
  stack_id    = data.scalingo_stack.s26.id   # optional; the attribute is stack_id, not stack
  project_id  = scalingo_project.main.id   # optional
  force_https = true
  environment = {
    DJANGO_SETTINGS_MODULE = "myapp.settings.prod"
    LOG_LEVEL              = "INFO"
  }
}
```

The provider's example resolves the stack through the `scalingo_stack` data source; Scalingo's own tutorials pass the slug directly (`stack_id = "scalingo-26"`). Other optional attributes: `router_logs`, `sticky_session`, `hds_resource`. `all_environment` is a read-only map that also includes addon-injected variables.

Outputs useful later: `scalingo_app.api.name`, `scalingo_app.api.id`, `scalingo_app.api.url` (the default subdomain URL), `scalingo_app.api.git_url`.

### `scalingo_addon`

```hcl
resource "scalingo_addon" "db" {
  provider_id = "postgresql"
  plan        = "postgresql-starter-512"
  app         = scalingo_app.api.id

  database_features = ["force-ssl"]   # optional toggles
}
```

The addon block is where you set up managed databases. List provider IDs with `scalingo addons-list` (or the `scalingo_addon_providers` data source) and plans with `scalingo addons-plans <provider>`. The provider's own example uses `scalingo-redis` with `redis-starter-256`. Import format: `<app-id>:<addon-id>`.

### `scalingo_container_type` — formation

One resource per process type:

```hcl
resource "scalingo_container_type" "web" {
  app    = scalingo_app.api.name
  name   = "web"
  amount = 2
  size   = "L"
}

resource "scalingo_container_type" "worker" {
  app    = scalingo_app.api.name
  name   = "worker"
  amount = 1
  size   = "M"
}
```

Manual CLI scaling will drift from Terraform state — either manage scaling exclusively via Terraform (and commit changes) or not at all. Mixing leads to confusion.

### `scalingo_domain`

```hcl
resource "scalingo_domain" "apex" {
  common_name = "example.com"
  app         = scalingo_app.api.id
}
```

Let's Encrypt cert is auto-issued once DNS points to Scalingo. The resource only supports `common_name`, `app`, `canonical` (redirect all traffic to this domain, one per app), and `letsencrypt_enabled` (default true):

```hcl
resource "scalingo_domain" "www" {
  common_name = "www.example.com"
  app         = scalingo_app.api.id
  canonical   = true
}
```

There are no `tls_cert` / `tls_key` attributes. To use a custom certificate, set `letsencrypt_enabled = false` and upload the certificate outside Terraform: `scalingo --app my-api domains-ssl --cert cert.pem --key key.pem secure.example.com`.

### `scalingo_collaborator`

```hcl
resource "scalingo_collaborator" "teammate" {
  app     = scalingo_app.api.id
  email   = "teammate@example.com"
  limited = false   # true = Limited Collaborator role
}
```

The person receives an email invite. Terraform doesn't wait for acceptance.

### `scalingo_scm_integration` and `scalingo_scm_repo_link`

GitHub.com and GitLab.com integrations are created once from the dashboard and read with a data source. The `scalingo_scm_integration` resource is only for `github-enterprise` and `gitlab-self-hosted`.

```hcl
data "scalingo_scm_integration" "github" {
  scm_type = "github"
}

resource "scalingo_scm_repo_link" "api_repo" {
  source                       = "https://github.com/org/api-repo"
  branch                       = "main"
  auth_integration_uuid        = data.scalingo_scm_integration.github.id
  app                          = scalingo_app.api.id
  auto_deploy_enabled          = true
  deploy_review_apps_enabled   = true
  delete_on_close_enabled      = true
  hours_before_delete_on_close = 2
  delete_stale_enabled         = true
  hours_before_delete_stale    = 168
}

# Self-hosted SCM only
resource "scalingo_scm_integration" "gitlab_internal" {
  scm_type     = "gitlab-self-hosted"
  url          = "https://gitlab.example.com"
  access_token = var.gitlab_token
}
```

This replaces the one-off CLI setup with reproducible config.

### `scalingo_log_drain`

```hcl
resource "scalingo_log_drain" "syslog" {
  app  = scalingo_app.api.id
  type = "syslog"
  host = "logs.example.com"
  port = "6514"
}

# Addon (database) log drain: same resource, with the addon UUID
resource "scalingo_log_drain" "db_syslog" {
  app   = scalingo_app.api.id
  addon = scalingo_addon.db.id
  type  = "syslog"
  host  = "logs.example.com"
  port  = "6514"
}
```

`url` is a sensitive helper that fills `host`/`port`/`token`; `token` and `drain_region` are used by some vendors. There is no `scalingo_log_drain_addon` resource.

### `scalingo_notifier` and notification platforms

For alerting integrations (Slack, email, webhook). Look up the platform with the `scalingo_notification_platform` data source, then create notifiers and `scalingo_alert` resources that reference them:

```hcl
data "scalingo_notification_platform" "email" {
  name = "email"
}

resource "scalingo_notifier" "ops" {
  app         = scalingo_app.api.id
  platform_id = data.scalingo_notification_platform.email.id
  name        = "Ops email"
  emails      = ["ops@example.com"]
}

resource "scalingo_alert" "web_cpu" {
  app            = scalingo_app.api.id
  container_type = "web"
  metric         = "cpu"
  limit          = 0.8
  notifiers      = [scalingo_notifier.ops.id]
}
```

### `scalingo_autoscaler`

```hcl
resource "scalingo_autoscaler" "web_rpm" {
  app            = scalingo_app.api.id
  container_type = "web"
  metric         = "rpm_per_container"
  target         = 200
  min_containers = 2   # the autoscaler needs at least 2
  max_containers = 10
}
```

Metric names match the CLI (`cpu`, `memory`, `swap`, `rpm_per_container`, `p95_response_time`, `5XX`); `cpu` targets are ratios (`0.8` = 80%). Set `disabled = true` to pause without deleting. Import format: `<app-id>:<autoscaler-id>`.

Managing autoscalers via Terraform is the cleanest approach for production — the formation block sets initial capacity, the autoscaler adjusts from there.

## Environment variables

There is no `scalingo_env_var` resource. Env vars live in the `environment` map of `scalingo_app`:

```hcl
locals {
  app_env = {
    DJANGO_SETTINGS_MODULE = "myapp.settings.prod"
    SENTRY_DSN             = var.sentry_dsn
    LOG_LEVEL              = "INFO"
  }
}

resource "scalingo_app" "api" {
  name        = "my-api"
  environment = local.app_env
}
```

Variables injected by addons (e.g. `SCALINGO_POSTGRESQL_URL`) don't belong in this map; read them from `scalingo_app.api.all_environment` if needed.

Remember: env changes via Terraform don't trigger a restart. Pair with a `null_resource` calling `scalingo restart` on changes, or restart manually:

```hcl
resource "null_resource" "restart_on_env_change" {
  triggers = {
    env_hash = md5(jsonencode(local.app_env))
  }

  provisioner "local-exec" {
    command = "scalingo --app ${scalingo_app.api.name} restart"
  }
}
```

## Projects, firewall rules, and Dedicated Resources databases

```hcl
resource "scalingo_project" "main" {
  name    = "payments"
  default = false
}

# Application Firewall: restrict inbound traffic to listed IPv4 CIDRs (provider >= 2.8)
resource "scalingo_app_firewall_rule" "office" {
  app   = scalingo_app.api.id
  cidr  = "192.0.2.0/24"
  label = "office"
}

# Dedicated Resources PostgreSQL (preview), not attached to an app
resource "scalingo_database" "pg" {
  name       = "my-postgres-db"
  technology = "postgresql-ng"
  plan       = "postgresql-ng-enterprise-4096"
  project_id = scalingo_project.main.id
}

resource "scalingo_database_firewall_rule" "office" {
  database_id = scalingo_database.pg.database_id
  cidr        = "203.0.113.0/24"
  label       = "Office network"
}
```

Also available: `scalingo_ssh_key`, and data sources such as `scalingo_stack`, `scalingo_region`, `scalingo_container_size`, `scalingo_project`, `scalingo_private_network_domain`, and `scalingo_database_endpoints` (provider >= 2.8).

## Importing existing resources

Existing apps and addons can be imported into Terraform state:

```bash
# App (import by name or ID)
terraform import scalingo_app.api my-api

# Addon (format: <app-id>:<addon-id>)
terraform import scalingo_addon.db <app-id>:ad-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Domain (format: <app-id>:<domain-id>, the domain's ID, not its hostname)
terraform import scalingo_domain.www <app-id>:<domain-id>

# Autoscaler (format: <app-id>:<autoscaler-id>)
terraform import scalingo_autoscaler.web <app-id>:<autoscaler-id>
```

Env vars are imported with the app (they are part of its `environment` map).

After import, run `terraform plan` and reconcile any drift.

## The opinionated community module

`scalingo-community/app/scalingo` bundles an app with its database, env, formation, log drains, and review-app config in one module call:

```hcl
module "api" {
  source  = "scalingo-community/app/scalingo"
  version = "~> 0.6"   # v0.6.0 needs provider ~> 2.7; default stack scalingo-26

  name   = "my-api"
  domain = "api.example.com"

  environment = {
    LOG_LEVEL = "INFO"
  }

  containers = {
    web    = { size = "L", amount = 2 }
    worker = { size = "M", amount = 1 }
  }

  addons = [
    { provider = "postgresql", plan = "postgresql-starter-512" },
    { provider = "redis",      plan = "redis-starter-256" },
  ]

  log_drains = [
    { type = "elk", url = "https://user:pass@logstash.example.com" }
  ]

  review_apps = {
    enabled                         = true
    delete_on_close_enabled         = true
    hours_before_delete_on_close    = "2"
    delete_stale_enabled            = true
    hours_before_delete_stale       = "168"
    automatic_creation_from_forks_allowed = false
  }
}
```

The module handles the resource wiring. Good for teams standardizing a fleet of similar apps.

## State and secrets

- Terraform state includes secret values (env vars, API tokens, log drain tokens). Store state remotely with encryption — S3+KMS, Terraform Cloud, or GitLab's managed backend.
- Never commit `.tfstate` to git.
- Use `sensitive = true` on outputs that include secrets.
- For secrets sourced from a vault (HashiCorp Vault, AWS Secrets Manager), use the relevant data source to avoid duplicating secrets into tfvars files.

## Multi-region setups

Target multiple regions in one Terraform run with provider aliases:

```hcl
provider "scalingo" {
  alias  = "fr1"
  region = "osc-fr1"
}

provider "scalingo" {
  alias  = "secnum"
  region = "osc-secnum-fr1"
}

resource "scalingo_app" "public" {
  provider = scalingo.fr1
  name     = "public-api"
}

resource "scalingo_app" "sovereign" {
  provider = scalingo.secnum
  name     = "sovereign-api"
}
```

Useful when a product spans both general and sovereign deployments.

## CI/CD patterns

Typical GitHub Actions flow:

1. CI builds and tests
2. CI runs `terraform plan` (on PRs) and `terraform apply` (on merges to main)
3. `scalingo deploy` (or git push) triggers the actual code deployment
4. Terraform manages infra; git push/SCM integration manages code

Keep infra and app deployments as separate pipeline stages — Terraform changes shouldn't gate code pushes, and vice versa.
