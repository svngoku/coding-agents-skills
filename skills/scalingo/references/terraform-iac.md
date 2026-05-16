# Scalingo Terraform Provider

The official Scalingo Terraform provider manages apps, addons, environment, containers, domains, SCM integrations, collaborators, and log drains declaratively.

- **Provider**: `Scalingo/scalingo`
- **Registry**: https://registry.terraform.io/providers/Scalingo/scalingo/latest/docs
- **Opinionated module**: `scalingo-community/app/scalingo` (wraps common patterns)

## Setup

```hcl
terraform {
  required_providers {
    scalingo = {
      source  = "Scalingo/scalingo"
      version = "~> 2.0"
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
resource "scalingo_app" "api" {
  name  = "my-api"
  stack = "scalingo-22"    # optional, defaults to current stack
}
```

Outputs useful later: `scalingo_app.api.name`, `scalingo_app.api.id`, `scalingo_app.api.url` (the default subdomain URL), `scalingo_app.api.git_url`.

### `scalingo_addon`

```hcl
resource "scalingo_addon" "db" {
  provider_id = "postgresql"
  plan        = "postgresql-starter-512"
  app         = scalingo_app.api.id
}
```

The addon block is where you set up managed databases. Provider IDs: `postgresql`, `mysql`, `mongodb`, `redis`, `opensearch`, `influxdb`, and non-database providers.

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

Let's Encrypt cert is auto-issued once DNS points to Scalingo. No need to manage certs in Terraform for standard cases. If you need a custom cert:

```hcl
resource "scalingo_domain" "custom" {
  common_name = "secure.example.com"
  app         = scalingo_app.api.id
  tls_cert    = file("cert.pem")
  tls_key     = file("key.pem")
}
```

### `scalingo_collaborator`

```hcl
resource "scalingo_collaborator" "teammate" {
  app   = scalingo_app.api.id
  email = "teammate@example.com"
}
```

The person receives an email invite. Terraform doesn't wait for acceptance.

### `scalingo_scm_integration` and `scalingo_scm_repo_link`

```hcl
resource "scalingo_scm_integration" "gh" {
  scm_type = "github"
  url      = "https://github.com"
  access_token = var.github_token
}

resource "scalingo_scm_repo_link" "api_repo" {
  source                     = "https://github.com/org/api-repo"
  branch                     = "main"
  auth_integration_uuid      = scalingo_scm_integration.gh.id
  app                        = scalingo_app.api.id
  auto_deploy_enabled        = true
  deploy_review_apps_enabled = true
  destroy_on_close_enabled   = true
  hours_before_delete_on_close = 2
}
```

This replaces the one-off CLI setup with reproducible config.

### `scalingo_log_drain`

```hcl
resource "scalingo_log_drain" "syslog" {
  app  = scalingo_app.api.id
  type = "syslog"
  url  = "syslog+tls://logs.example.com:6514"
}

# For addon-specific log drain
resource "scalingo_log_drain_addon" "db_syslog" {
  app       = scalingo_app.api.id
  addon_id  = scalingo_addon.db.id
  type      = "syslog"
  url       = "syslog+tls://logs.example.com:6514"
}
```

### `scalingo_notifier` and notification platforms

For alerting integrations (Slack, email, webhook). Configure platforms first, then notifiers that reference them.

### `scalingo_autoscaler`

```hcl
resource "scalingo_autoscaler" "web_rpm" {
  app            = scalingo_app.api.id
  container_type = "web"
  metric         = "rpm_per_container"
  target         = 200
  min_containers = 1
  max_containers = 10
}
```

Managing autoscalers via Terraform is the cleanest approach for production — the formation block sets initial capacity, the autoscaler adjusts from there.

## Environment variables

Two patterns:

### Inline per-variable resources

```hcl
resource "scalingo_env_var" "secret_key" {
  app   = scalingo_app.api.id
  name  = "SECRET_KEY"
  value = var.secret_key
}
```

### Map via `for_each`

```hcl
locals {
  app_env = {
    DJANGO_SETTINGS_MODULE = "myapp.settings.prod"
    SENTRY_DSN             = var.sentry_dsn
    LOG_LEVEL              = "INFO"
  }
}

resource "scalingo_env_var" "app_env" {
  for_each = local.app_env
  app      = scalingo_app.api.id
  name     = each.key
  value    = each.value
}
```

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

## Importing existing resources

Existing apps and addons can be imported into Terraform state:

```bash
# App (import by name)
terraform import scalingo_app.api my-api

# Addon (format: <app-name>:<addon-id>)
terraform import scalingo_addon.db my-api:ad-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Domain (format: <app-name>:<domain-name>)
terraform import scalingo_domain.apex my-api:example.com

# Env var (format: <app-name>:<var-id>)
terraform import scalingo_env_var.secret my-api:env-uuid
```

After import, run `terraform plan` and reconcile any drift.

## The opinionated community module

`scalingo-community/app/scalingo` bundles an app with its database, env, formation, log drains, and review-app config in one module call:

```hcl
module "api" {
  source  = "scalingo-community/app/scalingo"
  version = "~> 0.3"

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
    { provider = "redis",      plan = "redis-starter-128" },
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

- Terraform state includes secret values (env vars, API tokens, TLS keys). Store state remotely with encryption — S3+KMS, Terraform Cloud, or GitLab's managed backend.
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
