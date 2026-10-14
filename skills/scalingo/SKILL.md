---
name: scalingo
description: >
  Scalingo is a European (French) Platform-as-a-Service for deploying and operating web applications, background workers, and managed databases. Use this skill whenever the user mentions Scalingo, wants to deploy an app to Scalingo, work with the scalingo CLI, provision addons (PostgreSQL, MySQL, MongoDB, Redis, OpenSearch, InfluxDB), configure review apps, manage the scalingo.json manifest or a Procfile, scale containers, set up log drains, migrate from Heroku, use the Scalingo Terraform provider, or target sovereign regions (osc-fr1, osc-secnum-fr1 on SecNumCloud infrastructure). Also trigger for "deploy a French PaaS", "HDS-compliant hosting", "SecNumCloud deployment", or questions about Scalingo's Heroku-compatible buildpack model.
---

# Scalingo

Scalingo is a European PaaS in the Heroku lineage. You push code (usually via git), the platform builds it with a buildpack, and containers are started according to your `Procfile`. Managed databases are available as addons. The platform's differentiators are European/French jurisdiction, ISO 27001 / HDS certification, and a region (`osc-secnum-fr1`) running on SecNumCloud-qualified Outscale infrastructure.

This skill covers the platform end-to-end at a working level: CLI, deployment, addons, scaling, review apps, and IaC. For deep dives on any single topic, consult the `references/` files — pointers are given throughout.

## When to reach for each reference

The main file stays practical. Pull in a reference when:

- **`references/cli-reference.md`** — user needs a specific CLI command you don't remember, or wants a full command map
- **`references/deployment.md`** — deploying via non-git methods (GitHub/GitLab integration, tarball archive, WAR), or configuring auto-deploy
- **`references/addons-databases.md`** — provisioning, connecting to, or operating a managed database
- **`references/scaling-autoscaler.md`** — configuring horizontal/vertical scaling or the autoscaler's metric targets
- **`references/manifest-review-apps.md`** — writing a `scalingo.json`, configuring review apps, or building a one-click deploy button
- **`references/terraform-iac.md`** — managing Scalingo resources via the official Terraform provider
- **`references/buildpacks.md`** — understanding stacks, picking or chaining buildpacks, writing a custom buildpack

## Mental model

Three nouns to internalize:

- **App** — the unit of deployment. Has a name, a region, environment variables, and a git remote. One app = one deployable project. Apps can be grouped into **Projects**, which also scope Private Networks.
- **Process type** — a named command from the `Procfile` (`web`, `tcp`, `postdeploy`, or anything custom like `worker`). Each process type runs in one or more containers.
- **Addon** — a managed service attached to an app (database, cache). Exposes connection details via env vars like `SCALINGO_POSTGRESQL_URL`.

Everything else — containers, logs, metrics, domains, collaborators — hangs off an app.

## Installation and auth

```bash
# Install (macOS/Linux)
curl -O https://cli-dl.scalingo.com/install && bash install

# Log in (opens a browser)
scalingo login

# Or use an API token (for CI/non-interactive)
export SCALINGO_API_TOKEN="tk-us-..."
```

Set a default region once so you don't repeat `--region` on every command:

```bash
scalingo config --region osc-fr1          # Paris, standard
# OR
scalingo config --region osc-secnum-fr1   # Paris, SecNumCloud infrastructure (sovereign)
```

Region notes:

- `osc-fr1` — default; hosted on Outscale in Paris
- `osc-secnum-fr1` — runs on Outscale's SecNumCloud-qualified IaaS; used for French public-sector and sensitive workloads. Access is granted through Scalingo support. The Scalingo PaaS layer itself is not (yet) SecNumCloud-qualified; its qualification is in progress
- List current regions at any time with `scalingo regions`

## Quick start: create and deploy an app

```bash
# 1. Create the app (empty, in the default region)
scalingo create my-app

# 2. Add the git remote the CLI printed — or ask for it explicitly:
scalingo --app my-app git-show
git remote add scalingo git@ssh.osc-fr1.scalingo.com:my-app.git

# 3. Push. master OR main are accepted; other branches need mapping:
git push scalingo main
# git push scalingo feature-x:main   # deploy a different local branch

# 4. Watch the build
scalingo --app my-app logs -f
```

The build detects your stack from files in the repo (Gemfile → Ruby, package.json → Node, requirements.txt/pyproject.toml → Python, etc.) and runs the matching buildpack. Force a buildpack with `BUILDPACK_URL` if detection is wrong — see `references/buildpacks.md`.

By default, the `web` process type scales to 1 × M container. All other process types default to 0 and must be scaled explicitly to start.

## The Procfile

A `Procfile` at repo root declares how each process type starts:

```text
web: gunicorn app:app --bind 0.0.0.0:$PORT
worker: celery -A tasks worker --loglevel=INFO
postdeploy: python manage.py migrate
```

Reserved process types:

- `web` — receives HTTP traffic, must bind to `$PORT`, scaled to 1 by default
- `tcp` — receives raw TCP traffic through the TCP gateway addon
- `postdeploy` — runs once per deployment in a separate one-off container (size M, 20-minute limit) after the build and before traffic switches to the new containers; use it for migrations. It replaces the deprecated `scripts.postdeploy` in `scalingo.json`

There is no `release` process type on Scalingo (that's Heroku). Migrations belong in `postdeploy`.

Everything else is a custom process type — scale it to ≥1 to start it:

```bash
scalingo --app my-app scale worker:1:M
```

## Common operations (CLI cheatsheet)

The CLI almost always wants `--app <name>` (or `-a <name>`). All commands below assume that flag.

```bash
# Apps
scalingo apps                        # list your apps
scalingo create my-app               # create (--project-id to place it in a project)
scalingo --app my-app apps-info      # details (stack, region, status)
scalingo rename --app old --new-name new
scalingo destroy --app my-app        # delete (confirmation required)

# Deploy / restart
git push scalingo main               # standard deploy
scalingo --app my-app deploy my-app.tar.gz  # deploy from an archive (local path or URL)
scalingo --app my-app restart                # restart all containers
scalingo --app my-app restart web            # restart one process type

# Environment
scalingo --app my-app env                    # list
scalingo --app my-app env-set KEY=value OTHER=value
scalingo --app my-app env-unset KEY
# NOTE: env changes do NOT auto-restart; add `scalingo restart` after bulk changes.

# Scaling (amount:size syntax)
scalingo --app my-app scale web:2:L worker:1:M
scalingo --app my-app ps                     # current formation

# Logs
scalingo --app my-app logs                   # last lines
scalingo --app my-app logs -f                # stream
scalingo --app my-app logs --lines 500       # historical

# One-off containers (copy of production env, destroyed on exit)
scalingo --app my-app run bash               # interactive shell
scalingo --app my-app run --size XL rake db:migrate
scalingo --app my-app run --detached long-job.sh

# Domains + TLS (Let's Encrypt is automatic; certificates last 45 days and renew automatically)
scalingo --app my-app domains-add example.com
scalingo --app my-app domains                # list
scalingo --app my-app set-canonical-domain example.com

# Addons
scalingo --app my-app addons                 # list attached
scalingo --app my-app addons-add postgresql postgresql-starter-512
scalingo --app my-app addons-upgrade <addon-id> postgresql-business-1024

# Machine-readable output (global flag; not every command supports it yet)
scalingo --format json --app my-app apps-info
```

For the full command surface, see `references/cli-reference.md`.

## Environment variables

All configuration belongs in env (12-factor). Scalingo injects addon connection strings automatically — e.g. attaching PostgreSQL gives you `SCALINGO_POSTGRESQL_URL` and aliases it to `DATABASE_URL`. Your app reads these at runtime.

```bash
scalingo --app my-app env-set DJANGO_SETTINGS_MODULE=myapp.settings.prod
```

Changing env variables does **not** trigger a restart. Batch your changes, then `scalingo restart`.

For multi-line values (TLS keys, JSON blobs), either quote carefully or Base64-encode and decode in-app — the dashboard won't render multi-line values cleanly. Limits: names up to 64 characters, values up to 32,768 characters.

## Addons in one minute

```bash
# Discover what's available and see plans
scalingo addons-list
scalingo addons-plans postgresql

# Attach
scalingo --app my-app addons-add postgresql postgresql-starter-512

# Connect from a one-off (uses injected env vars)
scalingo --app my-app pgsql-console
scalingo --app my-app mongo-console
scalingo --app my-app redis-console
```

Main engines: PostgreSQL, MySQL, MongoDB, Redis (marketed as "Scalingo for Caching"; the addon ID is still `redis`), OpenSearch, InfluxDB. Elasticsearch is being phased out in favour of OpenSearch, and Sandbox plans can no longer be created. Plans come in service classes: Starter (single node, 98% SLA), Business (multi-node HA, 99.96%), and Enterprise (99.99%, PostgreSQL on Dedicated Resources only, in preview). Always confirm plan IDs with `scalingo addons-plans <addon>` before using them. Details and sizing guidance: `references/addons-databases.md`.

## scalingo.json (app manifest)

An optional file at repo root that declares how an app should be built when created from a one-click deploy button or as a review app. Minimal shape:

```json
{
  "name": "My App",
  "env": {
    "SECRET_KEY": { "generator": "secret" },
    "PUBLIC_URL": { "generator": "url" }
  },
  "formation": {
    "web":    { "amount": 1, "size": "M" },
    "worker": { "amount": 1, "size": "M" }
  },
  "addons": [
    { "plan": "postgresql:postgresql-starter-512" },
    { "plan": "redis:redis-starter-256" }
  ],
  "scripts": {
    "first-deploy": "bundle exec rake db:migrate db:seed"
  }
}
```

Triggers for this file: review apps, one-click deploy buttons. It is read only when such an app is created, and **not** applied on ordinary `git push` deploys — set the production formation/addons explicitly once. See `references/manifest-review-apps.md`.

## Scaling and the autoscaler

Two dimensions:

- **Horizontal** — `scalingo scale web:3:M` (three M-sized web containers). Load is balanced across them.
- **Vertical** — `scalingo scale web:1:XL` (upgrade the container tier). Sizes: S / M / L / XL / 2XL / 3XL; 3XL (8 GiB) is the largest and is self-service. Memory doubles per tier: 256 MiB → 512 MiB → 1 GiB → 2 GiB → 4 GiB → 8 GiB. By default a process type can run up to 10 containers.

Swap is being removed: new apps no longer get it since 2026-09-01, and all containers lose it on 2027-01-01. Size containers so the app fits in RAM.

The **Scalingo Autoscaler** adjusts horizontal scale automatically based on a target metric (`cpu`, `memory`, `swap`, `rpm_per_container`, `p95_response_time`, `5XX`). It needs a minimum of 2 containers, is keyed by container type (`autoscalers-add -c web -m cpu -t 0.75 --min-containers 2 --max-containers 10`), and cooldowns prevent flapping (1 min out, 3 min in). Full config and metric guidance: `references/scaling-autoscaler.md`.

## Logs and observability

- Live tail: `scalingo --app my-app logs -f`
- Router logs (HTTP-level) are off by default — enable in dashboard Routing settings
- Files rotate at 50 MiB to **Log Archives** (fetch via dashboard or CLI, paginated)
- Ingest caps: **64 MiB/min/container** and **16,384 lines/min/container** — excess lines are dropped for the rest of the minute
- Archived logs are kept for 1 year
- For retention beyond archives or for search/alerting, configure a **Log Drain** (syslog/HTTPS/ELK endpoint)

## Deployment options beyond git

When git-push isn't the right fit:

- **GitHub / GitLab integration** — link a repo, auto-deploy a branch, enable review apps per PR/MR
- **Archive deploy** — `scalingo --app my-app deploy my-app.tar.gz` (the archive must contain the code inside a single top-level folder, whatever its name). There is no deploy-from-Docker-image option
- **WAR** — `scalingo --app my-app deploy --war app.war` for a prebuilt Java WAR
- **One-click deploy button** — a URL that deploys a `scalingo.json`-equipped repo

Details and exact URL formats: `references/deployment.md`.

## Infrastructure as code (Terraform)

The official Scalingo Terraform provider (`Scalingo/scalingo`) covers apps (including their env vars), addons, container types, autoscalers, domains, SCM integrations, collaborators, log drains, alerts, projects, firewall rules, and Dedicated Resources databases. Useful when you need reproducible environments or multiple review/staging stacks.

```hcl
terraform {
  required_providers {
    scalingo = { source = "Scalingo/scalingo", version = "~> 2.0" }
  }
}

provider "scalingo" {
  region = "osc-fr1"   # or env: SCALINGO_REGION
  # api_token via env: SCALINGO_API_TOKEN
}

data "scalingo_stack" "s26" {
  name = "scalingo-26"
}

resource "scalingo_app" "api" {
  name     = "my-api"
  stack_id = data.scalingo_stack.s26.id   # stack ID, not the slug
  environment = {
    NODE_ENV = "production"
  }
}

resource "scalingo_addon" "db" {
  provider_id = "postgresql"
  plan        = "postgresql-starter-512"
  app         = scalingo_app.api.id
}
```

Full resource catalog and the opinionated community module (`scalingo-community/app/scalingo`): `references/terraform-iac.md`.

## Choosing a region

| Use case | Region | Notes |
|---|---|---|
| Default / general web apps | `osc-fr1` | Paris, Outscale infrastructure, ISO 27001 |
| French public sector, sensitive data, ANSSI compliance | `osc-secnum-fr1` | SecNumCloud-qualified Outscale infrastructure, stricter access controls, access via support |
| Health data (HDS) | `osc-fr1` + HDS-eligible plans | Certification covers hosting, not your app logic |

Set once: `scalingo config --region osc-secnum-fr1`. Each CLI call also accepts `--region` to override.

## Migrating from Heroku

Scalingo is intentionally compatible with the Heroku workflow: `Procfile`, buildpacks (mostly unchanged), `heroku run` → `scalingo run`, `app.json` → `scalingo.json` (same shape for the fields that matter). Heroku's `release:` Procfile line becomes `postdeploy:`. Typical migration steps:

1. Create the app on Scalingo and add its git remote alongside Heroku's
2. Provision matching addons (and copy data — `pg_dump`/`pg_restore` for PostgreSQL, etc.)
3. Copy env vars (`heroku config -s` → `scalingo env-set`)
4. Push; verify; cut over DNS

A common gotcha: Scalingo's git deploy accepts only `master` and `main` — map other branches with `local-branch:main` syntax.

## Anti-Patterns to Avoid

- **New process type doesn't start after deploy** — every process type except `web` defaults to 0. Run `scalingo scale <name>:1:M`.
- **`env-set` seemed to have no effect** — env changes don't auto-restart. Run `scalingo restart`.
- **Push rejected** — Scalingo git only accepts `master` or `main`. Use `git push scalingo mybranch:main`.
- **Attached one-off container disconnects** — it is closed after 30 minutes with no data exchanged, and any one-off is killed after 1 hour with no output. Use `--detached` and log progress for long jobs. If it never connects, check that your network allows the outbound connection the CLI opens.
- **Migrations put in a `release:` line never run** — Scalingo has no `release` type; use `postdeploy:`.
- **Build works locally but fails on Scalingo** — the default stack is `scalingo-26` (Ubuntu 26.04); `scalingo-24` is also supported and `scalingo-22` is deprecated (shut down 2027-06-01). Native dependencies compiled on your Mac won't transfer; the buildpack rebuilds them from source. Pin versions in lockfiles.
- **Logs suddenly quiet** — you may have hit the 64 MiB/min/container ingest cap. Reduce verbosity or add containers.
- **Review app has production credentials** — by default review apps inherit parent env. Override in `scalingo.json` (see `references/manifest-review-apps.md`).

## Compliance quick facts

Scalingo holds **ISO 27001:2022** (information security) and **HDS 2.0** (French health data hosting). The `osc-secnum-fr1` region runs on Outscale infrastructure that is **SecNumCloud-qualified** (ANSSI's qualification for trusted cloud providers). Scalingo's own PaaS layer is not SecNumCloud-qualified yet; its qualification is in progress, so don't describe a Scalingo deployment as "SecNumCloud-qualified" end to end. Hosting is in France, under European jurisdiction.

Certification applies to the platform; your application's data handling is still your responsibility.

---

When working on a Scalingo task, check the relevant reference before answering from memory — addon plan names, Terraform resource schemas, and autoscaler metric names change over time, and the references carry the current specifics.

## When to Use / Not Use

**Use this skill when:**

- Deploying or operating an app on Scalingo (git push, GitHub/GitLab integration, archives)
- Provisioning managed addons (PostgreSQL, MySQL, MongoDB, Redis, OpenSearch, InfluxDB)
- Scaling containers, configuring the autoscaler, or setting up review apps / `scalingo.json`
- Managing infrastructure with the Scalingo Terraform provider
- Migrating from Heroku or targeting French compliance (HDS, `osc-secnum-fr1` on SecNumCloud infrastructure)

**Do NOT use this skill when:**

- Deploying to a different platform — Vercel, Railway, or plain Kubernetes are out of scope
- The task is purely local development with no deployment component
- The user wants a generic Docker orchestration answer that isn't Scalingo-specific — prefer the platform they're actually on
