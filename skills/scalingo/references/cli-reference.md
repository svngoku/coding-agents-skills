# Scalingo CLI Reference

Complete command map for the `scalingo` CLI. Install with:

```bash
curl -O https://cli-dl.scalingo.com/install && bash install
```

All commands that act on an app accept `--app <name>` or `-a <name>`. Commands that act across regions accept `--region <region>` (or use the default set by `scalingo config --region`).

## Table of contents

1. [Global](#global)
2. [Apps](#apps) (incl. [Projects and private networks](#projects-and-private-networks))
3. [Deployment and Git](#deployment-and-git)
4. [Process types and scaling](#process-types-and-scaling)
5. [One-off containers](#one-off-containers)
6. [Environment variables](#environment-variables)
7. [Addons](#addons)
8. [Database helpers](#database-helpers)
9. [Domains and TLS](#domains-and-tls)
10. [Logs](#logs)
11. [Collaborators and access](#collaborators-and-access)
12. [SCM integrations and review apps](#scm-integrations-and-review-apps)
13. [Events and audit](#events-and-audit)
14. [Stats and metrics](#stats-and-metrics)
15. [Addon configuration](#addon-configuration)
16. [Firewall, output format, and preview features](#firewall-output-format-and-preview-features)

## Global

```bash
scalingo login                       # Browser-based login
scalingo logout
scalingo self                        # Show the logged-in user
scalingo regions                     # List regions (osc-fr1, osc-secnum-fr1, ...)
scalingo config --region osc-fr1     # Set default region
scalingo help <command>              # Per-command help
scalingo --version
```

Environment-based auth for CI:

```bash
export SCALINGO_API_TOKEN=tk-us-...
export SCALINGO_REGION=osc-fr1
```

Generate an API token in the dashboard: Account Settings → API Tokens.

## Apps

```bash
scalingo apps                        # List your apps in the current region
scalingo create <name>               # Create an app (validates name uniqueness)
scalingo create <name> --project-id <project-id>   # Create inside a project
scalingo --app <name> stacks-set scalingo-26       # Pin a stack (there is no `create --stack`)
scalingo rename --app old --new-name new
scalingo destroy --app <name>        # Delete (requires typing the app name)
scalingo --app <name> apps-info      # Metadata (region, stack, status)
```

`create` only accepts `--remote`, `--buildpack`, `--project-id`, and `--hds-resource`. Set the stack afterwards with `stacks-set` (available stacks: `scalingo stacks`).

## Projects and private networks

Projects group apps; Private Networks are attached to a project and let its apps reach each other without going through the public internet.

```bash
scalingo projects                    # List projects (see `scalingo help` for the projects-* family)
scalingo --app <name> project-set <project-id>     # Move an app into a project
scalingo --app <name> private-networks-domain-names   # Private DNS names of the app's containers
```

## Deployment and Git

```bash
scalingo --app <name> git-show       # Show the git remote URL for the app
scalingo --app <name> git-setup      # Add the `scalingo` remote to local repo

# Archive deploy: local tar.gz (or WAR) path, or a URL reachable from the platform.
# The archive must contain the code inside a single top-level folder (any name).
scalingo --app <name> deploy my-app.tar.gz
scalingo --app <name> deploy <https://.../archive.tar.gz>
scalingo --app <name> deploy <https://.../archive.tar.gz> <git-ref>

# Reset build cache (forces buildpack to run fresh)
scalingo --app <name> deployment-delete-cache

# Deployments list and status
scalingo --app <name> deployments
scalingo --app <name> deployment-follow <deployment-id>
```

## Process types and scaling

Scale uses the syntax `<process>:<amount>:<size>`. Size is optional and keeps the current value.

```bash
scalingo --app <name> ps             # Show formation
scalingo --app <name> scale web:2:L              # 2 web containers at L
scalingo --app <name> scale worker:1:M web:3     # Multiple types at once
scalingo --app <name> scale worker:0              # Stop a process type
scalingo --app <name> restart                    # All containers
scalingo --app <name> restart web                # One process type
```

Container sizes (memory / CPU priority):

| Size | Memory | CPU priority |
|------|--------|--------------|
| S    | 256 MB | low          |
| M    | 512 MB | standard (default) |
| L    | 1 GB   | standard     |
| XL   | 2 GB   | high         |
| 2XL  | 4 GB   | high         |
| 3XL  | 8 GB   | high         |

3XL is the largest size and is self-service. A process type can run up to 10 containers by default.

Autoscaler commands are keyed by container type, not by an ID:

```bash
scalingo --app <name> autoscalers
scalingo --app <name> autoscalers-add -c web -m cpu -t 0.75 --min-containers 2 --max-containers 10
scalingo --app <name> autoscalers-update -c web -t 0.6
scalingo --app <name> autoscalers-disable web
scalingo --app <name> autoscalers-enable web
scalingo --app <name> autoscalers-remove web
```

See `scaling-autoscaler.md` for metric names and targets.

## One-off containers

One-offs are ephemeral copies of your production environment. The number of concurrent one-offs is limited per app and per account.

```bash
# Attached (interactive, closed after 30 min with no data exchanged)
scalingo --app <name> run bash
scalingo --app <name> run --size XL python manage.py migrate
scalingo --app <name> run --env DEBUG=1 --env FOO=bar bash

# Detached (runs without a terminal; you must ensure it exits)
scalingo --app <name> run --detached long-job.sh

# List and stop detached one-offs
scalingo --app <name> ps
scalingo --app <name> one-off-stop <one-off-id>
```

Any one-off (attached or detached) is killed after 1 hour without producing output, so long jobs should log progress.

## Environment variables

```bash
scalingo --app <name> env            # Show all (values visible)
scalingo --app <name> env-get KEY    # Single value
scalingo --app <name> env-set KEY=value KEY2=value2
scalingo --app <name> env-unset KEY KEY2
```

Env changes do **not** restart the app — batch and then `scalingo restart`.

Limits: 64 chars for name, 32,768 chars for value.

For multi-line secrets, either quote carefully on the CLI or Base64-encode:

```bash
scalingo --app <name> env-set "PRIVATE_KEY=$(cat key.pem | base64 -w 0)"
# then decode in-app
```

## Addons

```bash
# Discovery
scalingo addons-list                            # List all available addon providers
scalingo addons-plans <provider-id>              # Plans for a given provider

# Lifecycle
scalingo --app <name> addons                     # List attached addons
scalingo --app <name> addons-add <provider> <plan>
scalingo --app <name> addons-info <addon-id>
scalingo --app <name> addons-upgrade <addon-id> <new-plan>
scalingo --app <name> addons-remove <addon-id>

# Example: attach a PostgreSQL database
scalingo --app my-api addons-add postgresql postgresql-starter-512
```

Attached addon IDs look like `ad-<uuid>`. Connection info is injected as env (e.g. `SCALINGO_POSTGRESQL_URL`), with `DATABASE_URL` set as an alias for single-database setups.

## Database helpers

Built-in REPL one-offs. They figure out credentials from env and open the right client:

```bash
scalingo --app <name> pgsql-console
scalingo --app <name> mysql-console
scalingo --app <name> mongo-console      # uses mongosh since CLI 1.50.0
scalingo --app <name> redis-console
scalingo --app <name> influxdb-console
scalingo --app <name> valkey-console
```

Inside a one-off `bash`, install CLI tools for your addon with:

```bash
dbclient-fetcher postgresql
dbclient-fetcher mysql 8.4
dbclient-fetcher mongo
dbclient-fetcher redis
```

Backup management (database commands take the addon with `--addon <addon-id>`):

```bash
scalingo --app <name> --addon <addon-id> backups              # List database backups
scalingo --app <name> --addon <addon-id> backups-download --backup <backup-id>
scalingo --app <name> --addon <addon-id> backups-create       # Manual backup
scalingo --app <name> --addon <addon-id> backups-config --schedule-at 3   # Daily backup hour
scalingo --app <name> --addon postgresql database-pitr-recovery-window   # Restorable time range
scalingo --app <name> --addon postgresql database-pitr-restore 2026-07-18T23:00:00Z   # PITR (RFC 3339 time)
```

 Plans can be listed with `scalingo database-list-plans`. For Dedicated Resources databases (not attached to an app), use `--database <database-id>` instead of `--app`/`--addon`.

## Domains and TLS

```bash
scalingo --app <name> domains                    # List
scalingo --app <name> domains-add example.com    # Add; Let's Encrypt cert auto-issued
scalingo --app <name> domains-remove example.com
scalingo --app <name> domains-ssl --cert cert.pem --key key.pem example.com   # Bring your own cert
scalingo --app <name> set-canonical-domain example.com   # Force redirects to this domain
scalingo --app <name> unset-canonical-domain
```

Let's Encrypt certificates are valid 45 days and renewed automatically. Every app has a default subdomain: `<name>.osc-fr1.scalingo.io` (or the equivalent for other regions). HTTPS is enabled by default and HTTP redirects to HTTPS automatically — disable via `FORCE_HTTPS=false` if needed.

## Logs

```bash
scalingo --app <name> logs                       # Recent
scalingo --app <name> logs -f                    # Stream
scalingo --app <name> logs --lines 1000          # More history
scalingo --app <name> logs --filter worker       # Filter by process type
scalingo --app <name> logs-archives              # List archived log files
scalingo --app <name> logs-archives --page 2
```

Log drains (forward to external systems):

```bash
scalingo --app <name> log-drains                 # List drains
scalingo --app <name> log-drains --with-addons   # Include addon drains
scalingo --app <name> log-drains-add --type syslog --host <host> --port <port>
scalingo --app <name> log-drains-add --type=elk --url=https://user:pass@host/path
scalingo --app <name> log-drains-add --type=datadog --url=<datadog-url>
scalingo --app <name> log-drains-remove <drain-url>

# Addon logs drain (database addons only; separate from app logs)
scalingo --app <name> log-drains-add --addon <addon-id> --type syslog --host <host> --port <port>
```

## Collaborators and access

```bash
scalingo --app <name> collaborators              # List
scalingo --app <name> collaborators-add email@example.com
scalingo --app <name> collaborators-remove email@example.com
```

Collaborators accept or decline by email. They can read/write to the app but can't destroy it unless granted ownership transfer. A **Limited Collaborator** role also exists for restricted access (set it in the dashboard).

## SCM integrations and review apps

Link an app to a GitHub/GitLab repo for auto-deploy and review apps:

```bash
# Link the Scalingo user account to the SCM provider first (one-time, in dashboard → Account → Integrations)
# Then link a specific app to a specific repo:
scalingo --app <name> integration-link-create \
  --auto-deploy --branch main \
  https://github.com/org/repo

scalingo --app <name> integration-link           # Show current link
scalingo --app <name> integration-link-update --auto-deploy --branch develop
scalingo --app <name> integration-link-delete
scalingo --app <name> integration-link-manual-deploy main

# Review apps
scalingo --app <name> integration-link-update --deploy-review-apps --destroy-on-close
scalingo --app <name> integration-link-manual-review-app <pull-request-number>
```

## Events and audit

```bash
scalingo --app <name> timeline       # Actions on this app
scalingo user-timeline               # Your actions across all apps
```

## Stats and metrics

```bash
scalingo --app <name> stats          # Live CPU / RAM per container
scalingo --app <name> ps             # Container list with size and status
```

Router metrics (RPM, response times, 5xx) are visible in the dashboard and used by the autoscaler.

## Addon configuration

Scalingo managed databases expose configurable features and users:

```bash
scalingo --app <name> --addon <addon-id> database-enable-feature force-ssl
scalingo --app <name> --addon <addon-id> database-disable-feature force-ssl
scalingo --app <name> --addon postgresql database-users-list
scalingo --app <name> --addon postgresql database-users-create --read-only <username>   # password prompted; blank = generated
scalingo --app <name> --addon postgresql database-users-update-password <username>
scalingo --app <name> --addon postgresql database-users-delete <username>
```

`--addon` accepts the addon ID or the addon type (e.g. `postgresql`) when the app has a single addon of that type. Features vary by engine — `force-ssl` (optional on every plan) and `publicly-available` are the common ones. PostgreSQL extensions such as `pg_stat_statements` are enabled with SQL (`CREATE EXTENSION`), not as features.

## Firewall, output format, and preview features

```bash
# Application Firewall (launched 2026-10): restrict inbound traffic to an app
scalingo --app <name> app-firewall-rules
scalingo --app <name> app-firewall-rule-add --cidr 203.0.113.42/32 --label office
scalingo --app <name> app-firewall-rule-remove <rule-id>

# Machine-readable output (global flag; not every command supports it yet)
scalingo --format json --app <name> apps-info

# Preview commands (Dedicated Resources databases, database firewall rules, net peerings, ...)
export SCALINGO_PREVIEW_FEATURES=true
scalingo help
```
