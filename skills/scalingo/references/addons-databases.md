# Addons and Managed Databases

Scalingo addons are managed services attached to an app. Each attachment injects connection credentials as env vars. The main category is managed databases; others include caching services, email relays, and APM integrations.

## Table of contents

1. [How addons work](#how-addons-work)
2. [PostgreSQL](#postgresql)
3. [MySQL](#mysql)
4. [MongoDB](#mongodb)
5. [Scalingo for Caching (Redis)](#scalingo-for-caching-redis)
6. [OpenSearch](#opensearch)
7. [InfluxDB](#influxdb) and [Elasticsearch (phasing out)](#elasticsearch-phasing-out)
8. [Backups](#backups)
9. [Service classes and sizing](#service-classes-and-sizing)
10. [Shared infrastructure vs dedicated](#shared-infrastructure-vs-dedicated)
11. [Architecture and networking](#architecture-and-networking)
12. [Migration and import](#migration-and-import)

## How addons work

Sandbox plans can no longer be created (since 2026-03-02); the smallest plan of each engine is now a Starter plan. Always confirm plan IDs with `scalingo addons-plans <provider>` before writing them into scripts, manifests, or Terraform.

Attaching an addon:

```bash
scalingo --app my-app addons-add <provider> <plan>
# e.g.
scalingo --app my-app addons-add postgresql postgresql-starter-512
```

On success:

- An addon ID (`ad-<uuid>`) is assigned
- Connection env vars are injected into the app (e.g. `SCALINGO_POSTGRESQL_URL`)
- For single-DB setups, `DATABASE_URL` is aliased to the primary DB URL
- The app is **not** restarted automatically — do so manually after attaching

Listing and inspecting:

```bash
scalingo --app my-app addons                      # list
scalingo --app my-app addons-info <addon-id>      # details + status
scalingo --app my-app addons-plans <provider>     # available plans
```

Upgrading/downgrading plans happens with minimal downtime on single-node plans and zero downtime on HA plans:

```bash
scalingo --app my-app addons-upgrade <addon-id> postgresql-business-1024
```

Removing is destructive — data is gone unless you've taken a backup:

```bash
scalingo --app my-app addons-remove <addon-id>
```

## PostgreSQL

Scalingo for PostgreSQL® — the most full-featured database on the platform.

Provider ID: `postgresql`. Connection env var: `SCALINGO_POSTGRESQL_URL`, also aliased to `DATABASE_URL`.

```bash
scalingo --app my-app addons-add postgresql postgresql-starter-512
```

Plan naming: `postgresql-starter-<RAM in MB>` (512, 1024, 2048, ...) and `postgresql-business-<RAM in MB>` (1024 and up).

Versions: PostgreSQL 14 to 18 are supported; new addons default to 17. PostgreSQL 14 reaches end of life on 2026-11-12, and remaining 14 databases are upgraded to 15 automatically on 2027-01-11 — upgrade on your own schedule before that.

Key features:

- Point-in-Time Recovery (PITR) on all plans (7-day window on shared resources)
- High availability on Business: 1 primary + 1 standby. The standby is for failover only and is **not** exposed for connections — there are no read-only replicas
- `pg_stat_statements`, `pg_trgm`, PostGIS available (enable extensions with `CREATE EXTENSION`)
- Major version upgrades via dashboard (one-click)
- TLS is available on every plan; forcing it is an optional toggle on any plan (`database-enable-feature force-ssl`)

Connect from a one-off:

```bash
scalingo --app my-app pgsql-console
# or inside `scalingo run bash`:
dbclient-fetcher postgresql
psql $SCALINGO_POSTGRESQL_URL
```

## MySQL

Provider ID: `mysql`. Env var: `SCALINGO_MYSQL_URL` (and `DATABASE_URL` alias for single-DB setups).

```bash
scalingo --app my-app addons-add mysql mysql-starter-512
```

MySQL 8.4 is the current version; remaining 8.0 databases were upgraded to 8.4 automatically on 2026-10-12. The `mysql-console` CLI one-off and `dbclient-fetcher mysql` work as for PostgreSQL.

## MongoDB

Provider ID: `mongodb`. Env var: `SCALINGO_MONGO_URL`, aliased to `MONGO_URL` and `MONGODB_URI` depending on app conventions.

```bash
scalingo --app my-app addons-add mongodb mongo-starter-512
```

Plans follow the same pattern; the smallest MongoDB plan is 256 MB. Business tier provides a 3-node replica set for HA. Since CLI 1.50.0, `mongo-console` uses `mongosh`.

Connect:

```bash
scalingo --app my-app mongo-console
# or
dbclient-fetcher mongo
mongosh $SCALINGO_MONGO_URL
```

## Scalingo for Caching (Redis)

Renamed from "Scalingo for Redis" to **Scalingo for Caching** in June 2025 for trademark reasons. The engine is still Redis OSS (7.2 by default, 6.2 available) and the provider ID is still `redis`. Env var: `SCALINGO_REDIS_URL`, often aliased to `REDIS_URL`.

```bash
scalingo --app my-app addons-add redis redis-starter-256   # smallest plan is 256 MB
```

Uses:

- Session store
- Job queue backend (Sidekiq, Celery, BullMQ)
- Cache layer
- Rate limiting

Business-tier plans run 3 nodes and provide persistent storage and replication. Starter plans are single-node and should not be the sole persistence layer for durable data.

```bash
scalingo --app my-app redis-console
```

## OpenSearch

Provider ID: `opensearch`. Env var: `SCALINGO_OPENSEARCH_URL` (check `scalingo env` after attaching). Launched in June 2025 as the replacement for Elasticsearch.

```bash
scalingo --app my-app addons-add opensearch opensearch-starter-1024   # smallest plan is 1 GB
```

Backups of OpenSearch (and Elasticsearch) databases can't be downloaded.

Use for full-text search, log analytics, or vector search (with the k-NN plugin on supported plans). Connect via HTTP using the URL directly — credentials are embedded.

## InfluxDB

Provider ID: `influxdb`. Env var: `SCALINGO_INFLUX_URL`.

Time-series data — metrics, IoT, sensor data. Connect with the InfluxDB client tooling using the URL and token from env, or `scalingo --app my-app influxdb-console`. InfluxDB is sold in the Starter class only (from 512 MB).

## Elasticsearch (phasing out)

Scalingo is gradually phasing out Elasticsearch in favour of OpenSearch. Don't provision new Elasticsearch addons; plan a migration to OpenSearch for existing ones.

## Backups

All managed databases take **daily automated backups**. Scheduled backup retention:

| Class | Daily | Weekly | Monthly |
|---|---|---|---|
| Starter | 7 | 4 | — |
| Business | 7 | 8 | 12 |

Manual backups are capped (10 on Starter, 50 on Business). After a database is deleted, its manual backups are kept 1 month (Starter) or 1 year (Business).

```bash
scalingo --app my-app --addon <addon-id> backups                  # list
scalingo --app my-app --addon <addon-id> backups-create           # trigger one
scalingo --app my-app --addon <addon-id> backups-download --backup <backup-id>
scalingo --app my-app --addon <addon-id> backups-config --schedule-at 3   # daily backup hour
```

For PostgreSQL, **Point-in-Time Recovery** is available on every plan. The recovery window is 7 days on shared resources (7/14/30 days for Starter/Business/Enterprise on Dedicated Resources). Restore from the dashboard or the CLI:

```bash
scalingo --app my-app --addon postgresql database-pitr-recovery-window
scalingo --app my-app --addon postgresql database-pitr-restore 2026-07-18T23:00:00Z
```

Export to external storage (S3, etc.) is your responsibility; scripts running as detached one-offs are a common pattern:

```bash
scalingo --app my-app run --detached bash -c 'pg_dump $DATABASE_URL | gzip | aws s3 cp - s3://bucket/backup.sql.gz'
```

## Service classes and sizing

Three service classes:

| Class | Architecture | SLA | Use case |
|---|---|---|---|
| Starter | Single node | 98% | Dev, staging, small prod |
| Business | PostgreSQL: 1 primary + 1 standby; Caching, MongoDB, OpenSearch: 3 nodes | 99.96% | Production with HA requirements |
| Enterprise | 1 primary + 2 standbys | 99.99% | Critical workloads (Dedicated Resources PostgreSQL only) |

Dedicated Resources is not a class but an architecture model (see below); it offers Starter, Business, and Enterprise.

All tiers are managed (no OS access, no manual patching). Billing is **per minute** based on the current plan.

Sizing starts at 256 MB (Caching, MongoDB), 512 MB (PostgreSQL, MySQL, InfluxDB), or 1 GB (OpenSearch). Public pricing for shared resources goes up to 128 GB (PostgreSQL Business); larger sizes are on request. For guidance, ask:

1. Working-set size — does it fit in RAM? That's the primary memory metric.
2. Connection count — PostgreSQL's pgBouncer availability depends on plan.
3. Write throughput — Starter's IO is best-effort; Business allocates dedicated IOPS.

Switch classes in minutes — moving from a single-node Starter to a multi-node Business plan (or back) is a supported upgrade path without downtime on HA-capable tiers.

## Shared infrastructure vs dedicated

- **Shared Resources** — dedicated *instance* (your own engine process, isolated data) on shared underlying infrastructure. This is the default for all Starter and Business plans. Suitable from dev through production.
- **Dedicated Resources** — dedicated VMs and a dedicated VPC spread over up to 3 availability zones. PostgreSQL only, still in **preview** and available on request. A Dedicated Resources database is not attached to an app: only the owner can create it, it can only be scaled up, and logical backups are the customer's responsibility. CLI commands use `--database <id>` instead of `--app`/`--addon` and need `SCALINGO_PREVIEW_FEATURES=true`; Terraform uses `scalingo_database`. Outscale Net Peering is supported.

## Architecture and networking

- All addons run in the same region as their app for low latency
- Connections go through an addon-specific hostname (embedded in the URL)
- For most engines, plain TCP over the public internet is supported but TLS is available and recommended
- Forcing TLS is opt-in on every plan (`force-ssl` feature)
- No private networking between app and addon in Starter tier — traffic is internal to the region but not on a private VLAN
- Dedicated Resources tier provides private networking

## Migration and import

### Into Scalingo

For PostgreSQL, the canonical path is `pg_dump`/`pg_restore`. From a machine with network access to both source and destination:

```bash
# Dump from source (Heroku example)
pg_dump $(heroku config:get DATABASE_URL -a old-app) > dump.sql

# Restore to Scalingo
scalingo --app new-app pgsql-console < dump.sql
# or via a one-off:
scalingo --app new-app run --size XL bash
# inside: curl dump-url | psql $DATABASE_URL
```

For MongoDB: `mongodump` / `mongorestore`. For MySQL: `mysqldump` / `mysql`. The `dbclient-fetcher` helper makes these tools available inside one-off containers.

### Out of Scalingo

Backups are downloadable as archive files; contents are standard dumps compatible with the engine's native restore tools.

## Log drains for addon logs

Addon logs (query logs, slow queries, replication events) can be drained separately from app logs:

```bash
scalingo --app my-app log-drains-add \
  --addon <addon-id> \
  --type syslog \
  --host logs.example.com --port 6514

scalingo --app my-app log-drains --with-addons   # list app and addon drains
```

Only database addons produce logs.

Useful for compliance — sending PostgreSQL audit logs to a long-term store.

## Non-database addons

The platform supports various non-database addons (Mailgun, Papertrail, Sentry, Redsmin, Bugsnag, and others). They're discovered via:

```bash
scalingo addons-list
```

Attach and use env vars the provider defines. For details on any specific one, use `addons-plans <provider>` and the provider's own docs.
