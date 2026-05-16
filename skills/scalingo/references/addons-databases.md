# Addons and Managed Databases

Scalingo addons are managed services attached to an app. Each attachment injects connection credentials as env vars. The main category is managed databases; others include caching services, email relays, and APM integrations.

## Table of contents

1. [How addons work](#how-addons-work)
2. [PostgreSQL](#postgresql)
3. [MySQL](#mysql)
4. [MongoDB](#mongodb)
5. [Redis (caching)](#redis-caching)
6. [OpenSearch](#opensearch)
7. [InfluxDB](#influxdb)
8. [Backups](#backups)
9. [Service classes and sizing](#service-classes-and-sizing)
10. [Shared infrastructure vs dedicated](#shared-infrastructure-vs-dedicated)
11. [Architecture and networking](#architecture-and-networking)
12. [Migration and import](#migration-and-import)

## How addons work

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

Starter-tier plan naming: `postgresql-starter-<RAM>` (512, 1024, 2048, ...).
Business-tier: `postgresql-business-<RAM>` (single-digit GB).

Key features:

- Point-in-Time Recovery (PITR) on all plans
- Read-only replicas (Business tier)
- `pg_stat_statements`, `pg_trgm`, PostGIS available
- Major version upgrades via dashboard (one-click)
- SSL forced by default on Business plans; optional toggle on Starter

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

Available versions typically include MySQL 8.x; check `addons-plans mysql` for current options. The `mysql-console` CLI one-off and `dbclient-fetcher mysql` work as for PostgreSQL.

## MongoDB

Provider ID: `mongodb`. Env var: `SCALINGO_MONGO_URL`, aliased to `MONGO_URL` and `MONGODB_URI` depending on app conventions.

```bash
scalingo --app my-app addons-add mongodb mongo-starter-512
```

Plans follow the same pattern. Business tier provides replica sets for HA.

Connect:

```bash
scalingo --app my-app mongo-console
# or
dbclient-fetcher mongo
mongosh $SCALINGO_MONGO_URL
```

## Redis (caching)

Provider ID: `redis`. Env var: `SCALINGO_REDIS_URL`, often aliased to `REDIS_URL`.

```bash
scalingo --app my-app addons-add redis redis-starter-128
```

Uses:

- Session store
- Job queue backend (Sidekiq, Celery, BullMQ)
- Cache layer
- Rate limiting

Business-tier plans provide persistent storage and replication. Starter plans are single-node and should not be the sole persistence layer for durable data.

```bash
scalingo --app my-app redis-console
```

## OpenSearch

Provider ID: `opensearch`. Env var: `SCALINGO_OPENSEARCH_URL`.

```bash
scalingo --app my-app addons-add opensearch opensearch-starter-512
```

Use for full-text search, log analytics, or vector search (with the k-NN plugin on supported plans). Connect via HTTP using the URL directly — credentials are embedded.

## InfluxDB

Provider ID: `influxdb`. Env var: `SCALINGO_INFLUX_URL`.

Time-series data — metrics, IoT, sensor data. Connect with the InfluxDB client tooling using the URL and token from env.

## Backups

All managed databases take **daily automated backups**. Starter plans retain the last 10; Business plans retain 50. Manual backups:

```bash
scalingo --app my-app backups <addon-id>               # list
scalingo --app my-app backups-create <addon-id>        # trigger one
scalingo --app my-app backups-download <backup-id>     # fetch
scalingo --app my-app backups-config <addon-id>        # schedule
```

For PostgreSQL, **Point-in-Time Recovery** lets you restore to any moment within the retention window — configured from the dashboard. Restoration spawns a new addon attached to a new or existing app.

Export to external storage (S3, etc.) is your responsibility; scripts running as detached one-offs are a common pattern:

```bash
scalingo --app my-app run --detached bash -c 'pg_dump $DATABASE_URL | gzip | aws s3 cp - s3://bucket/backup.sql.gz'
```

## Service classes and sizing

Three tiers per engine, with distinct characteristics:

| Tier | Architecture | SLA | Use case |
|---|---|---|---|
| Starter | Single-node | 98% | Dev, staging, small prod |
| Business | Multi-node cluster | 99.96% | Production with HA requirements |
| Dedicated | Dedicated VM + private network | 99.99% | Critical / SecNumCloud workloads (PostgreSQL only, access on request) |

All tiers are managed (no OS access, no manual patching). Billing is **per minute** based on the current plan.

Sizing from 256 MiB up to 512 GiB of RAM depending on engine. For guidance, ask:

1. Working-set size — does it fit in RAM? That's the primary memory metric.
2. Connection count — PostgreSQL's pgBouncer availability depends on plan.
3. Write throughput — Starter's IO is best-effort; Business allocates dedicated IOPS.

Switch classes in minutes — moving from a single-node Starter to a multi-node Business plan (or back) is a supported upgrade path without downtime on HA-capable tiers.

## Shared infrastructure vs dedicated

- **Shared Resources** — dedicated *instance* (your own engine process, isolated data) on shared underlying infrastructure. This is the default for all Starter and Business plans. Suitable from dev through production.
- **Dedicated Resources** — dedicated VM, dedicated private network, SecNumCloud-ready architecture. Currently PostgreSQL-only and gated (contact Scalingo Support/Sales).

## Architecture and networking

- All addons run in the same region as their app for low latency
- Connections go through an addon-specific hostname (embedded in the URL)
- For most engines, plain TCP over the public internet is supported but TLS is either default or recommended
- PostgreSQL Business plans enforce SSL
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
scalingo --app my-app log-drains-add-addon \
  --addon <addon-id> \
  --type syslog \
  --url syslog+tls://logs.example.com:6514
```

Useful for compliance — sending PostgreSQL audit logs to a long-term store.

## Non-database addons

The platform supports various non-database addons (Mailgun, Papertrail, Sentry, Redsmin, Bugsnag, and others). They're discovered via:

```bash
scalingo addons-providers
```

Attach and use env vars the provider defines. For details on any specific one, use `addons-plans <provider>` and the provider's own docs.
