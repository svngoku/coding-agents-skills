# Migrations and Zero-Downtime Schema Changes

Versioned migrations, the expand–contract pattern, backfilling, and the locking gotchas that make schema changes the riskiest part of a deploy.

## Table of contents

1. [Versioned migrations](#versioned-migrations)
2. [Tool comparison](#tool-comparison)
3. [Expand–contract in detail](#expand-contract-in-detail)
4. [Backfilling strategies](#backfilling-strategies)
5. [Locking and deployment gotchas](#locking-and-deployment-gotchas)

## Versioned migrations

Every schema change is a versioned file applied exactly once, in order. The tool tracks applied versions in a bookkeeping table (`alembic_version`, `_prisma_migrations`, `flyway_schema_history`).

Rules:

- Write both an upgrade (forward) and downgrade (backward) path where the tool supports it.
- **Never edit an already-applied migration** — append a new one. Applied migrations are historical record; rewriting them makes environments diverge.
- Run migrations in CI/test environments and automatically at deploy time; never hand-apply DDL in production.
- Keep migrations small and single-purpose — one logical change per migration makes rollback and bisection possible.

## Tool comparison

| Tool | Language | Style | Typical commands |
|---|---|---|---|
| Alembic | Python | autogenerate from SQLAlchemy models | `alembic revision --autogenerate -m "..."`, `alembic upgrade head`, `alembic downgrade -1` |
| Prisma Migrate | TypeScript | schema-first; generates SQL | `prisma migrate dev --name "..."`, `prisma migrate deploy` |
| Flyway | Any (SQL/Java) | SQL files `V1__name.sql` | `flyway migrate`, `flyway info`, `flyway repair` |

Example Alembic migration:

```python
def upgrade():
    op.add_column("orders", sa.Column("total_cents", sa.BigInteger(), nullable=True))
    op.create_index("idx_orders_customer_created", "orders", ["customer_id", "created_at"])

def downgrade():
    op.drop_index("idx_orders_customer_created", table_name="orders")
    op.drop_column("orders", "total_cents")
```

Note that autogenerate is a draft, not ground truth — always review the generated DDL, especially around defaults, types, and index changes.

## Expand–contract in detail

Three phases, each independently deployable and reversible:

1. **Expand** — an additive, non-breaking change: new nullable column, new table, new index. Deploy app code that *writes both old and new*.
2. **Migrate** — backfill the new structure without long locks.
3. **Contract** — switch reads to the new shape, then switch writes, then drop the old shape once no deployed code references it.

### Example A: add a NOT NULL column to a live table

```sql
-- 1. Expand: add nullable (no default). Constant-default ADD COLUMN is fast on
--    PostgreSQL 11+; the expensive part is NOT NULL validation, so keep it separate.
ALTER TABLE orders ADD COLUMN region TEXT;

-- 2. Backfill in batches
UPDATE orders SET region = 'eu' WHERE id BETWEEN 1 AND 10000 AND region IS NULL;
-- ... repeat across ranges, or run as a background job

-- 3. Contract
ALTER TABLE orders ALTER COLUMN region SET NOT NULL;
```

`SET NOT NULL` scans the table to validate and takes a lock — run it only after the backfill proves complete, ideally in a low-traffic window. In MySQL 8, `ADD COLUMN` is instant-ish (ALGORITHM=INPLACE) but the NOT NULL validation scan still applies.

### Example B: rename a column

A native `RENAME COLUMN` breaks both old and new code simultaneously. Safer sequence:

1. Add `new_name`; app dual-writes; backfill `new_name` from `old_name`.
2. Switch reads to `new_name`.
3. Switch writes to `new_name` only.
4. After one or two full deploys of both old and new clients, drop `old_name`.

### Example C: change a column type

Same skeleton: add a new column of the new type, backfill with casts (bounded batches, watch for cast failures), flip reads/writes, drop the old column. Type changes (`INT` → `BIGINT`, `VARCHAR` → `TEXT`) usually need this treatment rather than in-place `ALTER ... TYPE`, which rewrites the table and locks it.

## Backfilling strategies

| Strategy | When | Notes |
|---|---|---|
| Single UPDATE | tiny tables (< ~100k rows) | fine, but batching is still safer |
| Batched UPDATE by key range | large tables | `WHERE id > :last_id ORDER BY id LIMIT 1000`, loop until done; persist progress |
| Batched by partition | time-partitioned data | backfill one partition at a time |
| Dual-write + trigger/CDC | writes continue during migration | a trigger or change-data-capture stream feeds the new column live |
| Online schema change tools | MySQL at scale | gh-ost, pt-online-schema-change |

Batching matters because a single unbounded UPDATE holds row locks (and possibly the table) for minutes and floods the WAL, stalling replication lag on replicas. Keep each batch small and log progress.

## Locking and deployment gotchas

- **Long transactions hold locks** — keep migrations short; never run backfills inside the migration transaction on a live table.
- **`ADD COLUMN` with a volatile default (`now()`)** rewrites the table in PostgreSQL (a constant default since v11 does not). Prefer nullable column + backfill + `SET NOT NULL`.
- **Index creation blocks writes** on some engines: use `CREATE INDEX CONCURRENTLY` on PostgreSQL and `ALGORITHM=INPLACE` (online DDL) on MySQL 8.
- **`SET NOT NULL` scans and locks** — backfill first, validate completeness, then apply in a low-traffic window.
- **Deploy ordering:** for expand–contract, deploy the new app code *before* the contract step. Additive DB changes are safe either way; the app change is the risky half. For pure additive changes, DB-first is fine.
- **Rollback plan:** the contract step must be reversible — keep the old column until old clients have fully rolled out, and keep the downgrade migration working.
- **Test against production-like data:** schema changes that are instant on a dev table (small, warm cache) can take hours on production volume. Measure, don't assume.
