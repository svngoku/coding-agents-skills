# Indexing and Query Tuning

How B-trees work, how to design composite/covering/partial indexes, and how to read EXPLAIN output.

## Table of contents

1. [How a B-tree works](#how-a-b-tree-works)
2. [Index types](#index-types)
3. [Composite index column order](#composite-index-column-order)
4. [Covering indexes](#covering-indexes)
5. [Partial indexes](#partial-indexes)
6. [When an index hurts](#when-an-index-hurts)
7. [Reading EXPLAIN](#reading-explain)
8. [Finding unused and bloated indexes](#finding-unused-and-bloated-indexes)
9. [Tuning workflow](#tuning-workflow)

## How a B-tree works

A B-tree is a balanced, ordered multi-way tree. Internal nodes route lookups; leaf nodes hold (key, row pointer) pairs in sorted order, linked to each other so range scans walk the leaves without backtracking.

- Lookup cost is O(log n); range scans (`BETWEEN`, `>`, `ORDER BY`) are efficient because leaves are sorted and linked.
- B-tree is the default index type in PostgreSQL and MySQL/InnoDB — use it for equality, range, and ordering.
- **Random-key inserts** (random UUIDs, hashed values) cause page splits and index bloat; sequential keys (bigint identity, UUIDv7) keep pages packed.

## Index types

| Type | Purpose | Example predicate |
|---|---|---|
| B-tree (default) | Equality, range, ORDER BY | `price BETWEEN 10 AND 50` |
| Hash | Equality only, exact matches | `md5 = 'abc...'` |
| GIN | Arrays, JSONB, full-text membership | `data @> '{"color":"red"}'::jsonb`, `to_tsvector(...)` |
| GiST | Geometry, ranges, exclusion constraints | `range && other_range`, `ST_DWithin(...)` |
| BRIN | Huge tables with naturally ordered columns | timestamps on append-only logs |

## Composite index column order

Rules:

1. **Equality columns first** — they shrink the search space immediately.
2. Then **one range or ORDER BY column** — the index can stop there.
3. Columns *after* a leading range column are unusable for lookups (only for filtering already-fetched rows).

Worked example:

```sql
SELECT * FROM orders
WHERE customer_id = 42 AND created_at >= '2025-01-01'
ORDER BY created_at;
```

- Index `(customer_id, created_at)`: equality on customer, then range + order on created_at — one index serves the whole query.
- Index `(created_at, customer_id)`: range column first means customer_id can't use the index — the query scans all orders since the date and filters. Wrong.

**Indexes can also serve `ORDER BY` alone** (no WHERE) if the leading columns match the sort order: `(created_at DESC)` supports `ORDER BY created_at DESC LIMIT 20` without a Sort node.

## Covering indexes

A covering index contains every column the query needs, so the engine answers from the index alone (index-only scan) and never touches the table heap:

```sql
CREATE INDEX idx_orders_customer_total
  ON orders (customer_id) INCLUDE (total_cents, status);

-- Index-only scan:
SELECT total_cents, status FROM orders WHERE customer_id = 42;
```

Tradeoff: `INCLUDE` columns widen the index (more storage, more write cost) and are not used for lookups or ordering — only to avoid the heap fetch. Add them when a hot query reads a few extra columns.

## Partial indexes

An index with a `WHERE` clause covers only matching rows:

```sql
CREATE INDEX idx_orders_pending ON orders (created_at)
  WHERE status = 'pending';
```

Benefits: the index is tiny, scans are fast, and writes only maintain the subset. The planner uses it when the query's predicates imply the index predicate. Classic uses: queues, active-flag filters, tenancy scoping (`WHERE tenant_id = ...` on a filtered set).

## When an index hurts

- **Write amplification:** every INSERT/UPDATE/DELETE maintains every index on the table. Five indexes mean roughly five extra structures updated per write — a real cost on write-heavy tables.
- **Disk and cache pressure:** indexes compete with table data for buffer cache; wide indexes can push hot data out.
- **Unused indexes are pure overhead:** still maintained on every write, never read. See below for detection queries.
- **Bloat:** heavy random-key inserts and updates leave dead index entries; autovacuum/REINDEX reclaims them.
- **Planner variance:** an index only helps if the planner uses it — small tables often get sequential scans even with indexes. Always verify with EXPLAIN.

## Reading EXPLAIN

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders WHERE customer_id = 42;
```

```text
Index Scan using idx_orders_customer on orders  (cost=0.29..8.31 rows=1 width=64) (actual time=0.045..0.049 rows=1 loops=1)
  Index Cond: (customer_id = 42)
  Buffers: shared hit=4
```

Read it as: an index scan was used; `cost` is a planner estimate (startup..total, arbitrary units); `rows` is the estimate. With `ANALYZE`, `actual time`/rows are measured. A large gap between estimated and actual rows usually means stale statistics — run `ANALYZE`.

Node cheat sheet:

| Node | Meaning | Good sign |
|---|---|---|
| Seq Scan | full table scan | only on small tables |
| Index Scan | targeted lookup via index | desired for selective predicates |
| Index Only Scan | answered entirely from the index | covering index working |
| Bitmap Heap Scan | index → bitmap → heap (many matches) | better than Seq Scan at scale |
| Nested Loop | for each outer row, probe inner | fine with a small driving set and an indexed inner side |
| Hash Join | hash one side, probe the other | fine for medium equi-joins |
| Merge Join | merge two sorted inputs | sorted inputs (often from indexes) |
| Sort | explicit sort step | consider an index that matches the ORDER BY |

## Finding unused and bloated indexes

PostgreSQL:

```sql
-- Unused indexes (low idx_scan = candidates for removal)
SELECT s.relname AS table,
       i.relname AS index,
       s.idx_scan
FROM pg_stat_user_indexes s
JOIN pg_class i ON i.oid = s.indexrelid
WHERE s.idx_scan = 0
ORDER BY s.relname;

-- Bloat / size
SELECT relname, pg_size_pretty(pg_relation_size(relid))
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(relid) DESC;
```

Removal policy: drop indexes with zero scans over a full traffic cycle (a week in production), then re-measure. For bloated indexes on busy tables, `REINDEX INDEX CONCURRENTLY` (PostgreSQL 12+) avoids blocking writes. MySQL: `sys.schema_unused_indexes` and `ALTER TABLE ... DROP INDEX`.

## Tuning workflow

1. **Find the slow query** — pg_stat_statements, slow query log, or an alert.
2. **EXPLAIN (ANALYZE, BUFFERS)** it against realistic data volumes.
3. **Spot the problem node** — typically a Seq Scan on a large table or a Sort.
4. **Add an index matching the predicates**, applying the composite-order rules.
5. **Re-EXPLAIN** and measure actual time.
6. **Remove indexes that never get used** — they cost writes for nothing.
