# Backend Optimization Deep Dive

## Reading EXPLAIN ANALYZE

Run `EXPLAIN ANALYZE` (Postgres; MySQL: `EXPLAIN FORMAT=TREE`). Read from the innermost
node up. What to look for:

| Plan element | Meaning | Action |
|--------------|---------|--------|
| `Seq Scan` on a big table | Full table read | Index the filter columns |
| `rows` estimate >> actual | Planner misestimating | Refresh stats; missing index |
| `Nested Loop` with inner scan per row | N+1 inside SQL | Rewrite as JOIN / `LATERAL` |
| `Sort` after an index | ORDER BY not covered | Composite index with sort column |
| `actual time` >> `cost` | Real work far above estimate | Check correlated filters, stats |

Composite index column order: equality columns first, then range/ORDER BY columns:

```sql
-- WHERE customer_id = ? AND created_at > ? ORDER BY created_at
CREATE INDEX idx_customer_created ON orders (customer_id, created_at);
```

## Index types

| Index | Use for |
|-------|---------|
| B-tree (default) | Equality, range, ORDER BY |
| Composite | Multi-column filters; leftmost-prefix order matters |
| Partial (`WHERE deleted_at IS NULL`) | Filtering most rows out of the index |
| Covering (`INCLUDE` columns) | Index-only scans — no heap fetch |
| GIN | JSONB, arrays, full-text |
| BRIN | Huge append-only tables (time series) |

Every index costs writes — profile first, index second, and keep the count proportional to
real query patterns.

## N+1 fixes by ORM

| ORM | Fix |
|-----|-----|
| SQLAlchemy | `.options(selectinload(Model.relation))` or `joinedload()` |
| Django | `select_related("fk")` (JOIN) / `prefetch_related("m2m")` (batch) |
| Prisma | `include: { items: true }` in one query |
| Sequelize | `include: [...]` (separate queries by default — check `logging`) |
| Raw SQL | One JOIN or `IN (...)` instead of a loop of queries |

Verify with query logging (`echo=True`, Prisma `log: ["query"]`) and alert when queries
per request grow.

## Caching patterns

### Cache-aside (lazy population)

```text
GET key  -> hit: return; miss: read DB, SET key WITH TTL, return
WRITE    -> invalidate (DELETE) key, then write DB
```

Delete-on-write avoids the race of caching a stale value. Read-your-writes is best-effort —
acceptable when the TTL is short.

### Stampede protection (single-flight)

```python
def get_with_single_flight(key, loader, ttl=60):
    value = redis.get(key)
    if value is not None:
        return json.loads(value)
    with redis.lock(f"lock:{key}", timeout=5):   # one loader wins
        value = redis.get(key)                    # re-check inside the lock
        if value is None:
            value = loader()
            redis.setex(key, ttl, json.dumps(value))
    return json.loads(value)
```

### Write-through / write-behind

- **Write-through** — update the cache on every write. Simple reads, but writes cost more
  and cold caches still hit the DB.
- **Write-behind (async)** — acknowledge, then batch-write (e.g., Redis `INCR` flushed to
  the DB on a schedule). Fast writes but a data-loss window — only for counters/aggregates.

### Redis operational notes

- `maxmemory-policy allkeys-lru` for a bounded cache.
- `SETEX` (SET with TTL) is atomic; `SET` then `EXPIRE` is two commands — a crash between
  them leaves a permanent key.
- Cache the serialized payload (JSON), so hits never pay for serialization.

## Connection pooling

- Bounded pools everywhere: DB `pool_size`/`max_overflow`, HTTP `maxSockets`, pgbouncer
  `default_pool_size`.
- Sizing rule of thumb: pool ≈ cores x expected concurrent I/O per worker (often 10-30 for
  a typical web worker). Oversized pools exhaust DB connections under load spikes.
- Pool exhaustion shows up as *waiting* latency, not query time — watch pgbouncer
  `client_waiting` / pool stats.

## Batching external calls

Serialize → parallelize → batch:

```python
# 1. Serial (baseline) — 3x latency
# 2. Parallel with asyncio.gather / ThreadPoolExecutor — 1x latency
# 3. Batch endpoint (POST /bulk) — fewest round trips, amortized cost
```

Cap parallel fan-out with a semaphore (one caller must not open 10k sockets) and set
per-call timeouts — a hung dependency must not hang the whole request.

## Keyset pagination

```sql
-- OFFSET (slow: re-scans skipped rows)
SELECT * FROM orders ORDER BY id LIMIT 20 OFFSET 10000;

-- Keyset (fast: uses the index, stable under writes)
SELECT * FROM orders WHERE id > 10020 ORDER BY id LIMIT 20;
```

Keyset requires a unique sort key and a "next" cursor in the API response.
