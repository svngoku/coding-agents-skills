# Task: Optimize the Slow GET /orders Endpoint

The workspace contains `slow.py` — a Flask handler for `GET /orders` that
returns a customer's orders with their line items. It is **deliberately slow**:
an N+1 query loop (one database call per order), no index on the filter and
lookup columns, no caching, and synchronous DB calls. Read it first and
identify the bottleneck before changing anything.

Produce **two files in the current workspace**:

1. `optimized.py` — a fixed version of the same endpoint.
2. `NOTES.md` — a short write-up of what was slow, what you changed, and how
   you would verify the fix with measurements.

There is no real database and no network: the grader statically inspects the
two files, so write self-contained, syntactically valid Python (you may reuse
the fake in-memory tables from `slow.py`). Do not edit `slow.py`.

## Requirements for `optimized.py`

Preserve the endpoint behavior — `GET /orders?customer_id=N` must still return
each order with its `items` list — and fix the performance:

1. **Remove the N+1** — no database lookup per order inside a loop. Load the
   line items with one batched query (e.g. `WHERE order_id IN (...)`), or
   eager-load them with an ORM (`selectinload` / `joinedload` /
   `select_related` / `include`).
2. **Add a caching layer** — cache-aside with a TTL (Redis `SETEX`, an
   `lru_cache`-style decorator, or an in-memory dict with expiry) so repeated
   hits for the same customer skip the database. Every cache entry must
   expire — no unbounded cache.
3. **No blocking sleeps** — keep the DB calls synchronous but do not add
   `time.sleep` anywhere in the request path.

## Requirements for `NOTES.md`

Write 20–50 lines covering:

1. **What was slow** — name the original bottleneck (the N+1, the missing
   index, the per-request DB fan-out).
2. **What you changed** — the batch/eager-load fix, the cache (key and TTL),
   and how each maps to the bottleneck.
3. **How you would verify with measurements** — the EXPLAIN plan you would
   expect before/after (e.g. `Seq Scan` → `Index Scan`) and a concrete
   before/after latency measurement (e.g. p95: 480 ms → 28 ms, or queries per
   request: 51 → 2). Numbers matter; no hand-waving.
