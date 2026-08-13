---
name: performance-optimization
description: >
  Systematically find and fix performance problems in applications, frontend and backend,
  by measuring first. Use this skill whenever the user reports slow endpoints, high latency,
  poor throughput, memory leaks, large bundles, or slow page loads; wants to profile with
  cProfile, py-spy, Chrome DevTools, Node --cpu-prof, or perf; run load tests with locust or
  k6; set performance budgets or SLOs; optimize database queries (EXPLAIN, indexes, N+1);
  add caching (Redis); improve frontend performance (code splitting, lazy loading, image
  optimization, memoization, virtualization); or tune the network (CDNs, gzip/br, HTTP/2/3).
  Also trigger for p50/p95/p99 latency, throughput, error rate, connection pooling, async vs
  threads, GIL, and memory leak investigations.
---

# Performance Optimization

Most "slow" problems are diagnosed by guesswork. This skill replaces guessing with a
measure-first workflow: profile, fix minimally, re-measure against a budget. Covers backend
(database, caching, concurrency), frontend (bundles, images, rendering), network (CDN,
compression, HTTP), and memory, with concrete tools and before/after examples.

## Quick Reference

| Topic | Reference |
|-------|-----------|
| Profiling tools per runtime and reading flame graphs | [profiling-tools.md](references/profiling-tools.md) |
| EXPLAIN, index design, N+1 fixes, caching, pooling | [backend-optimization.md](references/backend-optimization.md) |
| Bundles, images, React rendering, Core Web Vitals | [frontend-performance.md](references/frontend-performance.md) |

## Core Workflow

### 1. Measure First

Never optimize before you can answer: *"how do I know it got faster?"*

1. **Reproduce** — get a repeatable trigger (endpoint, page, script, load profile).
2. **Baseline** — record current latency distribution, throughput, and resource usage.
3. **Budget** — set an explicit target (e.g., p95 < 200 ms, initial JS < 170 KB gzipped).
4. **Profile** — find where time actually goes (CPU, I/O wait, dependency latency, GC).
5. **Hypothesize** — one bottleneck, one predicted fix, one measurable effect.
6. **Fix minimally** — the smallest change that tests the hypothesis.
7. **Re-measure** — same tool, load, and environment; keep the fix only if it moves the metric.

### 2. The Triage Loop (symptom → cause)

| Symptom | Likely cause | Quick check |
|---------|--------------|-------------|
| High p95 but healthy p50 | Queueing, GC pauses, one slow dependency | Percentile breakdown; GC logs; sample slow requests |
| Slow only under load | Contention, pool exhaustion, saturation | Load test at increasing concurrency; watch pool waits |
| Slow in prod, fast locally | Cold cache, missing index on prod-size data | EXPLAIN on prod data; cache hit ratio |
| ORM makes dozens of queries | N+1 lazy loading | Log query counts; enable ORM query logging |
| High CPU | CPU-bound loop, regex, serialization | cProfile / py-spy dump |
| Memory grows monotonically | Leak or unbounded cache | Heap snapshots; cache size/capacity audit |
| Slow page load, fast API | Render-blocking JS/CSS, oversized images | Lighthouse waterfall; DevTools Network |
| High TTFB | Slow backend or no CDN | `curl -w` timing breakdown |

## Choosing the Right Metrics

Track **percentiles, not just the mean** — a mean hides whether 5% of users wait 10x longer.

| Metric | What it tells you | Pitfall |
|--------|-------------------|---------|
| Mean latency | Average experience | Skewed by outliers; hides the long tail |
| p50 latency | Typical experience | Misses everything bad |
| p95 / p99 latency | Worst-case experience | Noisy at low traffic volume |
| Throughput (RPS) | Capacity | Meaningless without a latency target |
| Error rate | Stability under load | Track *alongside* latency, never alone |
| Concurrency (in-flight) | Queueing behavior | High values mean requests are piling up |

Turn metrics into **SLOs** (e.g., "99% of GET /api/orders < 300 ms", "error rate < 0.1%") and
load test with k6 or locust *before* production does:

```javascript
import http from "k6/http";
export const options = {
  stages: [{ duration: "30s", target: 20 }, { duration: "1m", target: 100 }],
  thresholds: { http_req_duration: ["p(95)<250"] },
};
export default function () { http.get("http://localhost:8000/api/orders/42"); }
```

## Backend Optimization

### Database

```sql
-- Red flags: Seq Scan on large tables, rows far higher than returned, missing indexes
EXPLAIN ANALYZE
SELECT o.*, i.* FROM orders o JOIN order_items i ON i.order_id = o.id
WHERE o.customer_id = 123 ORDER BY o.created_at DESC LIMIT 20;
```

| Technique | What it fixes | Cost |
|-----------|---------------|------|
| Index on filter columns | Full table scans | Slower writes; keep index count sane |
| Composite index (column order matters) | Multi-column filters | More storage |
| Covering index | Avoids fetching table rows | Storage |
| Keyset pagination (`WHERE id > ?`) | Slow `OFFSET` on deep pages | Cursor in the API response |
| Query batching / eager loading | N+1 round trips | More complex ORM code |
| Denormalization (cached counts) | Expensive aggregates on hot paths | Invariants to maintain |

N+1 — one query for the parent, one per child — is the most common backend perf bug:

```python
# BAD: 1 + N queries
for order in session.query(Order).filter_by(customer_id=123):
    for item in order.items:      # lazy load -> one query per order
        ...
# GOOD: 2 queries
from sqlalchemy.orm import selectinload
orders = session.query(Order).options(selectinload(Order.items)).filter_by(customer_id=123)
```

### Caching

**Cache-aside** (read-through on miss, delete on write) is the default pattern:

```python
def get_order(order_id: int):
    cached = redis.get(f"order:{order_id}")
    if cached is not None:
        return json.loads(cached)          # cache hit — no DB
    order = db.query(Order).get(order_id)  # miss — DB
    redis.setex(f"order:{order_id}", 60, json.dumps(order.to_dict()))  # TTL 60s
    return order
```

Rules that prevent cache bugs:

- **Set a TTL on everything** — short TTLs make correctness bugs self-heal.
- **Invalidate on write** — delete the key in the same flow that mutates the row.
- **Cap cache size** (LRU / `maxmemory-policy allkeys-lru`) — an unbounded cache is a leak.
- **Guard against stampedes** — single-flight lock so concurrent misses don't all hit the DB
  (example in [backend-optimization.md](references/backend-optimization.md)).

### Concurrency & I/O

- **Pool your connections** (DB, HTTP, Redis) — establishing connections is expensive; size
  pools to cores x expected concurrency (pgbouncer, explicit pool bounds).
- **Batch external calls** — fan out parallel requests instead of serial ones:

```python
# BAD: 3 round trips, serial — latency adds up
a = api_a.get(x); b = api_b.get(x); c = api_c.get(x)
# GOOD: 3 round trips, parallel — latency of the slowest
import asyncio
a, b, c = await asyncio.gather(api_a.get(x), api_b.get(x), api_c.get(x))
```

- **Async I/O beats threads for I/O-bound work**; threads or multiprocessing beat async for
  CPU-bound work (see Concurrency Models below).

## Frontend Optimization

### Bundle size

- **Code splitting** — load route/feature chunks on demand with `React.lazy` + `Suspense` or
  dynamic `import()`.
- **Tree shaking** — import named exports, avoid side-effectful imports; verify with
  `webpack-bundle-analyzer` / `rollup-plugin-visualizer`.
- **Budget the initial JS** (roughly 170 KB gzipped) and fail CI over it with `size-limit`.

```tsx
const OrdersPage = React.lazy(() => import("./pages/OrdersPage"));
function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <OrdersPage />
    </Suspense>
  );
}
```

### Images & asset delivery

- Serve modern formats (AVIF > WebP > JPEG) via CDN resizing or build-time conversion.
- Use `srcset`/`sizes` so mobile users download small files; set `width`/`height` or
  `aspect-ratio` to prevent layout shift (CLS).
- `loading="lazy"` below the fold, `fetchpriority="high"` for the LCP image.
- Inline **critical CSS**; load JS with `defer` (or `async` where order doesn't matter).
- `preconnect` to CDNs/APIs hit early; `preload` the LCP image or font.
- Hashed assets: `Cache-Control: public, max-age=31536000, immutable`; no-cache on HTML.

### React rendering

- `React.memo` to skip re-renders of expensive subtrees; `useMemo`/`useCallback` only where
  identity/cost actually matters — don't wrap everything.
- **Virtualize long lists** (`react-window`) instead of rendering 10k rows.
- **Avoid layout thrash** — batch DOM reads then writes; prefer CSS transforms over JS layout.
- Profile with React DevTools Profiler first — if renders are cheap, the problem is elsewhere.

## Network Optimization

| Technique | What it improves | Notes |
|-----------|------------------|-------|
| CDN | TTFB, cacheable static content | Edge-cache assets and even API responses |
| Compression (Brotli > gzip) | Transfer size (often 60-80% smaller) | Never compress already-encoded data (images) |
| HTTP/2 | Multiplexing, one connection, header compression | Requires HTTPS |
| HTTP/3 (QUIC) | Connection setup, loss resilience | Best on mobile/lossy networks; needs CDN/LB support |
| Keep-alive | Reuse connections, skip per-request handshakes | Default in most stacks; verify with `curl -v` |
| Prefetch / preconnect | Perceived speed for the next page / first request | `dns-prefetch`, `preconnect`, `prefetch` |

```bash
curl -sS -o /dev/null -w "DNS %{time_namelookup} | TCP %{time_connect} | TLS %{time_appconnect} | TTFB %{time_starttransfer} | Total %{time_total}\n" https://example.com/api/orders/42
```

## Concurrency Models

| Model | Best for | Gotcha |
|-------|----------|--------|
| Threads | CPU-bound (non-GIL langs), blocking I/O | Python: the GIL serializes CPU work |
| Async I/O (event loop) | I/O-bound at high concurrency | One blocking call stalls everything |
| Multiprocessing / workers | CPU-bound in Python; isolation | Memory per process; IPC cost |
| Worker threads | Parallel CPU work in Node | No shared memory; message passing |

Python note: the **GIL** means threads only help I/O-bound work — use `multiprocessing` for
CPU-bound. Node's event loop is the mirror image: offload sync CPU-heavy work from request
handlers to `worker_threads` or a job queue.

## Memory

- **Leaks** — JS: retained listeners, detached DOM nodes, module-level caches, closures.
  Python: unclosed files/connections, global caches, cyclic refs with `__del__`. Prove with
  heap snapshots (`tracemalloc`, Chrome Memory panel) taken at two points and diffed.
- **Unbounded caches** — every cache needs a size cap (LRU) or a TTL: `lru_cache(maxsize=...)`,
  `maxmemory-policy allkeys-lru`.
- **Object churn** — allocating in hot loops forces GC pressure. Reuse buffers; prefer typed
  arrays over per-iteration string concatenation.

## Worked Example: Optimizing `GET /api/orders/{id}`

**Baseline** (k6, 50 concurrent users): p50 210 ms, p95 **480 ms**, error rate 0.4%.
**Step 1 — profile.** Attach `py-spy`; the dump shows ~380 ms spent in the DB driver:

```text
Thread 0x10a (idle 3ms):
  File "app/orders.py:42" _fetch_order
    File "sqlalchemy/orm/query.py:310" execute
      File "psycopg2/extensions.py:512" poll          <- waiting on DB (~370ms)
```

**Step 2 — hypothesis.** The latency is DB time, not Python. Confirm with EXPLAIN:

```sql
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 123 ORDER BY created_at DESC;
-- Seq Scan on orders (cost=0.00..1420.00 rows=1 actual rows=3800)  <- full scan
```

**Step 3 — fix** the missing `customer_id` index and the N+1 (3,800 queries/request); cache
hot orders for 60s:

```sql
CREATE INDEX idx_orders_customer ON orders (customer_id, created_at DESC);
```

```python
order = db.query(Order).options(selectinload(Order.items)).get(order_id)
cached = redis.get(f"order:{order_id}")
if cached is not None:
    return json.loads(cached)
redis.setex(f"order:{order_id}", 60, json.dumps(order.to_dict()))
```

**Step 4 — re-measure** with the identical k6 script:

| Metric | Before | After |
|--------|--------|-------|
| p50 | 210 ms | 9 ms |
| p95 | 480 ms | **28 ms** |
| Error rate | 0.4% | 0.0% |
| DB queries per request | 3,802 | 2 (or 0 on cache hit) |

The index removed the scan (380 ms → ~15 ms), eager loading removed 3,800 round trips, and
the cache removed the DB for repeated hits — each change measured in isolation.

## Anti-Patterns to Avoid

- **Optimizing before measuring** — "fast" micro-optimizations that move nothing.
- **Fixing symptoms, not bottlenecks** — adding a cache without reading the plan; sometimes
  the index alone is the whole fix.
- **Mean-only dashboards** — a healthy mean hides a terrible p99 and the long tail.
- **Unbounded caches** — no TTL, no size cap: today's cache is tomorrow's OOM.
- **Premature concurrency** — threads/async add complexity; apply only after profiling shows
  I/O wait is the bottleneck.
- **Blindly memoizing React** — `useMemo`/`React.memo` add overhead; measure render cost first.

## When to Use / Not Use

**Use** when:
- The user reports slow endpoints, slow page loads, high latency, or poor throughput.
- You need to profile, load test, or establish performance budgets/SLOs.
- The task involves DB query optimization, caching, bundle/image optimization, or tuning
  concurrency — and you can measure the effect.

**Do NOT use** when:
- The task is a pure correctness bug, feature, or refactor with no performance concern.
- The user only needs a one-off metric readout, not a systematic optimization pass.
- Performance already meets the budget — resist optimizing without a target.
