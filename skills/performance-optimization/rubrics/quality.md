# Rubric: Optimizing the Slow GET /orders Endpoint — score 0.0–1.0

The deterministic checks verify that the artifacts exist and the core fixes are
present; this rubric scores *how well* the agent applied the skill's
measure-first discipline. Score the four dimensions below (0.0–0.25 each) and
sum, or judge holistically against the anchors.

## Measure-first discipline (0.0–0.25)

- The bottleneck is identified from evidence, not guesswork: baseline numbers
  (latency, percentile, or query count) are stated before the fix is proposed.
- NOTES.md explains how the diagnosis would be verified (EXPLAIN plan, query
  logs, profiler) and how the fix would be re-measured.
- Claims are tied to a budget or target (e.g. "p95 < 100 ms"), not a vague
  "it should be faster".

## Minimal, effective fix (0.0–0.25)

- The fix targets the actual bottleneck: the N+1 is removed by batching or
  eager loading, not by a superficial rewrite (renaming, wrapping, or
  micro-tuning a loop that still queries per row).
- The cache is scoped, TTL'd, and invalidated sensibly — no unbounded cache,
  and the cache's purpose (which reads it protects) is explained.
- No gratuitous complexity: no premature async/threading, no new dependencies
  the bottleneck does not justify.

## Correctness preserved (0.0–0.25)

- The endpoint still returns the same shape: `GET /orders?customer_id=N`
  yields each order with its `items` list; empty results and unknown
  customers are handled.
- `optimized.py` is valid Python and keeps the handler/route intact.
- Batching does not change what is returned — the response fields and
  semantics are unchanged from `slow.py`.

## Evidence-based claims (0.0–0.25)

- NOTES.md states concrete before/after numbers (p95/p50/ms, queries per
  request) rather than adjectives.
- Expected EXPLAIN plans are described concretely (e.g. `Seq Scan` →
  `Index Scan`).
- The write-up distinguishes measured facts from hypotheses.

## Penalties

- **−20%** if `optimized.py` would not run (syntax errors, undefined names).
- **−10%** if the "fix" is a cosmetic copy of `slow.py` with no real change
  to the query pattern.
