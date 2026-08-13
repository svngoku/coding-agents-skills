# Observability and Distributed Tracing

Deep dive on making N services debuggable: structured logs, RED/USE metrics, OpenTelemetry tracing with context propagation, and health/readiness endpoints.

## Table of Contents
1. [The three pillars](#the-three-pillars)
2. [Structured logs](#structured-logs)
3. [Metrics: RED and USE](#metrics-red-and-use)
4. [Distributed tracing with OpenTelemetry](#distributed-tracing-with-opentelemetry)
5. [Health vs readiness](#health-vs-readiness)
6. [Checklist](#checklist)

---

## The three pillars

| Pillar | Answers | Shape |
|---|---|---|
| Logs | What happened, in detail? | Events; high volume; searchable |
| Metrics | How many / how fast / how often? | Aggregated numbers; dashboards, alerts |
| Traces | How did one request flow through services? | Spans with parent/child links |

Each pillar answers a different question; traces are what tie logs and metrics from different services together.

## Structured logs

Log as JSON with a fixed field set so logs are filterable and correlate-able across services:

```json
{
  "ts": "2024-05-01T12:00:00.123Z",
  "level": "info",
  "service": "orders",
  "trace_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
  "span_id": "1a2b3c4d5e6f7a8b",
  "event": "order.created",
  "order_id": "ord_123",
  "duration_ms": 42
}
```

Rules:
- Machine-parseable JSON only — no free-form prose fields that cannot be queried.
- Always include `service`, `trace_id`, and an `event` name; the trace_id is what links this line to every other service that handled the request.
- Include business identifiers (`order_id`, `user_id`) — you debug by order, not by process.
- Log at the edges (entry, exit, errors, side effects), not inside every loop iteration.

## Metrics: RED and USE

| Framework | Focus | Metrics |
|---|---|---|
| **RED** (services) | Rate, Errors, Duration | requests/sec, error rate, latency p50/p95/p99 |
| **USE** (resources) | Utilization, Saturation, Errors | CPU %, queue depth, memory, error counters |

Every service should expose: requests/sec, error rate, latency histogram, plus its own business metrics (order rate, refund rate). Alert on RED deviations; drill into USE when a RED metric degrades.

```python
# Prometheus client: latency histogram for a handler
from prometheus_client import Histogram, Counter, generate_latest

REQUESTS = Counter("http_requests_total", "Requests", ["service", "route", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Latency", ["service", "route"])

with LATENCY.labels("orders", "/orders").time():
    response = handle(request)
REQUESTS.labels("orders", "/orders", response.status_code).inc()
```

## Distributed tracing with OpenTelemetry

One SDK per service; a span per operation (HTTP handler, DB query, publish, consume); exporters send spans to a collector, which forwards to a backend (Jaeger, Tempo, Datadog, etc.).

Context propagation is what makes it "distributed":

- **Sync calls (HTTP):** propagate the W3C `traceparent` header. `fetch`/gRPC interceptors do this automatically when the SDK is initialized.
- **Async calls (events/messages):** put the trace context in message headers (e.g., Kafka headers) so the consuming service continues the same trace.
- **Queues:** use the message header, not ambient context — the producer and consumer run in different processes/threads.

```typescript
// Carry the trace across an async boundary via Kafka headers
import { trace } from "@opentelemetry/api";

const span = trace.getActiveSpan();
const headers: Record<string, string> = {};
if (span) {
  trace.propagate.inject(trace.getActiveContext(), headers);
}
await producer.send({
  topic: "orders",
  messages: [{ value: JSON.stringify(event), headers }],
});
```

Sampling:
- Low volume (most internal systems): sample 100% — traces are cheap and debugging is far easier.
- High volume: head-based sampling (e.g., 10%) or tail-based sampling that keeps traces containing errors or slow spans.
- Never sample only logs — sample the trace, and always emit the trace_id to logs regardless of sampling.

## Health vs readiness

Two endpoints with different semantics and different consumers:

| Probe | Path | Meaning | Failure action |
|---|---|---|---|
| Liveness | `/healthz` | Process alive? | Restart the container |
| Readiness | `/readyz` | Can it serve traffic? (DB, caches, queues reachable) | Stop routing traffic; gate deploys |

Rules:
- **Liveness must not depend on external services.** If the DB is down and liveness fails, the orchestrator restarts the pod in a loop while the DB is still down.
- **Readiness should reflect real dependencies**: DB connectivity, message broker reachable, background queues drained. This is also what gates canary rollouts — the new version is only served traffic when it reports ready.
- Kubernetes: `startupProbe` for slow-starting services, then `livenessProbe` + `readinessProbe`.

## Checklist

- [ ] All logs are structured JSON with service, trace_id, and event name
- [ ] RED metrics exposed per service (rate, errors, latency histograms)
- [ ] OpenTelemetry SDK initialized in every service; W3C traceparent propagated
- [ ] Trace context propagated across async boundaries via message headers
- [ ] Sampling strategy defined (100% at low volume)
- [ ] `/healthz` (liveness, no external deps) and `/readyz` (readiness, real checks) implemented
- [ ] Rollouts gated on readiness of the new version
