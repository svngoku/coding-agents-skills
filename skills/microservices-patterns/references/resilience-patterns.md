# Resilience Patterns

Deep dive on the patterns that keep a microservices system alive when dependencies degrade. Read this when wiring timeouts, retries, circuit breakers, bulkheads, fallbacks, or load shedding around service calls.

## Table of Contents
1. [The failure taxonomy](#the-failure-taxonomy)
2. [Timeouts and deadlines](#timeouts-and-deadlines)
3. [Retries with backoff and jitter](#retries-with-backoff-and-jitter)
4. [Circuit breaker](#circuit-breaker)
5. [Bulkhead](#bulkhead)
6. [Fallback and degraded mode](#fallback-and-degraded-mode)
7. [Load shedding and backpressure](#load-shedding-and-backpressure)
8. [A worked resilience config](#a-worked-resilience-config)
9. [Checklist](#checklist)

---

## The failure taxonomy

Match the pattern to the failure mode:

| Failure mode | Example | Primary defense |
|---|---|---|
| Slow dependency | DB query takes 30 s | Timeouts + overall deadline |
| Transient error | Connection reset, 503, timeout | Retry with backoff + jitter |
| Persistent outage | Dependency down for minutes | Circuit breaker + fallback |
| Resource exhaustion | Thread pool saturated | Bulkhead + timeouts |
| Overload | Traffic spike | Load shedding + backpressure |
| Duplicate delivery | Broker redelivers | Idempotent consumers |

## Timeouts and deadlines

Rules:
- Always set timeouts on remote calls. A call with no timeout holds a thread/connection forever.
- Separate **connect** and **read** timeouts (a connection can succeed while reads stall).
- Set an **overall deadline** per operation when it fans out to several services, so the total latency is bounded even if each call has its own timeout.
- Tune from measured p99s, not guesses. If p99 is 800 ms, a 2–5 s read timeout is generous; 30 s is a bug.

```typescript
// Node fetch: one total budget per call (covers connect + read)
const res = await fetch(url, { signal: AbortSignal.timeout(2500) });

// gRPC: per-call deadline
const res = await stub.GetOrder({ id }, { deadline: Date.now() + 2000 });
```

## Retries with backoff and jitter

Rules:
- Retry **only idempotent** operations (GETs, idempotency-keyed writes, event consumption). Never retry a non-idempotent POST blindly.
- Do not retry 4xx client errors — except 408 (request timeout) and 429 (rate limited), which honor `Retry-After`.
- Limit attempts (2–3 total). Retrying more amplifies load on an already failing dependency.
- Use exponential backoff with **jitter**. Full jitter prevents thundering herds:

```typescript
function backoff(attempt: number, baseMs = 100, capMs = 1000): number {
  const exp = Math.min(capMs, baseMs * 2 ** attempt);
  return Math.floor(Math.random() * exp); // full jitter: random in [0, exp)
}

// attempts: 0 → [0,100ms), 1 → [0,200ms), 2 → [0,400ms), ... capped
```

- Set a **retry budget** per request (max attempts × max backoff), and stop retrying when the circuit is open.

## Circuit breaker

Prevents retries from hammering a failing dependency. Three states:

| State | Behavior |
|---|---|
| Closed | Requests pass; failures counted in a sliding window |
| Open | Fail fast (no calls) for the open window; reject immediately |
| Half-open | Let a probe request through; success → closed, failure → open again |

Typical parameters (Resilience4j defaults): failure-rate threshold 50%, sliding window 100 calls, open duration 60 s, half-open permits 10.

```yaml
resilience4j.circuitbreaker:
  instances:
    payments:
      slidingWindowSize: 100
      failureRateThreshold: 50
      waitDurationInOpenState: 60s
      permittedNumberOfCallsInHalfOpenState: 10
      recordExceptions:
        - java.io.IOException
        - java.net.SocketTimeoutException
```

## Bulkhead

Isolates dependencies from each other so one slow dependency cannot exhaust the shared thread pool. Give each dependency its own pool or semaphore.

```yaml
resilience4j.bulkhead:
  instances:
    payments:
      maxConcurrentCalls: 10
      maxWaitDuration: 0   # fail fast when saturated
    inventory:
      maxConcurrentCalls: 20
      maxWaitDuration: 0
```

Thread-pool bulkheads (separate threads per dependency) also protect the calling thread but cost more memory; semaphore bulkheads are cheap and usually enough.

## Fallback and degraded mode

When a dependency is down or slow, return something useful instead of failing:

- **Stale cache** — serve the last known-good data, tagged as possibly stale.
- **Default/degraded values** — e.g., shipping estimates from a static table instead of the live carrier API.
- **Queue for later** — accept the request, do the work asynchronously, notify when done.

```typescript
const ratings = await circuitBreaker.executeFallback(
  () => ratingsService.get(productId),
  () => cache.get(`ratings:${productId}`) ?? { average: null, stale: true }
);
```

Fallbacks must be fast — a fallback that itself calls another failing service is just a longer failure path.

## Load shedding and backpressure

When a service cannot keep up, shed work deliberately instead of degrading for everyone:

- **Queue depth limits** — reject requests once the work queue exceeds a bound.
- **429 with Retry-After** — tell clients to back off; clients honor it (and jitter their retries).
- **Prioritized shedding** — shed low-priority (batch, non-interactive) work first; keep interactive traffic.
- **Backpressure upstream** — limit in-flight requests from callers (bounded concurrency), so overload propagates as latency rather than pile-ups.

Monitoring: track rejected/shed request rates. If shedding is constant, you are under-provisioned, not resilient.

## A worked resilience config

All patterns together for one caller (order service → payments service):

```yaml
resilience4j:
  timeout:
    instances:
      payments:
        timeoutDuration: 2s
  retry:
    instances:
      payments:
        maxAttempts: 3
        waitDuration: 100ms
        enableExponentialBackoff: true
        exponentialBackoffMultiplier: 2
        enableRandomizedWait: true
        randomizedWaitFactor: 1   # full jitter
        retryExceptions:
          - java.net.SocketTimeoutException
  circuitbreaker:
    instances:
      payments:
        slidingWindowSize: 100
        failureRateThreshold: 50
        waitDurationInOpenState: 60s
  bulkhead:
    instances:
      payments:
        maxConcurrentCalls: 10
```

Order of wrapping (outermost → innermost): bulkhead → circuit breaker → retry → timeout → call. The circuit breaker sits outside retries so it can open even while retries are configured.

## Checklist

- [ ] Every remote call has connect + read timeouts and an overall deadline
- [ ] Retries only on idempotent calls, bounded attempts, backoff + jitter
- [ ] Circuit breaker per dependency (not per service) with half-open probes
- [ ] Bulkheads isolate dependencies from the shared pool
- [ ] Fallbacks are fast, local, and degrade gracefully
- [ ] Load shedding defined for overload (429s, queue limits, prioritization)
- [ ] Consumers idempotent against redelivery
