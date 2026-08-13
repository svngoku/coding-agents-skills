# Error Handling, RFC 7807, and Idempotency

Deep dive for the [api-design](../SKILL.md) skill. Covers the error envelope in full, error-code catalogs, retry semantics, and the idempotency-key lifecycle.

## The RFC 7807 envelope

RFC 7807 (now RFC 9457) defines `application/problem+json` — a standard shape for error responses. Every error endpoint returns this shape; clients write one parser and use it everywhere.

| Field | Required | Purpose |
|---|---|---|
| `type` | ✅ | URI identifying the problem class, e.g. `https://api.example.com/problems/out-of-stock`. Should resolve to human documentation. |
| `title` | ✅ | Short, human-readable summary of the problem class (stable per `type`). |
| `status` | ✅ | HTTP status code (echoed so body-only logging still works). |
| `detail` | ❌ | Human explanation specific to *this* occurrence. |
| `instance` | ❌ | URI of the specific request/resource that failed. |

Your API will almost certainly want two extensions:

- **`code`** — a stable, documented machine-readable string clients branch on (`OUT_OF_STOCK`, `INVALID_ARGUMENT`, `RATE_LIMITED`). Never make clients parse `detail` or `title`; those change when copy changes.
- **`retryable`** — explicit boolean so clients do not need to re-derive policy. If omitted, clients should derive: retryable = status ≥ 500 or status in {408, 425, 429}.

```json
{
  "type": "https://api.example.com/problems/out-of-stock",
  "title": "Order cannot be placed",
  "status": 422,
  "detail": "SKU-42 has only 3 units in stock",
  "instance": "/orders/8f2",
  "code": "OUT_OF_STOCK",
  "retryable": false,
  "errors": [
    { "field": "items[0].qty", "code": "OUT_OF_STOCK", "message": "Only 3 units available" }
  ]
}
```

## Error-code catalogs

- Define codes per problem class, not per message. One `code` maps to one `type` + `title`.
- Use a naming convention: `DOMAIN_VERB` (`ORDER_NOT_FOUND`, `PAYMENT_DECLINED`) or verb-style (`INVALID_ARGUMENT`, `NOT_FOUND`, `PERMISSION_DENIED`) — pick one and stay consistent.
- Document every code in the API reference; a code a client cannot look up is a bug.
- Treat codes as API surface: renaming a code is a breaking change, versioned like any other.

### Field-level validation

Return one problem with an `errors[]` array rather than one error per field — a single round-trip for the client:

| Field | Meaning |
|---|---|
| `field` | JSON pointer or dot-path into the request body (`items[0].qty`) |
| `code` | machine-readable reason for this field |
| `message` | human-readable reason |

## Idempotency keys in depth

### The problem

`POST /orders` with a dropped response: the client retries, and the server creates two orders. `PUT` and `DELETE` are idempotent by HTTP semantics, so retries are safe; `POST` and `PATCH` are not.

### The mechanism

The client generates a UUID per logical operation and sends it in the `Idempotency-Key` header. The server:

1. Looks up the key in its idempotency store (usually Redis/DynamoDB with a TTL).
2. If **absent**: executes the operation, stores key → {status, headers, body}, returns the result.
3. If **present with the same body**: returns the stored response without re-executing.
4. If **present with a different body**: returns `409 Conflict` — the key is already bound to a different request.

```http
POST /orders
Idempotency-Key: 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d

{ "items": [ { "sku": "SKU-42", "qty": 2 } ] }
```

### Design decisions

| Decision | Recommendation | Why |
|---|---|---|
| Which endpoints | every POST/PATCH that creates or mutates; definitely money-adjacent ones | retries are the norm, not the exception, on flaky networks |
| Required or optional | require it on writes (400 if missing) or auto-generate server-side | required is simplest for clients to reason about; auto-generate hides it from curl-style tools |
| TTL | 24h (Stripe) is a common default; longer for payment flows | covers retry storms; short enough to keep the store small |
| Store | the response (status, headers, body) — not just "processed" | the whole point is replaying the *original* response |
| Concurrent first requests | lock on the key; one wins, the other gets the stored response | prevents double-execution under retry races |
| Key namespace | per-endpoint or global | global is simpler; per-endpoint avoids key collisions across unrelated operations |

## Retry semantics

Clients should retry only when the response says it is safe:

| Situation | Retry? | Headers |
|---|---|---|
| 429 Too Many Requests | yes, after backoff | `Retry-After` (seconds or date) |
| 5xx (500, 502, 503, 504) | yes, with capped exponential backoff + jitter | `Retry-After` on 503 |
| 4xx (400–422) | no — fix the request | — |
| 408/425 | yes, transient | — |
| Network timeout before response | yes — **with the same Idempotency-Key** | — |

Guidelines:

- Cap retries (e.g. 3–5 attempts) and jitter the backoff; never retry hot.
- Honor `Retry-After` when present; fall back to your own backoff when absent.
- A retry after a timeout must reuse the `Idempotency-Key` so the server deduplicates.
- Log retry counts and final outcomes; alert on retry storms.

## Full status-code quick reference

The complete table (the main SKILL.md carries the common subset):

| Code | Class | Use |
|---|---|---|
| 200 OK | 2xx | success — read, update, action |
| 201 Created | 2xx | resource created (+ `Location`) |
| 202 Accepted | 2xx | accepted for async processing |
| 204 No Content | 2xx | success, empty body |
| 206 Partial Content | 2xx | ranged response (resumable downloads) |
| 301 Moved Permanently | 3xx | permanent redirect (canonical URL) |
| 304 Not Modified | 3xx | conditional GET — cache is fresh (ETag / If-None-Match) |
| 308 Permanent Redirect | 3xx | redirect preserving method/body |
| 400 Bad Request | 4xx | malformed syntax, missing required field |
| 401 Unauthorized | 4xx | missing/invalid credentials |
| 403 Forbidden | 4xx | authenticated, not allowed (wrong scope) |
| 404 Not Found | 4xx | no such resource/route |
| 405 Method Not Allowed | 4xx | method not supported on this resource (+ `Allow`) |
| 406 Not Acceptable | 4xx | cannot honor `Accept` |
| 408 Request Timeout | 4xx | client took too long (transient, retryable) |
| 409 Conflict | 4xx | state conflict (idempotency-key body mismatch, version mismatch) |
| 410 Gone | 4xx | resource existed, permanently removed |
| 412 Precondition Failed | 4xx | If-Match / If-Unmodified-Since failed |
| 413 Payload Too Large | 4xx | request body over limit |
| 415 Unsupported Media Type | 4xx | wrong Content-Type |
| 422 Unprocessable Entity | 4xx | well-formed but semantically invalid |
| 425 Too Early | 4xx | replay risk (retryable) |
| 429 Too Many Requests | 4xx | rate limited (+ `Retry-After`) |
| 500 Internal Server Error | 5xx | unhandled server failure |
| 502 Bad Gateway | 5xx | upstream returned invalid response |
| 503 Service Unavailable | 5xx | down/overloaded (+ `Retry-After`) |
| 504 Gateway Timeout | 5xx | upstream timeout |

Rules that never change:

- 2xx success, 3xx redirect/cache, 4xx client error, 5xx server error. Never blur the boundary.
- Never return 500 for bad input; never return 200 for a failed operation.
- Use the narrowest code that fits (409 over 400 when the request is well-formed but conflicts with state).

## Error anti-patterns

- **HTML error pages from an API** — always JSON (or the negotiated format), always the same envelope.
- **Inconsistent envelopes** — one validation error shape, a different auth shape, a different 500 shape: three parsers, three bugs.
- **Leaking internals** — stack traces, SQL, file paths, or dependency versions in `detail`.
- **Codes that change with copy** — `code` must be stable; the human message may drift.
- **Idempotency keys stored without the response body** — you cannot replay what you did not store.
