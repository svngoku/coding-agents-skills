---
name: api-design
description: >
  Design and review intuitive, scalable, maintainable HTTP APIs. Use this skill whenever the user
  wants to design a new REST API, review an existing API or spec, write OpenAPI 3.x definitions,
  or work with HTTP semantics (GET/POST/PUT/PATCH/DELETE), status codes, idempotency
  (Idempotency-Key), error envelopes (RFC 7807 problem+json), pagination, filtering, versioning,
  or API auth (API keys, OAuth2 client credentials, rate limits). Also trigger for "API design",
  "RESTful", "endpoints", "OpenAPI", "Swagger", "ReDoc", "contract testing", or GraphQL and gRPC
  design questions.
---

# API Design

HTTP APIs are the contract between your service and every client that consumes it. A well-designed API is consistent and predictable: the URL says which resource you are touching, the HTTP method says what happens to it, and the status code plus body say whether it worked and why not. This skill covers designing and reviewing such APIs — resource modeling, HTTP semantics, status codes, error envelopes, pagination, versioning, spec-first OpenAPI 3.x, and authentication basics — REST first, with GraphQL and gRPC covered in passing.

## Quick Reference

| Task | Reference |
|------|-----------|
| Error envelopes, RFC 7807, idempotency keys, retry policy | [error-handling.md](references/error-handling.md) |
| Cursor vs offset pagination, filtering, sorting, field selection | [pagination-filtering.md](references/pagination-filtering.md) |
| Versioning strategies, backward compatibility, deprecation | [versioning-evolution.md](references/versioning-evolution.md) |

## Core Workflow

Design an endpoint set in five steps:

1. **Model resources and map HTTP semantics** — identify the nouns (users, orders, products) and their relationships; name them as plural collections; one method per operation: read → GET, create → POST, full replace → PUT, partial update → PATCH, delete → DELETE, domain action → POST to an action sub-resource.
2. **Design the responses** — success shapes, the error envelope (problem+json), and the exact status code per failure mode.
3. **Paginate and filter collections** — choose cursor or offset, set defaults, define filter/sort/field-selection syntax.
4. **Write the OpenAPI spec first** — the spec is the contract; generate docs and clients from it.
5. **Lock down auth and limits** — authentication scheme, least-privilege scopes, rate limits with standard headers.

## Resource Modeling

### Nouns, not verbs

Resources are **nouns**; HTTP methods are the verbs. URLs must never contain verbs.

| Anti-pattern | Good |
|---|---|
| `GET /getUser?id=1` | `GET /users/{id}` |
| `POST /createOrder` | `POST /orders` |
| `POST /deleteOrder?id=1` | `DELETE /orders/{id}` |

### Collections, sub-resources, actions

| Pattern | Example | Meaning |
|---|---|---|
| Collection | `GET /orders` | list (paginated) |
| Sub-resource | `GET /users/{id}/orders` | orders *owned by* that user |
| Action | `POST /orders/{id}/cancel` | domain action with side effects |

### Relationships

- Use **sub-resources only for genuinely nested, exclusive ownership** (`/users/{id}/orders` makes sense; `/orders/{id}/customer` does not — the customer is not owned by the order).
- For non-ownership links, keep the related resource at top level: `GET /orders?customer_id=...`. Keep nesting shallow — never deeper than one level; embed related **IDs** in payloads, not whole objects.

### Actions are POSTs

When a client operation is not CRUD — cancel, refund, publish, archive — model it as a **POST to a named action sub-resource** (`POST /orders/{id}/cancel`). POST is the only method with no idempotency guarantee, which fits side-effecting actions — but pair it with an `Idempotency-Key` (below).

## HTTP Semantics

| Method | Semantics | Safe | Idempotent | Notes |
|---|---|---|---|---|
| GET | read a collection or instance | ✅ | ✅ | never changes state; cacheable |
| POST | create a resource, or trigger an action | ❌ | ❌ | returns `201 Created` + `Location` |
| PUT | full replacement of an existing resource | ❌ | ✅ | client sends the entire representation |
| PATCH | partial update (changed fields only) | ❌ | ❌ by spec | send only what changes |
| DELETE | remove a resource | ❌ | ✅ | repeat DELETE → 204 or 404, both fine |

**Safe** = no side effects (safe to retry, cache, prefetch). **Idempotent** = N identical requests produce the same result as one.

### Idempotency (PUT vs POST, Idempotency-Key)

- **PUT is inherently idempotent** — use it when the client can supply the full representation. Same payload twice → same state.
- **POST and PATCH are not idempotent.** A retried `POST /orders` creates two orders. Fix this with the **`Idempotency-Key` request header**: the client sends a UUID, the server stores key → response, and replays the stored response instead of re-executing.

```http
POST /orders
Idempotency-Key: 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d

{ "items": [ { "sku": "SKU-42", "qty": 2 } ] }
```

- Require `Idempotency-Key` on **any POST/PATCH that creates or mutates** (payments, orders, subscriptions).
- A replay with the same key but a **different body** → `409 Conflict`.
- Expire keys after a bounded window (Stripe uses 24h); document the window.
- On replay, return the **original** stored response — that is what makes client retries safe.

## Status Codes

Pick the narrowest correct code; do not invent meanings. Full 2xx–5xx table (304, 405, 502, 504 and friends): [error-handling.md](references/error-handling.md).

| Code | Use | Example |
|---|---|---|
| 200 OK | successful read, update, or action | `GET /orders/{id}` |
| 201 Created | resource created (set `Location`) | `POST /orders` |
| 400 Bad Request | malformed syntax or missing required field | invalid JSON, unknown enum value |
| 401 Unauthorized | missing or invalid credentials | no/expired token |
| 403 Forbidden | authenticated but not allowed | valid token, wrong scope |
| 404 Not Found | no such resource or route | `GET /orders/9999` |
| 409 Conflict | state conflict; request cannot apply | duplicate `Idempotency-Key` body |
| 422 Unprocessable Entity | well-formed but semantically invalid | order with 0 items, negative price |
| 429 Too Many Requests | rate limited (send `Retry-After`) | burst over limit |
| 500 Internal Server Error | unhandled server failure | bug, DB outage |
| 503 Service Unavailable | down / overloaded (send `Retry-After`) | maintenance |

### Common mistakes

- **200 with an error body** — breaks every client's status-code handling. Errors live in 4xx/5xx.
- **500 for client errors** — validation failures are 4xx; 500 triggers alerts and pointless retries for something the client did.
- **401 vs 403** — 401: *who are you* (no/expired credentials); 403: *you are not allowed* (authenticated, lacking permission or scope).

## Error Responses

Use a **consistent envelope everywhere**, preferably **RFC 7807 `application/problem+json`**:

```json
{
  "type": "https://api.example.com/problems/out-of-stock",
  "title": "Order cannot be placed",
  "status": 422,
  "detail": "SKU-42 has only 3 units in stock",
  "code": "OUT_OF_STOCK",
  "retryable": false,
  "errors": [{ "field": "items[0].qty", "code": "OUT_OF_STOCK", "message": "Only 3 units available" }]
}
```

- `type` — stable machine-readable problem URI; `title` — short human summary; `status` — HTTP status; `detail` — human explanation; `instance` — URI of the request/resource.
- Add a **`code`** field: a stable, documented string clients can branch on (`OUT_OF_STOCK`, `INVALID_ARGUMENT`, `RATE_LIMITED`). Never branch on the human-readable `title`/`detail`.
- Add **`retryable`** (or derive it): 5xx and 429 are retryable; 4xx are not (except 408/425/429).
- Field-level validation failures → an `errors[]` array with `field`, `code`, `message`.
- Serve as `Content-Type: application/problem+json`; document every code in your API reference.

## Pagination, Filtering, Sorting

### Cursor vs offset

| Aspect | Offset (`?page=2&page_size=50`) | Cursor (`?cursor=...`) |
|---|---|---|
| Stability | drifts when rows are inserted/deleted mid-paging | stable — pages are anchored |
| UX | supports jump-to-page N | next/prev only (no jump) |
| Implementation | `OFFSET`/`LIMIT` (trivial) | keyset: `WHERE (id > :cursor) LIMIT n` |
| Best for | admin tables, small datasets | live, high-churn feeds |
| Ordering | any | requires a stable sort key |

Default to **cursor for public/live collections**, offset for internal/admin. Set a default page size (25–50) and a hard max (100–1000); clamp or return 400 on overflow — and document it.

```json
{ "data": [{ "id": "ord_8f2", "total": "129.00" }],
  "pagination": { "next_cursor": "eyJpZCI6Im9yZF84ZjIifQ", "has_more": true } }
```

### Filtering, sorting, field selection

| Concern | Convention | Example |
|---|---|---|
| Exact match | `?key=value` on known fields | `?status=shipped` |
| Ranges | `*_gte` / `*_lte` suffix (or `[gte]`) | `?created_at_gte=2024-01-01` |
| Sorting | `sort` param; `-` prefix = descending | `?sort=-created_at,sku` |
| Field selection | sparse fieldsets | `?fields=id,name,price` |

Validate every filter and sort field against a whitelist — unknown keys → `400`, never silently ignored.

## Versioning

| Strategy | Example | Pros | Cons |
|---|---|---|---|
| URL path | `/api/v2/orders` | visible, cacheable per version, easy routing | multiple code paths over time |
| Media type | `Accept: application/vnd.api+json; version=2` | clean URLs | hidden from casual inspection, harder to debug |
| Query param | `/orders?v=2` | trivial to add | pollutes cache keys and logs; widely discouraged |

**URL path is the default** for public APIs. Never version with query parameters for a public API.

### Backward compatibility

| Additive (safe, no version bump) | Breaking (requires a new version) |
|---|---|
| new optional fields or query parameters | remove or rename a field |
| new endpoint; widen enums; relax validation | change a field's type or meaning; make an optional field required |

Evolve additively as long as you can; version only when a breaking change is unavoidable. When deprecating: announce it, keep the old version alive on a documented sunset schedule, and emit `Deprecation` / `Sunset` (RFC 8594) response headers.

## Spec-First with OpenAPI 3.x

1. **Write the OpenAPI spec before the implementation** — it is the single source of truth for shape, examples, and behavior.
2. **Lint it in CI** — [Spectral](https://github.com/stoplightio/spectral) rules catch missing responses, undocumented status codes, and naming drift.
3. **Generate docs and clients from the spec** — [ReDoc](https://github.com/Redocly/redoc) / Redocly, Swagger UI, [openapi-generator](https://github.com/OpenAPITools/openapi-generator), openapi-typescript, orval.
4. **Contract-test it and treat spec and implementation as one** — [Schemathesis](https://github.com/schemathesis/schemathesis) fuzzes the spec against the running service; [Pact](https://pact.io) handles consumer-driven contracts; when a review finds a mismatch, fix the spec first, then the code.

## Authentication & Rate Limiting

### Schemes

| Scheme | Use for | How | Notes |
|---|---|---|---|
| API key | simple machine clients, first-party tools | `Authorization: Bearer <key>` or `X-API-Key: <key>` | long-lived; treat as a secret; support rotation |
| OAuth2 client credentials | machine-to-machine | `POST /oauth/token` (`grant_type=client_credentials&scope=read:orders`) → access token | short-lived tokens |
| OAuth2 authorization code | user-facing apps acting for a user | browser flow → access token | delegate to an IdP (Auth0, Clerk, …) |

Use **scopes for least privilege**: `read:orders`, `write:orders`, `admin`. A token with only `read:orders` calling a write endpoint → `403`.

### Rate limiting

On exceeding the limit, return `429` with `Retry-After` (seconds) plus the standard headers: `X-RateLimit-Limit` (window quota), `X-RateLimit-Remaining`, `X-RateLimit-Reset` (epoch seconds when the window resets). Document your window ("100 requests/minute per API key") and whether limits are per-key, per-IP, or per-user.

## Worked Example: Orders API

An orders resource with a lifecycle — the endpoint set, and why each choice:

| Method | Path | Semantics |
|---|---|---|
| GET | `/orders` | list, cursor-paginated, filterable by `status`, sortable by `created_at` |
| POST | `/orders` | create; requires `Idempotency-Key`; returns `201` + `Location` |
| GET | `/orders/{id}` | fetch one; `404` if unknown |
| PATCH | `/orders/{id}` | partial update (e.g. shipping address) |
| POST | `/orders/{id}/cancel` | domain action (triggers refunds); idempotency-keyed; `409` if already shipped |
| GET | `/orders/{id}/items` | sub-resource — items belong exclusively to the order |

Notes: no `DELETE /orders` — physical deletion is not part of the lifecycle; cancellation is a named action. No `PUT` — partial updates fit PATCH. Errors use the RFC 7807 envelope; collections use cursor pagination.

OpenAPI 3.1 excerpt (paths, security, components — trimmed):

```yaml
openapi: 3.1.0
info: { title: Orders API, version: 2.0.0 }
servers: [{ url: https://api.example.com/v2 }]
security: [{ oauth2: [read:orders, write:orders] }]
paths:
  /orders:
    get:
      summary: List orders
      parameters:
        - { name: status, in: query, schema: { type: string, enum: [created, confirmed, shipped, cancelled] } }
        - { name: cursor, in: query, schema: { type: string, description: "Opaque pagination token" } }
        - { name: limit, in: query, schema: { type: integer, maximum: 100, default: 25 } }
      responses:
        "200": { description: A page of orders, content: { application/json: { schema: { $ref: "#/components/schemas/OrderPage" } } } }
        "429": { description: Too many requests, headers: { Retry-After: { schema: { type: integer } } }, content: { application/problem+json: { schema: { $ref: "#/components/schemas/Problem" } } } }
    post:
      summary: Create an order
      parameters:
        - { name: Idempotency-Key, in: header, required: true, schema: { type: string, format: uuid } }
      requestBody: { required: true, content: { application/json: { schema: { $ref: "#/components/schemas/NewOrder" } } } }
      responses:
        "201": { description: Order created, headers: { Location: { schema: { type: string } } }, content: { application/json: { schema: { $ref: "#/components/schemas/Order" } } } }
        "422": { description: Unprocessable entity, content: { application/problem+json: { schema: { $ref: "#/components/schemas/Problem" } } } }
components:
  securitySchemes:
    oauth2: { type: oauth2, flows: { clientCredentials: { tokenUrl: https://auth.example.com/oauth/token, scopes: { read:orders: Read orders, write:orders: Create and update orders } } } }
  schemas:
    Problem: { type: object, required: [type, title, status, code], properties: { type: { type: string, format: uri }, title: { type: string }, status: { type: integer }, detail: { type: string }, instance: { type: string }, code: { type: string }, retryable: { type: boolean } } }
    Order: { type: object, required: [id, status, total, currency], properties: { id: { type: string, example: ord_8f2 }, status: { type: string, enum: [created, confirmed, shipped, cancelled] }, total: { type: string, example: "129.00" }, currency: { type: string, example: USD } } }
    NewOrder: { type: object, required: [items], properties: { items: { type: array, minItems: 1, items: { type: object, required: [sku, qty], properties: { sku: { type: string }, qty: { type: integer, minimum: 1 } } } } } }
    OrderPage: { type: object, required: [data, pagination], properties: { data: { type: array, items: { $ref: "#/components/schemas/Order" } }, pagination: { type: object, required: [has_more], properties: { next_cursor: { type: string }, has_more: { type: boolean } } } } }
```

## Pre-Review Checklist

Run this on any existing API before approving changes or declaring it done:

| # | Area | Check |
|---|---|---|
| 1 | Naming & URLs | Plural nouns, consistent casing, no verbs in URLs |
| 2 | Methods & idempotency | GET never mutates; PUT full replace, PATCH partial; creating/mutating POST/PATCH accept `Idempotency-Key` |
| 3 | Status codes | No 200-with-error-body; 4xx for client errors; narrowest correct code |
| 4 | Errors | One envelope everywhere (RFC 7807); stable machine-readable `code`s documented |
| 5 | Pagination | Cursor or offset chosen and documented; default + max page size; `has_more` / next token |
| 6 | Filtering/sorting | Params whitelisted and validated; unknown keys → 400 |
| 7 | Versioning | Strategy documented; no silent breaking changes; deprecation has a sunset plan |
| 8 | Spec fidelity | OpenAPI exists, linted, matches behavior (contract tests in CI) |
| 9 | Auth & rate limits | Least-privilege scopes; `X-RateLimit-*` headers; `Retry-After` on 429 |
| 10 | Security | TLS everywhere; no secrets in URLs/logs; no stack traces in responses |

## Anti-Patterns to Avoid

- **Verbs in URLs** (`/getUser`, `/createOrder`) — the method is the verb.
- **200 with an error body** — clients branch on status codes; you have just broken all of them.
- **500 for client mistakes** — mis-validated input is a 4xx; 500 means a server bug.
- **Deep nesting** (`/users/{id}/orders/{oid}/items/{iid}`) — flat resources + query params scale better.
- **Bare POST without idempotency** on creates and money-adjacent operations — double-submits are real.
- **Unbounded collections** — every list endpoint paginates or dies.

## When to Use / Not Use

**Use this skill when:**

- Designing a new REST API, or a new set of endpoints for an existing service
- Reviewing an API design doc, an OpenAPI spec, or a PR that changes endpoint behavior
- Establishing API design standards for a team or platform

**Do not use when:**

- The "API" is a library or function interface (Python SDK internals, method signatures) — different concerns
- Designing event payloads or message-broker contracts (Kafka, SQS) — no status codes or idempotency keys in the same sense
- The user wants a deep GraphQL- or gRPC-only treatment — this skill covers them in passing only; use dedicated GraphQL/gRPC guides
