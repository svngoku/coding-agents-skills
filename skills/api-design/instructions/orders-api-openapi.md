# Task: Write an OpenAPI 3.x Spec for an Orders API

Design an orders resource with a lifecycle and write its contract as an
**OpenAPI 3.x** specification. Save the spec as **`openapi.yaml`** in the
current workspace — that is the only artifact graded. There is no server, no
code, and no network: the file must be a complete, self-consistent, valid
OpenAPI document that a spec linter and code generator could consume.

## API surface

| Method | Path | Semantics |
|---|---|---|
| GET | `/orders` | list, **cursor-paginated**, filterable by `status`, sortable |
| POST | `/orders` | create an order; **requires `Idempotency-Key`** header; returns `201 Created` + `Location` |
| GET | `/orders/{id}` | fetch one order |
| PATCH | `/orders/{id}` | partial update (only the changed fields) |
| POST | `/orders/{id}/cancel` | domain action — cancels the order and triggers refunds |

## Requirements

1. **Metadata** — `info.title` and a non-empty `info.version`.
2. **GET /orders** — query parameters `cursor` (opaque pagination token) and
   `limit` (page size, with a documented default and maximum), a `status`
   filter, and a `sort` parameter (e.g. `-created_at` for descending).
3. **POST /orders** — a **required** `Idempotency-Key` header parameter: a
   client-supplied key that makes retries safe (a retried request with the same
   key must not create two orders). Include a request body describing the order
   being created.
4. **Responses** — every operation documents its responses, including at least:
   - `POST /orders` → `201 Created` with a `Location` header pointing at the
     new order.
   - `GET /orders` (and/or the other read operations) → `200 OK`.
   - `422 Unprocessable Entity` for well-formed but semantically invalid input
     (e.g. an order with zero items).
   - `429 Too Many Requests` for rate limiting (include `Retry-After`).
5. **Error envelope** — the error responses (at least `422` and `429`) use the
   RFC 7807 `application/problem+json` content type and reference a reusable
   `Problem` schema in `components.schemas` exposing at least `type`, `title`,
   `status`, and a stable machine-readable `code` property.
6. **Pagination schema** — the list response body uses a pagination envelope:
   a `data` array plus a `pagination` object exposing `next_cursor` and
   `has_more`.
7. **Security** — an OAuth2 **client credentials** security scheme
   (`securitySchemes.oauth2` with `flows.clientCredentials`) and
   least-privilege scopes `read:orders` and `write:orders`; apply the scopes to
   the endpoints (reads need `read:orders`, writes need `write:orders`).

## Conventions (from the api-design skill)

- Resources are plural nouns; the HTTP method says what happens. No verbs in URLs.
- POST for creates and domain actions; PATCH for partial updates. No PUT, no
  DELETE — orders are cancelled, not deleted.
- Pick the narrowest correct status code. Errors live in 4xx/5xx — never a
  `200` carrying an error body, never `500` for client mistakes.
- Cursor pagination for the live collection; validate filter and sort values
  against a whitelist (enum for `status`).
- Keep the spec realistic: a `servers` entry, a global `security` requirement,
  descriptions, and sensible examples.

The deterministic grader statically checks `openapi.yaml` only. Any
syntactically valid YAML that satisfies the requirements above passes —
quoting, formatting, and naming variations are not graded.
