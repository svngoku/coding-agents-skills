# Rubric: Orders API OpenAPI Spec — score 0.0–1.0

Judge the agent's `openapi.yaml` as a production API contract. The
deterministic checks verify that the required surface exists; this rubric
scores the quality of the design decisions. Score the five dimensions below
(0.0–0.2 each) and sum, or judge holistically against the anchors.

## REST semantics & resource modeling (0.0–0.2)
- Plural-noun resources; one method per operation: GET reads, POST creates,
  PATCH partial-updates, POST to a named action sub-resource for `cancel`.
- No verbs in URLs (`/createOrder`, `/getOrders`), no deep nesting, no
  `PUT`/state-changing `GET` where PATCH/POST belong.
- `Idempotency-Key` is required on creating/mutating POST and PATCH — the
  retry-safety story is documented, not just declared.

## Status-code correctness (0.0–0.2)
- `201 Created` + `Location` on create; `200` for reads/updates/actions;
  `404` for an unknown resource.
- `422` for well-formed-but-invalid input; `429` with `Retry-After` for rate
  limits; no `200`-with-error-body; no `500` reserved for client mistakes.

## Error envelope consistency (0.0–0.2)
- Error responses use `application/problem+json` referencing one shared
  `Problem` schema — the same envelope on 422, 429, 404, 409 alike.
- `Problem` carries `type`/`title`/`status` plus a stable machine-readable
  `code` clients can branch on; no branching on human-readable `detail`.

## Pagination & versioning choices (0.0–0.2)
- Cursor pagination with an opaque `cursor` and `limit` (default and maximum
  documented), and a `next_cursor`/`has_more` envelope in the list response.
- `status` filter and `sort` are documented and validated (enum/whitelist);
  the versioning strategy (URL path or media type) is explicit and consistent.

## Spec realism & polish (0.0–0.2)
- Valid, complete OpenAPI 3.x structure: `openapi`, `info`, `servers`,
  `paths`, `components`.
- OAuth2 client-credentials scheme with least-privilege `read:orders`/
  `write:orders` scopes actually applied to the endpoints.
- Request/response schemas are defined (not just `description`), with sensible
  examples; descriptions explain behavior where it is not obvious.
