# Versioning, Backward Compatibility, and Deprecation

Deep dive for the [api-design](../SKILL.md) skill. Covers choosing a versioning strategy, what counts as breaking, and how to deprecate without breaking consumers.

## Philosophy: evolve additively, version rarely

The cheapest version bump is the one you never make. Treat every released API as a contract: prefer additive changes (new fields, new endpoints, new optional parameters) that old clients can ignore. Reserve a new version for changes old clients *cannot* ignore.

## Strategy comparison

| Strategy | Example | Pros | Cons |
|---|---|---|---|
| URL path | `https://api.example.com/v2/orders` | visible in logs, cacheable per version, trivial routing | duplicate code paths; version lingers forever |
| Media type | `Accept: application/vnd.api+json; version=2` | URLs stay clean, version is part of content negotiation | hidden — hard to inspect/debug; extra client work |
| Custom header | `X-API-Version: 2` | explicit | non-standard header; easy to forget on the client |
| Query param | `/orders?v=2` | trivial to add | pollutes cache keys and logs; URLs are not stable identifiers; widely discouraged |

**Recommendation:** URL path for public APIs (visible, cacheable, debuggable); media-type only when you already do content negotiation and want version-per-representation (rarely worth it). Never query-param version a public API.

## What counts as breaking

### Breaking — requires a new version

- Removing or renaming a field, endpoint, or error `code`
- Changing a field's type, format, or meaning (string → number, ID format, date timezone)
- Making a previously-optional field required, or tightening validation old payloads would fail
- Removing values from an enum (a client sending `shipped` now gets 400)
- Changing status codes clients branch on (422 → 400)
- Changing pagination shape or semantics
- Reordering/renaming response fields that clients access by position (rare but real)

### Additive — safe without a version bump

- New optional field in a response (old clients ignore it)
- New endpoint, new HTTP method on an existing path
- New optional query parameter or request field
- Widening an enum (adding values) — old clients never send the new values
- Relaxing validation
- New error codes (old clients fall through to their default handler)

### Ambiguous — document your choice

- Changing `detail`/error message copy: fine, clients must not parse it.
- Increasing default page size: fine; decreasing it is breaking-ish (payload contract).
- Fixing a bug where responses were wrong: still a contract change if clients depend on the bug — announce it, or version it.

## The deprecation workflow

1. **Announce** — changelog, developer email, API docs banner: what changes, when, and the migration path. Give a minimum of months (Stripe: 1+ year; common minimum: 6–12 months).
2. **Mark it in the spec** — OpenAPI `deprecated: true` on the path/operation/parameter; keep serving it.
3. **Signal it at runtime** — emit response headers on the old version so callers see it in their own logs:

```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: Thu, 31 Dec 2026 23:59:59 GMT
Link: <https://api.example.com/v2/orders>; rel="successor-version"
```

- `Deprecation: true` — this version is deprecated.
- `Sunset` (RFC 8594) — the date the old version stops working; clients should migrate before it.
- `Link` successor — where to migrate to.

4. **Monitor and migrate** — watch traffic on the old version; work with large consumers; publish migration guides per breaking change.
5. **Cut over** — after the sunset date, serve 410 Gone (with a successor link) rather than failing silently.

## A v1 → v2 walkthrough

**v1 shipped:** `GET /orders` returns `{ "id": 1, "status": "shipped", "amount": 12.5 }`; `amount` is a float, `id` is an integer.

**v2 changes:** monetary amounts become integer cents (`amount_cents`), IDs become strings (`ord_1`), and `status` gains a value.

| | v1 (frozen) | v2 (new) |
|---|---|---|
| URL | `/api/v1/orders` | `/api/v2/orders` |
| id | `1` (int) | `"ord_1"` |
| amount | `12.5` (float USD) | `amount_cents: 1250` |
| status | created, shipped | created, confirmed, shipped, cancelled |

Migration mechanics:

- Keep v1 routes alive (same code path or a compatibility shim) until sunset.
- New code targets v2; v1 handlers map v2 internal models back to the v1 shape.
- Deprecation headers on v1 responses from day one of v2's release.
- Publish a migration guide: "if you read `amount`, divide `amount_cents` by 100; if you stored `id` as int, use the string form."

## Code organization

Two endpoints, one backend is the pragmatic default:

- **Shared core** — domain logic knows nothing about versions.
- **Per-version adapters** — each version's request parsing and response rendering live separately.
- **Version in the route, not in the handler** — the version is selected by URL prefix, then the adapter for that version runs.

A single endpoint that switches on a version header (or worse, branches inside handlers) becomes unmaintainable after two versions.

## Anti-patterns

- **Versioning everything "just in case"** — an unused v1 is dead weight; evolve additively until a real breaking change appears.
- **Silent breaking changes** — renaming fields, tightening validation, or changing semantics inside an existing version breaks clients with no signal.
- **Deprecating without a sunset** — "v1 is deprecated" with no date is a permanent threat, not a plan.
- **Giving the old version no migration path** — every breaking change needs a documented "old → new" mapping.
- **Query-param versioning** — unstable URLs, polluted caches.
- **Versioning error codes out of sync** — if v2 changes error codes, version them too, and document the mapping.
