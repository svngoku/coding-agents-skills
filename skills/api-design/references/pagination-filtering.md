# Pagination, Filtering, Sorting, and Field Selection

Deep dive for the [api-design](../SKILL.md) skill. Covers choosing and implementing cursor vs offset pagination, filtering/sorting syntax, sparse fieldsets, and response envelopes.

## Choosing a strategy

| Decision | Offset/page | Cursor |
|---|---|---|
| Stability under inserts/deletes | drifts (items can appear twice or be skipped) | stable — anchored to a position |
| Jump to page N | yes (page 3 of 50) | no — next/prev only |
| Implementation cost | `OFFSET x LIMIT y` — trivial | keyset predicates — more code, indexed |
| Performance on deep pages | degrades (OFFSET scans skipped rows) | constant-ish (index seek) |
| Sort flexibility | any | needs a stable sort key (unique column) |
| Caching | page N is stable while data is stable | token URLs are opaque, cache-friendly |

**Defaults:** cursor for public/live collections; offset/page for admin tables, small datasets, and "jump to page" UIs.

## Cursor pagination in practice

A cursor is an **opaque token** the server issues; clients must never decode or construct it. Base64-encode an internal value (id, or id+sort-key tuple) so the wire format stays stable even if internals change.

### Keyset SQL

For `GET /orders?limit=25` sorted by `created_at DESC, id DESC`:

```sql
SELECT * FROM orders
WHERE (created_at, id) < (:cursor_created_at, :cursor_id)
ORDER BY created_at DESC, id DESC
LIMIT 25;
```

The cursor encodes the last row's `created_at` + `id`; the predicate is a pure index seek — no full scan, stable even if new rows are inserted between pages.

### Response

```json
{
  "data": [ { "id": "ord_8f2", "total": "129.00" } ],
  "pagination": { "next_cursor": "eyJpZCI6Im9yZF84ZjIifQ", "has_more": true }
}
```

- `has_more` = "fetch one more row than requested, then report whether it existed".
- `next_cursor` is null/absent when `has_more` is false.
- Request the next page as `GET /orders?cursor=<token>&limit=25`.
- Keep `limit` client-controlled within [default, max]; the sort stays server-fixed per endpoint (or travels in the token).

### Cursor encoding

```python
import base64, json

def encode_cursor(row):
    payload = json.dumps({"id": row["id"], "created_at": row["created_at"]})
    return base64.urlsafe_b64encode(payload.encode()).decode()

def decode_cursor(token):
    return json.loads(base64.urlsafe_b64decode(token.encode()))
```

Opaque to clients, debuggable by the server team. Rotate the encoding (add a version prefix, sign it) if cursors might be forged.

## Offset pagination

`GET /orders?page=2&page_size=50` or `?offset=20&limit=10`.

- Fine for: admin tables, internal tools, static reports, datasets that do not change mid-paging.
- Problems: deep pages are slow (OFFSET scans), and concurrent inserts/deletes make pages drift or skip.
- Always respond with `total`/count when the UI needs "Page 3 of 50".

## Response envelopes

Pick one and stay consistent:

| Style | Shape | Notes |
|---|---|---|
| Wrapper | `{ "data": [...], "pagination": {...} }` | most common; keeps room for metadata |
| Bare array | `[...]` | simplest; metadata goes in headers — awkward for cursors |
| JSON:API | `{ "data": [...], "links": { "next": "..." }, "meta": { "total": 42 } }` | standard, heavier |

If you ever need the first page to carry top-level metadata (totals, facets), use the wrapper style from day one — adding a wrapper later is breaking.

## Filtering syntax

| Pattern | Syntax | Example |
|---|---|---|
| Exact match | `?field=value` | `?status=shipped` |
| Range | `?field_gte=`, `?field_lte=`, or `?field[gte]=` | `?created_at_gte=2024-01-01&created_at_lte=2024-12-31` |
| Membership | repeated params or CSV | `?status=shipped&status=cancelled` or `?status=shipped,cancelled` |
| Existence | `?has_field=true` | `?has_shipped_at=true` |
| Search | `?q=` (documented as fuzzy/full-text, not exact) | `?q=acme` |
| Nested fields | dot-path | `?billing.country=US` |

Rules:

- Whitelist every filterable field in the OpenAPI spec; unknown keys → `400` (silently ignoring filters is how data leaks in reports).
- Validate types and enum values — `?status=bogus` is a client error, not an empty result.
- Use ISO 8601 timestamps (`2024-01-01T00:00:00Z`) for ranges; document whether they are inclusive.
- Combine filters with AND; document if OR is supported.

## Sorting

- `?sort=field` ascending; `?sort=-field` descending; comma-separated multi-key: `?sort=-created_at,id` (id as tiebreaker for stable paging).
- Whitelist sortable fields; unknown sort keys → `400`.
- Document null ordering — e.g. `?sort=-updated_at` puts null `updated_at` last.
- Cursor pagination + sortable columns: the cursor must encode the full sort tuple, or the sort must be server-fixed for paginated endpoints.

## Sparse fieldsets (field selection)

`GET /orders?fields=id,status,total` returns only those fields. Benefits: smaller payloads on mobile, less sensitive data exposure, faster responses.

- Only selectable fields may be requested; unknown → `400`.
- Always include the resource `id` and fields required by the media type.
- Document that default (no `fields`) returns the full representation.

## Defaults and limits

| Setting | Recommendation | Notes |
|---|---|---|
| Default page size | 25–50 | balance payload size vs round-trips |
| Max page size | 100–1000 | protect the server from `?limit=999999` |
| Over-limit request | clamp or 400 | document which; clamping is friendlier |
| Zero/negative | 400 | never silently return empty |
| Invalid cursor | 400 or 410 | treat as client error; a stale cursor should not 500 |

## GraphQL note

GraphQL lists use Relay-style **connection** pagination: `edges { node, cursor }`, `pageInfo { hasNextPage, endCursor }`. The cursor is again opaque and keyset-based. The design principles (opaque token, has_more, stable sort) transfer directly.
