# Linkup Bulk Extract

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

The Extract endpoint turns a known web page into a table of structured records. Give it a seed URL plus a natural-language description of the rows you want, and it returns one JSON object per line (NDJSON), handling pagination automatically. It's built for 10s–1000s of records from a single listing page.

**Extract is in closed beta.** Access is limited; request it at contact@linkup.so. Parameters and response shape may change. If the rows you need are all on one page and you don't need link-following, use `linkup-fetch` with `schema` instead: it is generally available and costs $0.001 extra on a normal fetch.

This uses the REST API, so it needs `LINKUP_API_KEY`:

```shell
test -n "$LINKUP_API_KEY" || echo "Missing LINKUP_API_KEY"
```

## When to use Extract

| Use `linkup-extract` when... | Use instead... |
| --- | --- |
| You have one URL and want many structured rows that span pagination or detail pages (team, catalog, jobs, speakers) | — |
| You want one page's content as prose/Markdown | `linkup-fetch` |
| You want typed fields from one page, no link-following | `linkup-fetch` with `schema` |
| You need to find information across the web | `linkup-search` |
| You need synthesis across many sources | `linkup-research` |

## How to call it

Always provide a `schema` for production use (consistent output, clear required fields). Enable `verifyUrls` when extracted URLs will be used downstream (adds latency).

```shell
curl -sS -X POST "https://api.linkup.so/v1/extract" \
  -H "Authorization: Bearer $LINKUP_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "q": "All pricing plans with plan name, monthly price, annual price, features, and usage limits",
    "url": "https://competitor.com/pricing",
    "schema": {"type":"object","properties":{"planName":{"type":"string"},"monthlyPrice":{"type":"string"},"annualPrice":{"type":"string"},"features":{"type":"array","items":{"type":"string"}}}}
  }'
```

## Async lifecycle

`POST /v1/extract` returns `{id, status: "pending"}`. Poll `GET /v1/extract/{id}` about every 30s (crawls run longer than research). When `completed`, the output has a `resultUrl` (24h expiry) — download it and parse one JSON object per line.

If you don't know the URL yet, first discover it with `linkup-search` (`"Find the careers page URL for {company}"`), then extract from the result.

For the full parameter reference, patterns, and pricing notes, read `linkup-specialized-endpoints.md` (Extract section) in this skill's directory.
