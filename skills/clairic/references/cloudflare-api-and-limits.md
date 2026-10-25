# Official Cloudflare API and Limits

Checked **2026-10-26** against [the official model page](https://developers.cloudflare.com/workers-ai/models/clairic/),
[input schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-input.json),
[output schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-output.json), and platform docs. The service is
**preview**. SDK details below apply specifically to unofficial `clairic==0.1.0`.

## Pick the Interface Before Writing Code

| Concern | Unofficial Python SDK | Official REST | Official Workers binding |
| --- | --- | --- | --- |
| Invocation | `client.extract(text, ExtractionClass)` | POST to the model's run endpoint | `env.AI.run(modelId, input)` |
| Field declaration | Pydantic answer types + specs | Raw `fields` object | Raw `fields` object |
| Authentication | SDK account/token env or kwargs | Account ID + bearer API token | Configured `AI` binding |
| Answers | `result.data.total.value` | `payload["result"]["answers"]["total"]["value"]` | `response.answers.total.value` |
| Images / output options | Not exposed | Supported by model-specific schema | Supported by model-specific schema |

The official model identifier is **`@cf/cloudflare/clairic`**. The input body
also has `model`, with `"clairic"` or the full identifier accepted by its schema.
Some launch examples omit it, but the published schema marks it required; include
it in new requests. The SDK always sends `"clairic"`.

## REST

Real requests are metered. Use an account-scoped Workers AI token, not a token
embedded in a browser or committed file. The [run endpoint](https://developers.cloudflare.com/api/resources/ai/methods/run/) documents Workers
AI Read / Workers AI Write permissions; use the platform's token flow and scope
it to the required account.

```bash
curl --fail-with-body \
  "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clairic" \
  -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "clairic",
    "state": "Invoice INV-42. Total USD 10.",
    "fields": {
      "invoice_number": {"type": "text"},
      "total": {"type": "number", "unit": "USD", "decimals": 2}
    }
  }'
```

Send the model-specific body directly, not `{"input": {...}}`. REST wraps the
model response in `{"success": true, "errors": [], "messages": [], "result": ...}`.
Check HTTP/envelope success before reading `result.answers`. The official model
page's illustrative output is the **unwrapped model result**, not the full REST
envelope. See [REST setup](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).

The official model-page sample uses `CLOUDFLARE_AUTH_TOKEN`; this example uses
`CLOUDFLARE_API_TOKEN` consistently with the unofficial SDK. Environment variable
names are application wiring, not a difference in bearer-token format.

## Workers Binding

In Wrangler configuration:

```json
{"ai": {"binding": "AI"}}
```

Inside a configured JavaScript Worker:

```javascript
export default {
  async fetch(_request, env) {
    const response = await env.AI.run("@cf/cloudflare/clairic", {
      model: "clairic",
      state: "Invoice INV-42. Total USD 10.",
      fields: {
        invoice_number: { type: "text" },
        total: { type: "number", unit: "USD", decimals: 2 },
      },
    });
    return Response.json(response);
  },
};
```

This is the **official platform binding**, not `new Clairic()` from an invented
JavaScript package. The binding returns model output directly and does not need
an explicit bearer token in the call. Verify current Workers types when using
TypeScript. [Binding documentation](https://developers.cloudflare.com/workers-ai/configuration/bindings/).

## Features the SDK Does Not Expose

The official input schema supports:

- `images`: up to four PNG/JPEG/WebP inputs, encoded as data URLs or
  `{content_type, base64}` objects. Remote image URLs are not accepted.
- Image limits: 4 MiB per image, 8 MiB decoded image data total, 16 MiB request
  body. Base64 overhead still affects body size.
- `options.include_quantiles: true`: percentiles 1–99 for supported distributions.
- `options.return_state_text: true`: `state_text` for span-reference text.
- `reference_date`: an ISO date anchor.

SDK 0.1.0 exposes only the text/model-class/date extraction arguments. It can
parse quantiles but cannot request them; it cannot request or expose `state_text`
through its public result. Use raw REST/binding when these features are required.
Images are not evidence that the service accepts arbitrary PDF/DOCX files.

No Clairic-specific `stream` request/response is documented in the inspected
schemas. Generic binding examples using `stream: true` for another model do not
establish Clairic streaming support.

## Context, Passes, and Cost

As checked on 2026-10-26:

- Model context: **65,536 tokens**.
- Fields can be split into parallel **8,192-token passes**, each reading the
  input. One SDK call does not imply one pass or one reading of the state.
- Model price: **$0.10 per million input tokens**, no output-token charge.
  Pricing also lists **9,091 neurons per million input tokens**.
- Workers AI free allowance: **10,000 neurons/day**, reset 00:00 UTC. Beyond
  allocation requires Workers Paid, listed at **$0.011/1,000 neurons**.

Refresh [pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) before quoting costs. Use returned
`usage.input_tokens` and monitor `usage.passes`; do not estimate charges from
`state_tokens` alone or promise a fixed relation between field count and cost.
Free allowance does not mean unmetered extraction or free retries.

The [limits page](https://developers.cloudflare.com/workers-ai/platform/limits/) gives generic text-generation limits and warns that
preview/beta models may be lower. It does not establish a dedicated Clairic
quota or SLA. Do not promise a model-specific RPM number from another category.
Wrangler local-mode inference still counts toward service usage; use mocks for
offline tests.

## Reliability and Safety Boundaries

- The model is preview/as-is; availability and behavior can change. Planned
  open weights are not evidence that weights are already released or locally
  supported by this SDK.
- Extraction and estimates can be wrong. Extract operands and calculate totals
  in code rather than trusting long arithmetic chains or row aggregation.
- Give estimates useful evidence/comparables; evaluate forecast accuracy on
  held-out data. Basic forecasts can miss holidays or flatten over long horizons.
- Validate units, date context, required evidence, and business rules before
  acting. A confidence value is not authorization or verified account ownership.
- Real inputs go to Cloudflare. Minimize sensitive data and keep credentials
  server-side. No inspected source establishes a Clairic-specific zero-retention,
  residency, compliance, or prompt-injection-immunity guarantee.
- The published raw output schema is not the Python result contract. For example,
  SDK `Usage` requires six fields while the raw schema requires only two.

See [extraction concepts](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/) and the [source/compatibility
ledger](sources-and-compatibility.md) for evidence and known documentation gaps.
