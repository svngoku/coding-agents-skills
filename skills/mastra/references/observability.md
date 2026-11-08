# Observability and Operational Monitoring

Trace the complete execution path while minimizing sensitive data. Use traces to explain behavior,
metrics to detect regressions, and application audit records for consequential business actions.
Observability APIs below were checked on 2026-11-08.

## Configure Tracing

```bash
npm install @mastra/observability
```

```typescript
import { Mastra } from '@mastra/core'
import { MastraStorageExporter, Observability } from '@mastra/observability'

export const mastra = new Mastra({
  environment: 'production',
  observability: new Observability({
    configs: {
      default: {
        serviceName: 'support-agent',
        sampling: { type: 'ratio', probability: 0.1 },
        requestContextKeys: ['tenantId', 'experimentId'],
        exporters: [new MastraStorageExporter()],
        serializationOptions: {
          maxStringLength: 2048,
          maxDepth: 5,
          maxArrayLength: 25,
          maxObjectKeys: 30,
        },
      },
    },
  }),
})
```

Mastra traces agents, model calls, tools, memory, workflows, control flow, and nested operations.
Choose always/never/ratio/custom sampling per environment. Keep higher-fidelity local development
separate from production. Storage and exporters must support the fields you expect to query.

A response `traceId` is available only when tracing is enabled and the request is sampled. Do not make
business logic depend on its presence. Propagate valid external trace/span IDs when joining an existing
OpenTelemetry or other tracing system.

## Protect Inputs, Outputs, and Metadata

```typescript
const result = await agent.generate(input, {
  requestContext,
  tracingOptions: {
    tags: ['production', 'support-v2'],
    metadata: { operationType: 'ticket-triage', workflowVersion: '3' },
    hideInput: true,
    hideOutput: true,
  },
})
```

`hideInput` and `hideOutput` apply trace-wide and filter data at export time. Use both for sensitive
operations. The default Sensitive Data Filter can redact common fields, but test it against actual
payloads and exporters. Configure serialization limits to reduce oversized trace data.

Extract only safe identifiers from request context. Do not add access tokens, raw messages, full order
records, payment details, or arbitrary user-controlled values as metadata. Tags are string labels;
metadata is structured. Keep cardinality bounded for filtering and cost control.

When adding child spans, pass request context only when its extracted metadata is appropriate. End spans
on success and record safe error categories on failure. Do not place raw queries, results, or exceptions
containing secrets into span input/output.

## Interpret Outcomes Accurately

Span `outcome`, `stopReason`, and compact `stopReasonContext` distinguish:

| Situation | Expected outcome/reason |
| --- | --- |
| Completed work | `success` |
| Needed work beyond allowance | `limit` / `max_steps` |
| Run deadline | `limit` / `timeout` |
| Caller cancellation | `abort` / `caller_cancelled` |
| Descendant stopped by cancellation | `interrupted` / `caller_cancelled` |
| Guardrail stopped execution | `blocked` / `guardrail` |
| Intentional wait | `suspend` |
| Observer failure interrupted run | `interrupted` / `observer_failed` |

Completing on the last permitted step is success; needing another step after exhausting the allowance
is a limit. A null outcome does not necessarily mean an unfinished span in older observability data;
check `endTime` and storage-version behavior.

Do not reduce every non-success to "provider error". Alerts and dashboards should distinguish denial,
guardrail block, suspension, cancellation, limit, application failure, and unknown outcome.

## Monitor System and Business Health

| Signal | Why it matters |
| --- | --- |
| Request/run rate and p50/p95/p99 latency | Capacity and user experience |
| Model tokens, estimated cost, context size | Budget and prompt/retrieval regressions |
| Step-limit, timeout, abort, and retry rates | Runaway loops and dependency health |
| Tool calls/errors by safe tool name | Contract or provider drift |
| Retrieval empty rate, source count, rerank latency | Grounding and corpus health |
| Tripwires by processor and safe category | Attack or policy changes |
| Suspended-run age and approval decline/expiry | Review queue and stale actions |
| Idempotent replays and unknown external outcomes | Duplicate-effect risk |
| Memory observer/buffering failures | Context loss and shutdown risk |
| Eval gate/scorer trends by version | Quality and model drift |

For consequential actions, keep a protected audit table with operation ID, actor, reviewer, action
fingerprint, policy version, idempotency key, and receipt. Traces are diagnostic evidence and may be
sampled or retained for a shorter period; they are not the sole audit ledger.

## Debug With a Trace

1. Start with root outcome, stop reason, versions, and safe tenant/operation IDs.
2. Inspect workflow/agent steps for unexpected branching, delegation, or repeated calls.
3. Inspect model usage and processed chunks without assuming hidden raw input is available.
4. Inspect tool latency, validation/error category, and retry sequence.
5. Inspect retrieval and memory operations for empty, stale, or oversized context.
6. Correlate application operation IDs and provider receipts for unknown side effects.
7. Reproduce locally with sanitized fixtures and add the failure to the eval set.

MCP client/server traces remain separate but linked. Follow both sides when diagnosing external tool
failures, while respecting each system's data-access policy.

## Operational Tests and Alerts

Verify that sensitive input/output never reaches exporters, sampled-out requests behave normally,
trace export failure does not silently corrupt business state, shutdown flushes required telemetry,
and high-cardinality metadata cannot be injected by users.

Page on actionable conditions such as repeated unknown payment outcomes, cross-tenant authorization
test failures, sustained tool/provider outage, or oldest approval exceeding its safe lifetime. Use
lower-severity trends for gradual cost or quality drift.

## Observability Review Checklist

- [ ] Agent, workflow, model, tool, retrieval, memory, and nested paths are attributable.
- [ ] Sampling, storage, exporters, retention, and flush behavior are configured.
- [ ] Sensitive inputs/outputs and metadata are minimized, hidden, and tested at export.
- [ ] Outcomes distinguish success, limits, aborts, interruptions, blocks, and suspension.
- [ ] Business audit state exists independently of sampled traces.
- [ ] Dashboards and alerts cover reliability, quality, cost, review age, and unknown writes.
- [ ] Every production incident becomes a sanitized regression case where possible.
