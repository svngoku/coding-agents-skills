# Workflows, Durability, and Recovery

Use workflows for known stages, explicit control flow, and recoverable execution. Use agents inside
steps only where semantic choice is needed. Examples use `createWorkflow` and `createStep` from
`@mastra/core/workflows`, checked on 2026-11-08.

## Define Typed Steps

```typescript
import { createStep, createWorkflow } from '@mastra/core/workflows'
import { z } from 'zod'

const intakeSchema = z.object({
  ticketId: z.string().uuid(),
  message: z.string().min(1).max(10_000),
})

const classificationSchema = z.object({
  category: z.enum(['billing', 'technical', 'other']),
  needsReview: z.boolean(),
})

const classify = createStep({
  id: 'classify',
  inputSchema: intakeSchema,
  outputSchema: classificationSchema,
  execute: async ({ inputData }) => {
    const response = await triageAgent.generate(inputData.message, {
      maxSteps: 2,
      structuredOutput: {
        schema: z.object({
          category: classificationSchema.shape.category,
          needsReview: z.boolean(),
        }),
      },
    })
    return response.object
  },
})

export const supportWorkflow = createWorkflow({
  id: 'support-workflow',
  inputSchema: intakeSchema,
  outputSchema: classificationSchema,
})
  .then(classify)
  .commit()
```

A step's `execute` receives an object containing `inputData` and other execution facilities. This is
different from a tool's `execute(inputData, context)` signature. Give each step a stable ID and narrow
input/output schema; workflow composition passes typed output forward.

## Run and Inspect the Status Union

```typescript
const run = await supportWorkflow.createRun()
const result = await run.start({
  inputData: { ticketId, message },
})

switch (result.status) {
  case 'success':
    return result.result
  case 'failed':
    throw new Error(`Support workflow failed: ${result.error}`)
  case 'suspended':
    return { pending: true, suspended: result.suspended }
  case 'tripwire':
    return { blocked: true, reason: result.tripwire }
  case 'paused':
    return { paused: true }
}
```

Treat status as a discriminated union. Do not read `result.result` before checking for success. Preserve
run IDs for asynchronous clients and provide an authorized status endpoint rather than holding every
HTTP request open.

## Control Flow

| Construct | Use | Important behavior |
| --- | --- | --- |
| `.then(step)` | Sequential dependency | Previous output becomes next input |
| `.parallel([a, b])` | Independent work | Outputs are keyed by step ID |
| `.branch([...])` | Conditional route | Only selected branches run; output is keyed by step ID |
| `.map(...)` | Transform workflow data | Keep transformations deterministic |
| `.foreach(step)` | Process a collection | Defaults to concurrency 1 and returns an array |
| `.dountil(...)` / `.dowhile(...)` | Bounded iteration | Add explicit maximum iterations and stop conditions |

Parallel and branch outputs require lookup by the configured step ID. Start with foreach concurrency
1; increase only after downstream capacity, idempotency, and rate limits are measured. A loop without
an application-level bound is unsafe even if individual agents have `maxSteps`.

## Persistent Snapshots

Configure a Mastra storage provider for workflows that suspend, resume, or recover across restarts.
In-memory snapshots vanish on process exit. A production store must be shared by every instance that
may resume the run and protected by backup, access control, and retention policies.

Persist only necessary state. Secrets should remain in a secret manager; snapshot fields and metadata
can otherwise become another sensitive-data copy. Version workflow definitions and snapshot data so a
new deployment can safely handle old suspended runs.

## Suspend and Resume

A suspending step declares `resumeSchema`, optionally `suspendSchema`, and receives `resumeData` and
`suspend` in its execution context.

```typescript
const reviewStep = createStep({
  id: 'review-refund',
  inputSchema: refundProposalSchema,
  outputSchema: refundReceiptSchema,
  suspendSchema: z.object({ reviewUrl: z.string().url() }),
  resumeSchema: z.object({ decision: z.enum(['approve', 'reject']) }),
  execute: async ({ inputData, resumeData, suspend }) => {
    if (!resumeData) {
      return suspend({ reviewUrl: await createReview(inputData) })
    }
    if (resumeData.decision === 'reject') {
      throw new Error('Refund rejected by reviewer')
    }
    return executeReviewedRefund(inputData)
  },
})
```

Return immediately after `suspend(...)`. Resume with the matching step and validated data:

```typescript
const result = await run.resume({
  step: 'review-refund',
  resumeData: { decision: 'approve' },
})
```

The application must authenticate and authorize the reviewer, bind the resume to the correct run and
pending action, reject stale or duplicate resumes, and revalidate policy before execution. Workflow
resume data is not proof of reviewer identity.

Tool-driven agent suspension uses a related but distinct mechanism; see
[human-in-the-loop.md](human-in-the-loop.md).

## Idempotency and Replay

A snapshot records what Mastra knows, not necessarily what an external provider completed. The process
may stop after a payment succeeds and before the receipt is persisted. Design each side effect for replay:

1. Persist a local operation ID and intended arguments.
2. Derive a stable idempotency key from that operation ID.
3. Call a provider that enforces idempotency.
4. Persist the external receipt and final state atomically where possible.
5. On unknown outcome, query/reconcile by idempotency key before retrying.

Do not generate a new idempotency key on each attempt. Do not assume workflow durability provides
exactly-once external effects.

## Failure and Recovery Tests

Exercise the actual workflow entry point for:

- validation failure before any side effect
- agent or retrieval failure in an intermediate step
- denial and guardrail tripwire
- suspension followed by authorized, rejected, stale, and duplicate resume
- process restart while suspended
- process termination immediately after external write
- repeated delivery of the same request
- branch and parallel partial failures
- concurrency limit and downstream rate limiting

Verify both business state and user-visible status. A green workflow status is insufficient if a
duplicate payment or orphaned approval remains possible.

## Workflow Review Checklist

- [ ] Every step and workflow has typed input/output and a stable ID.
- [ ] Known control flow is explicit and every loop/fan-out is bounded.
- [ ] Status is checked before status-specific result fields are read.
- [ ] Persistent shared storage is configured for recovery.
- [ ] Resume is authenticated, authorized, exact-action bound, and revalidated.
- [ ] External writes are idempotent and unknown outcomes are reconciled.
- [ ] Snapshot schema upgrades and retention are planned.
- [ ] Replay, interruption, and partial-failure tests cover real side effects.
