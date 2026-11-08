# Human Approval and Interruption Handling

Human review adds accountability only when the reviewer sees authoritative action details and the
system binds their decision to those exact details. It is not a replacement for authentication,
authorization, validation, or idempotency. Mastra APIs below were checked on 2026-11-08.

## Gate a Sensitive Tool

```typescript
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'

export const issueRefund = createTool({
  id: 'issue-refund',
  description: 'Issue a previously authorized and reviewed refund.',
  requireApproval: true,
  inputSchema: z.object({
    refundId: z.string().uuid(),
    amountMinor: z.number().int().positive(),
    currency: z.enum(['USD', 'EUR']),
  }),
  outputSchema: z.object({ receiptId: z.string() }),
  execute: async (inputData, context) => {
    // Re-fetch actor and object, reauthorize, verify policy/version, then execute idempotently.
    return refunds.executeReviewed(inputData, context.requestContext.get('actor'))
  },
})
```

For request-wide gating, `requireToolApproval` can require approval for eligible tool calls. Prefer
narrow per-tool approval where only selected operations are consequential.

## Streaming Approval Lifecycle

A stream emits a `tool-call-approval` chunk. Persist the `runId` and `toolCallId`, then display the
authoritative tool name and complete arguments to the reviewer.

```typescript
// After the reviewer approves the exact pending call:
const resumed = await agent.approveToolCall({
  runId,
  toolCallId,
})
```

Use the corresponding decline method when rejected. Continue consuming the resumed stream and handle
subsequent approval, tripwire, error, and completion events. Never automatically approve a second call
because the first was approved.

For `.generate()`, a suspended call produces `finishReason: 'suspended'`; use the generate-specific
approval continuation such as `approveToolCallGenerate`. Check the installed API reference for the
exact return and continuation shape rather than treating stream and generate flows as interchangeable.

## Bind Approval to the Exact Action

Create a review record containing:

- run ID, tool call ID, tool name, and operation ID
- authenticated actor and target tenant/object
- amount, currency, destination, reason, and other consequential arguments
- policy and workflow definition versions
- cryptographic hash or immutable record of the exact reviewed payload
- creation/expiry, reviewer, decision, and decision timestamp

On continuation, compare every field. Reject when arguments, target, policy, actor permission, or
review validity changed. Approval of "refund $20" must not authorize a later mutation to $200 or a
different customer.

Do not render a model-written summary as the only review evidence. Fetch the current order and policy
from authoritative services and show both the proposed and current values.

## Agent Suspension and Resume

For custom interaction inside a tool, define `suspendSchema` and `resumeSchema`, then use
`context.agent.suspend(...)`. Return immediately after suspension. Persistent storage is required if
the suspended run must survive a restart.

```typescript
const clarificationTool = createTool({
  id: 'request-clarification',
  description: 'Ask for missing information before continuing.',
  inputSchema: z.object({ question: z.string().max(500) }),
  outputSchema: z.object({ answer: z.string().max(2000) }),
  suspendSchema: z.object({ question: z.string() }),
  resumeSchema: z.object({ answer: z.string().max(2000) }),
  execute: async (inputData, context) => {
    const { resumeData, suspend } = context.agent ?? {}
    if (!resumeData?.answer) {
      if (!suspend) throw new Error('Agent suspension is unavailable in this execution context')
      return suspend({ question: inputData.question })
    }
    return { answer: resumeData.answer }
  },
})
```

Resume a stream with `agent.resumeStream({ answer }, { runId })`, or a generate call with
`agent.resumeGenerate({ answer }, { runId })`. Automatic resumption can use the next message only when
memory, the same resource/thread, and a `resumeSchema` are configured; it does not approve a tool that
uses `requireApproval`. Keep this mechanism distinct from workflow step suspension.

Use `listSuspendedRuns()` to rediscover work after a restart. A resumption endpoint must look up the
run and authorize its owning resource; possession of a run ID alone is not permission.

## Workflow Approval Alternative

For known business processes, a dedicated workflow review step is often clearer than agent tool
approval. The workflow stores the immutable proposal and suspends; the reviewer resumes with a typed
decision; the next deterministic step reauthorizes and executes. See [workflows.md](workflows.md).

Choose tool approval for dynamic agent-driven action, workflow suspension for a fixed review stage.
Both require persistent state and application-owned review records for consequential effects.

## Decline, Expiry, and Cancellation

A complete lifecycle includes:

- **Decline:** record reviewer/reason, perform no side effect, return a clear terminal state.
- **Expiry:** invalidate old approval when policy or target state may have changed.
- **Duplicate decision:** return the recorded outcome without executing twice.
- **Cancellation:** stop active work and leave a recoverable or explicitly terminal record.
- **Timeout after execution:** reconcile by operation ID; do not assume rejection or failure.
- **Concurrent edit:** compare version or precondition before committing.

Approval cannot make an unauthorized action valid. Reauthorize on execution because roles, ownership,
and balances may have changed while waiting.

## Approval Tests

Test the real UI/API path for altered tool arguments, wrong run ID, wrong reviewer, expired review,
revoked permission, double-click approval, decline after approval, process restart, stale policy, and
provider timeout after the external write. Verify no action occurs in every denied path and one
business effect occurs across every valid retry.

## Approval Review Checklist

- [ ] Sensitive tools declare approval or a workflow gate.
- [ ] Reviewer identity and permission are independently verified.
- [ ] Exact authoritative arguments and policy version are displayed and bound.
- [ ] Resume revalidates target state and permissions.
- [ ] Persistent storage and suspended-run rediscovery are configured.
- [ ] Decline, expiry, duplicate, cancellation, and unknown outcomes are handled.
- [ ] Execution is idempotent and produces an auditable receipt.
