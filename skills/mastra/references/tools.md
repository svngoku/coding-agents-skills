# Tools, Validation, and Safe Actions

Tools are the boundary between probabilistic reasoning and application capabilities. Build each as a
narrow, typed, authorized operation. Current Mastra uses `createTool`; do not substitute plain objects
or older callback signatures. APIs checked on 2026-11-08.

## Define a Tool

```typescript
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'

export const getOrder = createTool({
  id: 'get-order',
  description: 'Return limited order details for a refund eligibility check.',
  inputSchema: z.object({
    orderId: z.string().uuid(),
  }),
  outputSchema: z.object({
    orderId: z.string().uuid(),
    status: z.enum(['paid', 'refunded', 'cancelled']),
    refundableMinor: z.number().int().nonnegative(),
    currency: z.enum(['USD', 'EUR']),
  }),
  execute: async (inputData, context) => {
    const tenantId = context.requestContext.get('tenantId') as string
    if (!tenantId) throw new Error('Missing authenticated tenant')

    const order = await orders.findForTenant(inputData.orderId, tenantId, {
      signal: context.abortSignal,
    })
    if (!order) throw new Error('Order not found')

    return {
      orderId: order.id,
      status: order.status,
      refundableMinor: order.refundableMinor,
      currency: order.currency,
    }
  },
})
```

The first `execute` argument is validated input data. The optional second context carries facilities
such as `requestContext`, `tracingContext`, and `abortSignal`. Do not destructure the first argument
as if it were the old context wrapper.

## Design the Model-Facing Contract

- Use a stable, specific ID and a description that states when and when not to call.
- Constrain formats, lengths, numeric ranges, enumerations, and array sizes with Zod.
- Accept opaque IDs rather than raw URLs, SQL, shell commands, or unrestricted file paths.
- Return only necessary fields, with maximum item counts and text lengths.
- Prefer domain operations such as `schedule-reply` over generic `execute-request`.
- Separate read and write tools; omit write tools from agents that only advise.

Schema descriptions should explain semantics, not attempt to implement authorization. Strict schemas
reduce malformed calls but do not stop a valid-shaped malicious request.

## Authorize at Execution Time

Derive `actorId`, `tenantId`, roles, and allowed capabilities in the server after authentication.
Store them in request context and enforce object-level access in the repository or service query.
Never accept an alternate tenant, role, or actor from model arguments without independent policy.

```typescript
async function authorizeRefund(
  actor: { id: string; tenantId: string; permissions: string[] },
  orderId: string,
) {
  if (!actor.permissions.includes('refunds:request')) {
    throw new Error('Refund permission denied')
  }
  const order = await orders.findForTenant(orderId, actor.tenantId)
  if (!order) throw new Error('Order not found')
  return order
}
```

Return the same public "not found" behavior where existence disclosure matters. Log a safe internal
reason without returning secrets or another tenant's data to the model.

## Idempotent Writes and Partial Failure

A durable workflow can replay a step after the external system succeeded but before its response was
stored. Every write adapter therefore needs a stable idempotency key derived from the business intent,
not from the current attempt number.

```typescript
const idempotencyKey = `refund:${refundId}`
const receipt = await payments.refund({
  orderId,
  amountMinor,
  currency,
  idempotencyKey,
}, { signal: context.abortSignal })
```

The payment service must enforce uniqueness for that key and return the original result on replay.
Store the local intent before calling, persist the provider receipt afterward, and reconcile unknown
outcomes instead of blindly creating a new refund. For non-idempotent providers, use their query or
reconciliation API before any retry.

A timeout means "outcome unknown", not necessarily "failed". Surface that distinction to recovery
logic. Compensating actions are separate business operations, not automatic erasure of history.

## Errors, Timeouts, and Cancellation

Classify errors for the next component:

| Class | Examples | Behavior |
| --- | --- | --- |
| Invalid input | Schema or domain validation | Do not retry; return actionable validation error |
| Denied | Missing permission or object access | Do not retry; record safe audit event |
| Not found | Authorized lookup has no result | Return bounded absence result or domain error |
| Transient | Rate limit, temporary unavailable | Bounded retry with jitter if safe |
| Unknown outcome | Timeout after write submission | Reconcile by idempotency key before retry |
| Cancelled | Caller disconnect or shutdown | Stop promptly; preserve recoverable state |

Do not include raw provider responses, stack traces, credentials, or full documents in model-facing
errors. Trace safe error codes and let operators inspect protected internal logs when necessary.

## Approval-Sensitive Tools

Add `requireApproval: true` to consequential tools and handle the approval lifecycle explicitly.

```typescript
export const issueRefund = createTool({
  id: 'issue-refund',
  description: 'Issue the already authorized refund exactly as reviewed.',
  requireApproval: true,
  inputSchema: z.object({
    refundId: z.string().uuid(),
    orderId: z.string().uuid(),
    amountMinor: z.number().int().positive().max(100_000),
    currency: z.enum(['USD', 'EUR']),
  }),
  outputSchema: z.object({ receiptId: z.string(), status: z.literal('submitted') }),
  execute: async (inputData, context) => {
    // Reauthorize actor and object here; approval alone is not authorization.
    return refundService.executeAuthorized(inputData, {
      actor: context.requestContext.get('actor'),
      signal: context.abortSignal,
    })
  },
})
```

Bind approval to the exact reviewed arguments and invalidate it if any argument or policy version
changes. See [human-in-the-loop.md](human-in-the-loop.md).

## Tool Output as Untrusted Input

External output can contain prompt injection, malicious links, oversized content, or stale data.
Return typed fields, escape it for the destination UI, cap collection sizes, preserve source metadata,
and never execute a string returned by a tool. Tool annotations and MCP declarations are hints, not
security proof.

## Tool Review Checklist

- [ ] Current `createTool` API and callback signature are used.
- [ ] Input and output schemas enforce practical bounds.
- [ ] Server-derived actor and tenant are enforced in the data query.
- [ ] Tool privileges and returned fields are minimal.
- [ ] Timeout, abort, error, and empty-result behavior are explicit.
- [ ] Writes have stable idempotency and reconciliation behavior.
- [ ] Approval is exact-argument bound and execution reauthorizes.
- [ ] Output is bounded and treated as untrusted content.
