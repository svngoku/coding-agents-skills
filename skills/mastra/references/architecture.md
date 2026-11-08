# Agent Architecture and Requirements

Use this guide before choosing Mastra primitives. The central design decision is not which model
to use, but which decisions require semantic judgment and which must remain exact. Examples follow
Mastra v1 APIs checked on 2026-11-08; see [sources.md](sources.md).

## Write the Behavioral Contract

Define these items before implementation:

- **Outcome:** observable user result, not "use an agent".
- **Actors:** authenticated identity, tenant, roles, and delegated permissions.
- **Inputs:** schemas, supported channels, size limits, locales, and data classification.
- **Outputs:** typed result, citations or receipts, uncertainty behavior, and user-facing errors.
- **Capabilities:** minimum tools and data each stage needs; explicit forbidden actions.
- **Completion:** deterministic acceptance conditions and maximum work per request.
- **Failure:** denial, no evidence, ambiguity, timeout, cancellation, partial completion, and recovery.
- **Operations:** retention, tracing, evaluation, deployment ownership, and incident response.

Turn the contract into examples: normal, boundary, adversarial, interrupted, and replayed requests.
These become evaluation fixtures rather than prompt tweaks discovered in production.

## Assign Decisions to the Right Layer

| Decision | Owner | Reason |
| --- | --- | --- |
| Intent classification or summarization | Model | Language variation and ambiguity |
| Exact amount, totals, or state transition | TypeScript | Arithmetic and invariants must be reproducible |
| Object-level permission | Application policy | Security cannot depend on model compliance |
| Evidence selection | Retrieval plus optional model ranking | Relevance is semantic; tenant filter is exact |
| Next step in a fixed process | Workflow | Known states and recovery semantics |
| Next tool in an open-ended task | Agent | Tool choice depends on context |
| Irreversible external effect | Policy, reviewer, idempotent service | Requires accountability and exactly-once business behavior |

A model proposal is untrusted input to the next deterministic boundary. Validate it again even if
it came from `structuredOutput`: a schema proves shape, not truth or permission.

## Start With the Smallest Sufficient Design

1. Use a plain function for deterministic transformations.
2. Use one agent for one open-ended decision with a small toolset.
3. Use a workflow when stages, branching, persistence, or recovery are known.
4. Put an agent inside a workflow step when only one stage needs semantic judgment.
5. Add subagents only for independently owned expertise, tools, context, or evaluation.
6. Add queues and workers when work must outlive a request or needs scheduled retries.

A workflow is not automatically better than an agent. Workflows make control visible; agents make
choice flexible. Hybrid designs usually provide the clearest production boundary.

## Reference Support Architecture

```text
HTTP request -> authentication -> tenant-scoped request context
                               -> ticket intake workflow
                                  1. validate and persist ticket
                                  2. triage agent (typed classification)
                                  3. retrieve authorized policy evidence
                                  4. deterministic refund eligibility
                                  5. approval gate when policy requires review
                                  6. idempotent refund service
                                  7. receipt, response, trace, evaluation sample
```

The triage agent cannot call the refund service. Only the later workflow step owns that capability.
This is stronger than giving one agent every tool and instructing it to be careful.

## State and Trust Boundaries

Separate four kinds of state:

| State | Example | Required protection |
| --- | --- | --- |
| Conversation | Thread messages, user preferences | Resource/thread ownership and retention |
| Shared knowledge | Policies, product documents | Tenant filter, provenance, version, access control |
| Execution | Workflow input, step output, suspension | Persistent snapshots and authorized resume |
| Business record | Ticket, refund, audit event | Transactional constraints, idempotency, authorization |

Do not use memory as an audit log or a workflow snapshot as a business ledger. Keep their retention
and access policies independent.

Treat these as untrusted: user text, uploaded documents, retrieved chunks, tool output, MCP metadata,
subagent summaries, and model-generated identifiers. Authentication of a transport does not make its
content true. Encode trust labels in application data where downstream decisions need them.

## Budgets and Backpressure

Set budgets per request and per tenant:

- Agent `maxSteps` and workflow iteration count
- Tool timeout, retries, and maximum result size
- Parallel and per-item concurrency
- Retrieval `topK`, final context tokens, and document size
- Queue depth and worker concurrency
- Overall deadline and cancellation behavior

Retry only transient failures with capped exponential backoff and jitter. Do not retry validation,
authorization, or deterministic business-rule failures. An idempotency key makes a safe retry
possible; it does not make every operation safe to retry.

## End-to-End Support Example

A good support/refund design has a small agent contract:

```typescript
import { z } from 'zod'

export const refundProposalSchema = z.object({
  ticketId: z.string().uuid(),
  eligible: z.boolean(),
  proposedAmountMinor: z.number().int().nonnegative().nullable(),
  currency: z.enum(['USD', 'EUR']).nullable(),
  evidenceIds: z.array(z.string()).max(5),
  rationale: z.string().max(500),
})
```

The application then verifies the ticket, calculates the permitted amount from the order ledger,
checks every evidence ID against the same tenant, and ignores a model-proposed amount when it
conflicts with policy. No evidence means an abstention or manual review, not a guessed refund.

Before execution, the reviewer sees the authoritative order ID, amount, currency, reason, and policy
version. The approval record binds those exact values. Resume reauthorizes the actor and verifies the
snapshot has not become stale. The payment adapter uses a stable idempotency key and stores the
provider receipt before reporting success.

## Architecture Review Questions

- What incorrect model output could cause harm, and which deterministic check stops it?
- Can a user, document, tool, or subagent change tenant or privilege scope?
- What happens when evidence is missing, contradictory, stale, or prompt-injected?
- Can every external write be retried without duplicating its business effect?
- Which state survives process restart, and who is allowed to resume it?
- Are limits enforceable in code rather than suggested in instructions?
- Can traces and evaluations reconstruct a failure without exposing sensitive content?
- Can the system be disabled or rolled back without orphaning active work?

If these answers are unclear, reduce autonomy before adding more framework features.
