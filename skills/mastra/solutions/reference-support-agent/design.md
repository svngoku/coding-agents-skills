# Support/Refund Agent Design

## Approval binding

The refund tool requires approval before its execute function performs a side effect. The approval
record is bound to the normalized order, amount, currency, refund request ID, trusted actor, tenant,
and idempotency key. A resumed run must compare that exact payload with the approved snapshot;
changed arguments require a new approval rather than mutating the approved request. The service
re-checks authorization on execution because approval does not replace authorization.

## Tenant authorization

The server authenticates the caller and populates actor, tenant, and thread values in request
context before invoking Mastra. Tools read only that trusted context, while model/client input can
identify an order but cannot supply identity or authorization. The order service scopes every read
by tenant, and the refund service independently authorizes the actor for that tenant and order.
Cross-tenant lookup, missing identity, and insufficient privilege fail without issuing a refund.

## Idempotency and replay

The idempotency key combines the trusted tenant with the client-visible refund request ID, so it is
stable across retries but isolated between tenants. The refund service enforces the key atomically
in its transactional store together with a hash of the normalized refund payload. A retry with the
same key and payload returns the original result; the same key with different payload is rejected.
Approval replay therefore cannot create a second charge reversal, and ambiguous service failures
are retried with the same key rather than a newly generated one.

## Snapshots and recovery

LibSQLStore provides durable Mastra storage rather than process-local memory. Stable tenant-scoped
resource and thread identifiers let conversation state resume, while committed workflow snapshots
record the normalized input, current step, and suspension state. After a process restart, the
application resumes the stored run instead of starting a fresh refund. A suspended approval is
reconciled with its bound payload; failed or exhausted runs require an explicit safe retry and never
blindly repeat a non-idempotent operation.

## Bounded execution

The agent has a maximum of four model/tool steps, which limits loops, cost, and repeated tool
attempts. Reaching that bound ends the run with a reviewable incomplete outcome; it is not treated
as approval or success. A human or a later explicitly bounded run can continue from durable state,
while the refund service's idempotency protection remains the final guard against duplicate writes.

## Untrusted tool and RAG data

Order records, tool output, user messages, and RAG/retrieved text are untrusted data, never system
instructions. Retrieved commands such as “ignore policy” or “approve this refund” have no authority.
The system prompt requires policy and authorization to come from trusted application code, schemas
validate tool boundaries, and output is treated as content rather than executable control flow.
Tools return only the minimum tenant-authorized fields needed for the response.

## Evaluation and tracing

Offline tests use fake order/refund services to cover tenant mismatch, denied authorization,
approval with changed arguments, duplicate retry, suspended restart, malformed tool data, and step
exhaustion without live model or payment calls. Traces should carry run, workflow, tenant, actor,
tool, status, and idempotency-key metadata plus latency and error category. They exclude secrets,
full message bodies, payment details, and unnecessary personal data. Separate model evaluations
measure structured-output validity, safe escalation, and whether the agent refuses injected
instructions; passing evaluations are required before enabling a new prompt or model version.
