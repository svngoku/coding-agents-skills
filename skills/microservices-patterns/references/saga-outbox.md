# Sagas, Compensation, and the Outbox Pattern

Deep dive on keeping data consistent across services without distributed transactions. Read this when designing a flow that spans services and needs atomicity-in-the-large, or when events must be published reliably.

## Table of Contents
1. [Saga fundamentals](#saga-fundamentals)
2. [Choreography vs orchestration](#choreography-vs-orchestration)
3. [Compensating actions](#compensating-actions)
4. [The outbox pattern](#the-outbox-pattern)
5. [Idempotent consumers](#idempotent-consumers)
6. [Checklist](#checklist)

---

## Saga fundamentals

A saga is a sequence of local transactions T1...Tn with compensating transactions C1...Cn (one per Ti, run in reverse). Each participant commits locally and publishes an event; the next participant reacts. If Tk fails, the saga runs Ck-1...C1 to undo the partial work.

Properties:
- No global lock, no coordinator (in choreography) → high availability.
- **Eventual consistency**: an operation completes, but the system converges over time.
- Failures are handled by **business compensation**, not database rollback.

Design questions to answer before building:
1. What is the saga's success condition? (e.g., order confirmed)
2. What is each step's compensating action? (cancel, refund, restock)
3. What happens if a step times out? (timeout → run compensation)
4. Is every step idempotent? (retries and redeliveries are guaranteed)

## Choreography vs orchestration

### Choreography

Each participant publishes events and subscribes to the events it cares about. No central coordination.

Pros: minimal infrastructure, participants decoupled, easy to add new subscribers.
Cons: the flow is implicit and spread across services; tracing what happened requires reading every participant; failure handling is distributed and easy to get subtly wrong.

Good for: simple chains with few participants (2–4) and stable business rules.

### Orchestration

A central orchestrator (or state machine) issues commands to participants and tracks outcomes. The orchestrator is the only component that understands the whole flow.

Pros: the flow is explicit and auditable in one place; compensation logic is centralized; easier to add steps, timeouts, and retries.
Cons: one more service to operate; participants are coupled to orchestrator commands; the orchestrator itself must be resilient (persist its state).

Good for: complex flows, many participants, changing business rules, or flows where visibility matters (payments, onboarding).

Example orchestrator as an explicit state machine:

```typescript
enum SagaState { AWAITING_PAYMENT, AWAITING_INVENTORY, CONFIRMED, COMPENSATING, REFUNDED }

class CheckoutSaga {
  private state: SagaState = SagaState.AWAITING_PAYMENT;
  constructor(private orderId: string) {}

  async onPaymentCaptured(event: PaymentCaptured) {
    if (this.state !== SagaState.AWAITING_PAYMENT) return; // ignore stale/duplicate
    this.state = SagaState.AWAITING_INVENTORY;
    // inventory reservation happens asynchronously; wait for its event
  }

  async onInventoryReserved(event: InventoryReserved) {
    if (this.state !== SagaState.AWAITING_INVENTORY) return;
    this.state = SagaState.CONFIRMED;
    await this.publish(new OrderConfirmed(this.orderId));
  }

  async onInventoryFailed(event: InventoryFailed) {
    if (this.state === SagaState.CONFIRMED) return;
    this.state = SagaState.COMPENSATING;
    await this.publish(new PaymentRefundRequested(this.orderId)); // compensation
  },
}
```

Persist the saga state per aggregate (order id) so a restart resumes where it left off.

## Compensating actions

Rules:
- Compensate in **reverse order** of the steps that succeeded.
- Compensations must be **idempotent** — a saga can fail mid-compensation, and redelivered events may retry them.
- Compensations are **business-level**: "refund the payment", not "delete the row".
- A compensating action can itself fail → retry with backoff; escalate to a manual resolution queue when retries are exhausted.
- Record saga state per aggregate so restarts resume where they left off.

Typical step → compensation pairs:

| Step | Compensation |
|---|---|
| Reserve inventory | Release reservation (idempotent by reservation id) |
| Capture payment | Refund (idempotent by payment/refund id) |
| Create shipment | Cancel shipment |
| Send confirmation email | None needed (side effect with no consistency requirement) |

## The outbox pattern

The outbox guarantees the event is published if and only if the transaction committed. Two writes that must be atomic: the business row and the outbox row.

```sql
CREATE TABLE outbox (
  id            UUID PRIMARY KEY,
  aggregate_id  UUID NOT NULL,
  event_type    TEXT NOT NULL,        -- e.g. order.created
  payload       JSONB NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  published_at  TIMESTAMPTZ           -- NULL until the relay publishes it
);

-- Written in the SAME local transaction as the business change:
BEGIN;
  INSERT INTO orders (id, status) VALUES (:id, 'pending');
  INSERT INTO outbox (id, aggregate_id, event_type, payload)
    VALUES (gen_random_uuid(), :id, 'order.created',
            jsonb_build_object('orderId', :id));
COMMIT;
```

Relay options:

| Relay | How it works | Pros | Cons |
|---|---|---|---|
| **Polling publisher** | Background job SELECTs unpublished rows, publishes, marks `published_at` | Simple, portable, no extra infra | Polling latency; must batch and prune old rows |
| **Transactional outbox via CDC** | Debezium / Maxwell tail the WAL (binlog) and emit changes | Low latency, no polling, scales | An extra component (CDC connector) to operate |

Publishing is **at-least-once**: the relay may crash after the broker accepted the message but before marking `published_at`, so consumers see duplicates. Handle with idempotent consumers.

## Idempotent consumers

Consumers must dedupe. The dedupe record must be written in the same transaction as the effect, or a crash between effect and dedupe re-applies the event.

```typescript
async function handleOrderCreated(event: OrderCreated) {
  // Dedupe: processed event id is a unique PK, written in the same tx as the effect
  const inserted = await db.processedEvents.create({ id: event.id });
  if (!inserted) return; // already handled — redelivery
  await db.orders.update({ id: event.orderId, status: 'pending_payment' });
}
```

For API calls (not events), require clients to send an `Idempotency-Key` header; store the key + response and replay the stored response for repeat keys.

## Checklist

- [ ] Every step has a compensation defined, to run in reverse order
- [ ] Compensations are idempotent and retried with backoff
- [ ] Saga state persisted per aggregate (resumable after restart)
- [ ] Timeout handling defined for every async step (no step hangs forever)
- [ ] Events published via outbox (or an equivalent exactly-once-write mechanism)
- [ ] Consumers dedupe by event id in the same transaction as their effect
- [ ] Orchestration chosen over choreography when flow visibility matters
