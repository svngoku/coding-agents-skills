# Rubric: Choreographed Checkout Saga — score 0.0–1.0

Judge `checkout-design.md` and `events.py` together. The deterministic
checks verify that the required constructs exist; this rubric scores the
quality of the design. Score the four dimensions below (0.0–0.25 each) and
sum, or judge holistically against the anchors.

## Choreography coherence (0.0–0.25)
- The flow is genuinely event-driven: services react to events; there is no
  central orchestrator, no coordinator service, and no synchronous
  request/reply chain across services.
- The event sequence is a coherent saga: each success event advances the flow
  (OrderPlaced → PaymentSucceeded + InventoryReserved → OrderConfirmed →
  NotificationSent).
- The Mermaid diagram matches the prose: participants are the services, arrows
  are events, and both the happy path and the failure branch are drawn.

## Failure and compensation paths (0.0–0.25)
- Every participant reacts to success **and** failure events; a failed step
  triggers compensation (e.g. InventoryFailed → refund payment).
- Compensations are business-level undos (refund, release reservation) run in
  reverse order of the steps that succeeded — not database rollbacks.
- Compensation is idempotent and handles partial failure; timeouts on async
  steps are considered rather than letting a step hang forever.
- `events.py` models the failure events (PaymentFailed, InventoryFailed,
  CompensationTriggered), not only the happy path.

## Outbox correctness (0.0–0.25)
- The outbox is described as a write in the **same local transaction** as the
  business change (state + event are atomic), not as a separate
  after-the-fact publish.
- A relay (polling publisher or CDC such as Debezium) publishes outbox rows;
  delivery is at-least-once.
- The consequence is drawn: at-least-once → consumers must deduplicate.

## Service-boundary discipline (0.0–0.25)
- Each service owns its data exclusively; no shared database, no cross-service
  table access.
- Each event is published by exactly one service and named in the business
  language of the emitting context (OrderPlaced, not orders_inserted).
- Boundaries are coherent: responsibilities and event ownership are stated per
  service and consistent between the prose and the diagram.
