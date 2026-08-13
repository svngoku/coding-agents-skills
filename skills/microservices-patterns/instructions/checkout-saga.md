# Task: Choreographed Checkout Saga Across Microservices

Design a checkout flow as a **choreographed saga** across microservices, and
produce **two files in the current workspace**:

1. `checkout-design.md` — the design document.
2. `events.py` — the domain event definitions.

The grader only inspects these two files; there is no network and no running
services. Make `events.py` valid, importable Python (standard library only)
and `checkout-design.md` self-contained Markdown.

## Domain

An e-commerce checkout spans four services, each owning its own data:

- **Orders** — owns the order aggregate and the saga's success/failure state.
- **Payments** — captures (and refunds) payment for an order.
- **Inventory** — reserves stock for the order's line items.
- **Notifications** — sends the confirmation email and cancellation notices.

No shared database, no synchronous call chain, no two-phase commit. The flow
must be event-driven: services publish events to a broker and react to the
events they subscribe to. No central orchestrator — the saga is **choreographed**.

## Requirements for `checkout-design.md`

A design document covering (roughly 60–120 lines):

1. **Service boundaries** — for each of the four services: its responsibility,
   the data it owns exclusively, and the events it publishes/subscribes to.
   State explicitly that services communicate only through events and never
   share a database.
2. **Event choreography** — a Mermaid diagram (`sequenceDiagram` or
   `flowchart`) showing the event flow: checkout request → order created →
   payment captured / inventory reserved → order confirmed → notification
   sent. Include the failure branch: inventory short → `InventoryFailed` →
   payment refunded.
3. **Failure handling** — what happens when a step fails: compensating actions
   (e.g. refund payment, release reservation) run in reverse order of the
   steps that succeeded; every participant reacts to both success and failure
   events.
4. **The outbox pattern** — explain why events must be published reliably: the
   business row and an `outbox` row are written in the **same local
   transaction**, and a relay (polling publisher or CDC) publishes them to the
   broker, which delivers at-least-once.
5. **Idempotent consumers** — at-least-once delivery means consumers
   deduplicate (by event id, recorded in the same transaction as the effect)
   so retries and redeliveries are safe.

## Requirements for `events.py`

Define the domain events as Python `dataclasses`. Each event carries at least
an `order_id` (the saga aggregate) and an `event_id` (for idempotent
consumers). Include at least:

- `OrderPlaced` — published by Orders when the order is created.
- `PaymentSucceeded` and `PaymentFailed` — published by Payments.
- `InventoryReserved` and `InventoryFailed` — published by Inventory.
- `CompensationTriggered` — published by Orders when it starts compensating.

At minimum, provide `OrderPlaced` **plus at least two more** of the
`*Succeeded` / `*Reserved` / `*Failed` events (≥ 3 event classes total).
Use only the Python standard library. Save as `events.py` in the current
directory.
