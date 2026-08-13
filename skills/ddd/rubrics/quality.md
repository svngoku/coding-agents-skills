# Rubric: Checkout Domain Model — score 0.0–1.0

Judge the agent's `domain.py` and `context-map.md` together. The
deterministic checks verify that the required constructs exist; this rubric
scores the *quality* of the DDD decisions and their coherence. Score the four
dimensions below (0.0–0.25 each) and sum, or judge holistically against the
anchors.

## Aggregate boundaries & tactical correctness (0.0–0.25)
- Order is the aggregate root; order lines / items live *inside* the
  aggregate (the consistency boundary), not outside it.
- Order references Customer and Product **by ID only** (`customer_id`,
  `product_id`) — no embedded entity objects, no navigation to another
  aggregate's internals.
- Entities (Order, Customer, Product) are mutable and identified by `id`;
  value objects (Money, Address) are immutable and defined by attributes.
- Repositories are ports (interfaces), one per aggregate root, with no
  infrastructure/SQL leakage into the domain layer.

## Ubiquitous language (0.0–0.25)
- Class and method names use domain terms (`Order`, `place`, `cancel`,
  `capture`, `reserve`) — not DTO/CRUD jargon (`OrderRecord`,
  `process_order`, `update_status_flag`).
- The vocabulary in `domain.py` is consistent with the terms used in
  `context-map.md` (e.g. the Sales context's "Order" is the same concept the
  code models).

## Invariant enforcement (0.0–0.25)
- Invariants are enforced *inside the aggregate methods*, not in the
  application layer or a validator dumped outside the model: methods raise on
  invalid transitions (e.g. placing an empty order, adding a line to a placed
  order, capturing payment before placement).
- Value objects are self-validating where it matters (Money non-negative,
  currency present; quantity > 0) and behave as immutable values
  (arithmetic returns new instances, currency mismatch is rejected).
- The domain is *not anemic*: behavior lives on the model, not in a service
  that pokes at getters/setters.

## Strategic design coherence (0.0–0.25)
- Bounded contexts (Sales, Inventory, Billing, Payment) are named with clear,
  non-overlapping responsibilities, and the context that **owns** the Order
  aggregate is stated explicitly.
- Relationships between contexts are realistic and *directional*:
  upstream/downstream (Customer-Supplier) is stated with the correct
  direction, and integration patterns (ACL, Conformist, OHS/Published
  Language, Shared Kernel, ...) are chosen appropriately for each pair
  (e.g. an ACL toward a legacy/external system, conformist where the
  downstream has no influence).
- The map reflects the code: nothing in the tactical model contradicts the
  strategic boundaries (e.g. Sales owns Order; Payment is a separate context
  reached by events, not a class inside the Order aggregate).

## Penalties
- **−15%** if either file is present but mostly a copy-paste of the skill's
  examples with renamed labels and no reasoning about this domain.
- **−10%** if `domain.py` would not import as written (undefined names,
  bad indentation) even though the deterministic checks passed.
