# Rubric: E-Commerce Schema Design — score 0.0–1.0

Judge the agent's `schema.sql` and `DESIGN.md` together. The deterministic
checks verify that the required constructs exist; this rubric scores the
quality of the design decisions and their explanation. Score the four
dimensions below (0.0–0.25 each) and sum, or judge holistically against the
anchors.

## Relationship correctness (0.0–0.25)
- 1:N relationships modeled with the foreign key on the "many" side:
  customers → orders, orders → order_items, products → order_items.
- order_items acts as the association table between orders and products
  (composite key or unique pair); no misplaced, missing, or dangling FKs.
- Referential actions match the lifecycle: RESTRICT / NO ACTION for protected
  entities (customers, products); ON DELETE CASCADE for order_items.

## Index rationale (0.0–0.25)
- Composite index on orders(customer_id, created_at) with the equality column
  first, explained in terms of the "recent orders for a customer" query.
- Index on order_items(order_id) justified by order-detail reads.
- Indexes map to named queries; no index-everything bloat; rationale is
  concrete, not generic.

## Constraint discipline (0.0–0.25)
- Invariants live in the database, not just the ORM: NOT NULL + UNIQUE on
  business keys (email), CHECK (quantity > 0), CHECK (unit_price_cents >= 0).
- Money stored as integer cents (no floats).
- Surrogate BIGINT primary keys with UNIQUE natural keys where appropriate.

## Clarity of design notes (0.0–0.25)
- DESIGN.md explains normalization to 3NF and what was decomposed and why.
- The surrogate vs natural key tradeoff is stated, not just asserted.
- Index and constraint choices are tied back to the requirements; the note is
  concise, well-structured, and readable.
