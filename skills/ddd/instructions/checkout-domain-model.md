# Task: Model the Checkout Domain with DDD

You are designing the domain model for an **e-commerce checkout** flow using
Domain-Driven Design. Produce **two files in the current workspace**:

1. `domain.py` — the tactical domain model (Python 3, standard library only).
2. `context-map.md` — the strategic context map.

The grader only inspects these two files; there is no database, no web
framework, and no network. Python 3.9+ syntax is fine.

## 1. `domain.py` — tactical model

Model the checkout flow around an **Order aggregate**. Use only the standard
library (`dataclasses`, `typing`, `abc`, `enum`, `uuid`,
`datetime`). Include all of the following:

- **Entities** (identity matters, mutable): `Order`, `Customer`,
  `Product` — each with an `id` field (e.g. UUID or string). Equality is by
  identity, not by attributes.
- **Value objects** (immutable, defined by attributes, no identity):
  `Money` (carries both `amount` and `currency` — use
  `@dataclass(frozen=True)` or a frozen model) and `Address` (street, city,
  postal code, country).
- **Order aggregate root** that enforces its invariants:
  - references its customer by `customer_id` only — never embed a
    `Customer` object inside `Order`;
  - holds order lines (an `OrderLine` entity or an `items` collection of
    product_id + quantity + unit price);
  - exposes behavior methods (e.g. `add_line`, `place`, `cancel`) that
    **raise** on invalid transitions — e.g. cannot place an empty order,
    cannot modify a placed order. An explicit `validate` / `can_*` method
    is also acceptable.
- **Domain events** (past tense, immutable): `OrderPlaced` and
  `PaymentCaptured` (add `OrderCancelled` etc. as you see fit). The
  aggregate should record them (e.g. a `collect_events()` method).
- **Repository interface** (a port, not an implementation): e.g.
  `OrderRepository` declared with `Protocol` or `ABC` +
  `abstractmethod`, exposing `find_by_id`, `save`,
  `find_by_customer`. No SQL, no concrete implementation.

## 2. `context-map.md` — strategic model

Document the bounded contexts of the checkout system:

- **Contexts**: name at least these four — `Sales` (owns the Order
  aggregate), `Inventory` (stock / reservation), `Billing` (invoices,
  payment capture records), `Payment` (payment-provider integration).
- **Relationships**: for the integrations between contexts, name the pattern
  and the direction — e.g. upstream/downstream (Customer-Supplier),
  **Conformist**, **Anti-Corruption Layer (ACL)**, Open Host Service,
  Published Language. At least one relationship type must be stated
  explicitly.
- **Ownership**: state explicitly which context owns the `Order` aggregate.

You may use a Mermaid diagram, a table, or prose — any format is fine as long
as the contexts, the relationship types/directions, and the ownership of
`Order` are all named.

## Guidance

- Use **ubiquitous language**: domain terms (`place`, `capture`,
  `reserve`) in methods and classes — not DTO/CRUD jargon.
- `domain.py` must be syntactically valid Python that imports cleanly;
  `context-map.md` must be readable Markdown.
- Follow the skill's conventions: aggregates reference other aggregates by
  ID only, invariants live inside the aggregate, and the domain layer has no
  infrastructure dependencies.

Save both files as `domain.py` and `context-map.md`, then double-check
that both exist in the current directory before finishing.
