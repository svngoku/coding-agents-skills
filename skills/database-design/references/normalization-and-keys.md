# Normalization and Keys: Worked Examples

Deep dive on turning an unnormalized spreadsheet into 3NF, choosing natural vs surrogate keys, and deciding between UUID and bigint.

## Table of contents

1. [Normalization from scratch](#normalization-from-scratch)
2. [1NF: atomic values](#1nf-atomic-values)
3. [2NF: no partial dependencies](#2nf-no-partial-dependencies)
4. [3NF: no transitive dependencies](#3nf-no-transitive-dependencies)
5. [BCNF and beyond](#bcnf-and-beyond)
6. [Denormalizing deliberately](#denormalizing-deliberately)
7. [Natural vs surrogate keys](#natural-vs-surrogate-keys)
8. [Composite keys](#composite-keys)
9. [UUID vs bigint](#uuid-vs-bigint)
10. [Foreign key referential actions](#foreign-key-referential-actions)

## Normalization from scratch

The classic source of redundancy is a spreadsheet-style table holding order, customer, and product data in one row per line item:

| order_no | customer | customer_city | product | category | qty | unit_price | order_date |
|---|---|---|---|---|---|---|---|
| 1001 | Ada | Paris | Keyboard | Hardware | 1 | 120.00 | 2025-01-05 |
| 1001 | Ada | Paris | Mouse | Hardware | 2 | 25.00 | 2025-01-05 |
| 1002 | Lin | Berlin | Monitor | Hardware | 1 | 300.00 | 2025-02-10 |

Problems: customer and product data repeat per line item; a move or a price change must touch many rows; and the price at order time is conflated with the product's current price.

### 1NF: atomic values

Rule: every column holds a single value; no repeating groups or lists. A `phones` column holding "555-0100,555-0199" violates 1NF — split it into a child table, one row per value. Our table already satisfies 1NF.

### 2NF: no partial dependencies

Rule: every non-key column must depend on the *whole* primary key. Only applies when the key is composite.

The natural composite key here is `(order_no, product)`. `customer` depends only on `order_no` (a partial dependency), so it belongs in an `orders` table keyed by `order_no`. `category` depends only on `product`, so it belongs in a `products` table:

```sql
orders(order_no PK, customer, customer_city, order_date)
products(product PK, category)
order_items(order_no FK, product FK, qty, unit_price)
```

### 3NF: no transitive dependencies

Rule: no non-key column depends on another non-key column. In `orders`, `customer_city` depends on `customer`, not on `order_no` — move it to `customers`:

```sql
customers(customer_id PK, name, city)
orders(order_id PK, customer_id FK, order_date)
products(product_id PK, name, category_id FK)
categories(category_id PK, name)
order_items(order_id FK, product_id FK, qty, unit_price_cents)
```

Note that `unit_price_cents` stays on `order_items`: it is the *price at order time*, a fact of the order, not of the product. That is historical accuracy, not a transitive dependency. Current price lives in `products` and is joined when needed.

### BCNF and beyond

Boyce-Codd normal form tightens 3NF for edge cases where a non-key column is itself a candidate key of part of a table (classic example: `professor, course, department` where department is determined by professor). 3NF is the practical target; BCNF fixes rare anomalies. 4NF/5NF handle multi-valued and join dependencies — almost never worth the extra joins in hand-designed schemas.

## Denormalizing deliberately

Denormalization copies or aggregates data to serve a specific hot read. Justify it with a measured query:

| Benefit | Cost |
|---|---|
| One-table reads, no joins | Write amplification: every write maintains the copies |
| Precomputed aggregates (order totals, counters) | Drift risk if updated outside the owning transaction |
| Simpler reporting shapes | Backfill work when the derivation logic changes |
| Lower read latency at scale | Harder to reason about where truth lives |

Rules of thumb: keep the copy *derived* (computed from the source of truth); update it in the same transaction as its source (recompute `orders.total_cents` whenever an item changes); prefer materialized views or read models over hand-maintained copies where the DB supports them.

## Natural vs surrogate keys

| | Natural key | Surrogate key |
|---|---|---|
| Examples | `isbn`, `email`, `tax_id`, `slug` | `id BIGINT GENERATED ALWAYS AS IDENTITY` |
| Pros | Meaningful, stable external identifiers; no extra column for business lookups | Never changes, compact, no domain coupling, DB-generated |
| Cons | Can change (`email`), be re-issued (`tax_id`), or be long/variable | Meaningless (UNIQUE business keys still needed); joins are opaque |
| Risks as PK | Every referencing row must be updated when the key changes | — |

Decision procedure: ask whether the natural value is truly immutable, guaranteed unique, and compact. If all three hold (e.g. `isbn`), it is a fine PK. Otherwise use a surrogate PK and add the natural value as a UNIQUE business key. Never use a mutable value (`email`, `username`) as a PK.

## Composite keys

Use when the natural identity is a pair or triple: `order_items(order_id, product_id)`, `reservation(room_id, start_time)`.

| Pros | Cons |
|---|---|
| No extra column; uniqueness enforced naturally | Awkward as a foreign-key target (must repeat all columns) |
| Natural for junction tables | Wider indexes; column order matters |
| — | Painful to rename or extend later |

Recommendation: composite PKs are fine for pure junction tables. Elsewhere use a surrogate PK plus a UNIQUE constraint on the composite, so foreign keys stay single-column.

## UUID vs bigint

| | bigint IDENTITY | UUIDv4 | UUIDv7 |
|---|---|---|---|
| Storage | 8 bytes | 16 bytes | 16 bytes |
| Ordering | sequential | random | time-ordered |
| Index locality | excellent | poor (page splits, bloat) | good |
| Generation | server sequence (single point of coordination) | anywhere (client) | anywhere (client) |
| Guessable/enumerable | yes (sequential) | no | partially (timestamp embedded) |
| Distributed writes / merging | hard (collision risk, coordination) | easy | easy |
| Best for | high-write OLTP, single-writer | low-write, client-generated, privacy-sensitive | distributed systems that still want ordered IDs |

Guidance:

- **Default to bigint** for most OLTP schemas: smaller, sequential, cache-friendly, and index-fragment-free.
- **UUIDv7** when IDs must be generated client-side or across many writers and you still want reasonable index locality.
- **UUIDv4** is acceptable at low write rates or when IDs must be unguessable; accept index bloat on hot tables.
- Store UUIDs in a native `uuid` type (PostgreSQL) or convert to an ordered binary form (MySQL `BINARY(16)`) — storing them as `CHAR(36)` wastes space and sorts wrongly.

## Foreign key referential actions

```sql
ALTER TABLE order_items
  ADD CONSTRAINT fk_items_order FOREIGN KEY (order_id)
  REFERENCES orders(id) ON DELETE CASCADE;

ALTER TABLE orders
  ADD CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id)
  REFERENCES customers(id) ON DELETE RESTRICT;
```

| Action | Behavior | Use when |
|---|---|---|
| `RESTRICT` / `NO ACTION` | Refuse to delete a referenced row | Default; safest |
| `CASCADE` | Delete children with the parent | Owner-child lifecycles (order → items); audit the blast radius first |
| `SET NULL` | Null out the child FK | Optional references (deleted user keeps orders) |

Gotchas:

- `CASCADE` deletes can silently cascade through many levels — a user delete wiping orders, items, and refunds. Map the graph before choosing it.
- **PostgreSQL does not automatically index FK columns.** Without an index on the child FK, every delete/update of a parent row triggers a sequential scan of the child table. MySQL auto-creates an index on FK columns; PostgreSQL does not — create it yourself.
- FK cycles (A references B and B references A) make deletes ambiguous; break the cycle or use nullable FKs + `SET NULL`.
