# Task: Design a PostgreSQL Schema for a Small E-Commerce Domain

You are designing the database for a small e-commerce application. Produce
**two files in the current workspace**:

1. `schema.sql` — the PostgreSQL DDL.
2. `DESIGN.md` — a short design note explaining your decisions.

The grader only inspects these two files; there is no database and no network.
Make the DDL syntactically valid PostgreSQL.

## Domain

- **customers** — people who place orders. Email must be present and unique.
- **products** — items sold in the store.
- **orders** — one order belongs to exactly one customer.
- **order_items** — line items on an order; each line references one order and
  one product. Deleting an order must remove its line items.

## Requirements for `schema.sql`

Write PostgreSQL `CREATE TABLE` statements, followed by explicit
`CREATE INDEX` statements. Requirements:

1. **One table per entity**: `customers`, `products`, `orders`,
   `order_items` (at least 4 tables).
2. **Surrogate primary keys**: every table gets a `BIGINT` primary key,
   e.g. `id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY`.
3. **Foreign keys with explicit referential actions** — state the action on
   every foreign key:
   - `order_items.order_id` → `orders(id)` with `ON DELETE CASCADE`
     (line items live and die with their order).
   - `order_items.product_id` → `products(id)` and
     `orders.customer_id` → `customers(id)` with `ON DELETE RESTRICT`
     (or `NO ACTION`).
4. **CHECK constraints**: `quantity > 0` on `order_items.quantity`, and
   `unit_price_cents >= 0` (or `> 0`) on `order_items.unit_price_cents`.
   Store money as integer cents, never floats.
5. **Business keys**: `customers.email` is `NOT NULL` and `UNIQUE`.
6. **Indexes** (explicit `CREATE INDEX` statements):
   - Composite index on `orders (customer_id, created_at)` — serves "recent
     orders for a customer" (equality column first, then the range/sort column).
   - Index on `order_items (order_id)` — serves order-detail reads.

## Requirements for `DESIGN.md`

A concise design note (aim for roughly 30–60 lines) covering:

1. **Normalization**: explain how the schema is normalized to 3NF — which facts
   live where and why (e.g. why `order_items` stores only `product_id` and
   not the product name).
2. **Key choices**: why surrogate `BIGINT` identity keys instead of natural
   keys, and why `email` is kept as a `UNIQUE` business key rather than the
   primary key.
3. **Index rationale**: which queries each index serves, and why the composite
   index puts `customer_id` first.

If anything is ambiguous, prefer the conventions the requirements above state
explicitly.
