# Task: Write a pytest suite for a buggy orders module

The current workspace contains `orders.py` - a small shopping-cart module
with **three intentional bugs**. Read it first.

| # | Bug (current behavior) | Intended behavior |
|---|------------------------|-------------------|
| 1 | `Cart.add_item("book", 1000, -1)` silently stores the item | raise `ValueError` |
| 2 | `apply_discount(100, 10)` returns `80` (discount applied twice) | return `90` |
| 3 | `Cart().average_unit_price()` raises a raw `ZeroDivisionError` | raise `ValueError` |

Your job is **not** to fix `orders.py`. Write a pytest test suite that
**pins the intended behavior**, so every bug above is caught by a failing
assertion against the current (buggy) implementation.

## Deliverable

Create exactly one file: **`test_orders.py`** in the current directory.

The suite must:

1. Import the module (`import orders` or `from orders import ...`).
2. Contain **at least 3 test functions**, named `test_*` with
   **given-when-then** style names (e.g.
   `test_given_empty_cart_when_average_unit_price_then_raises_value_error`).
3. Structure every test with **arrange-act-assert**: build the state, perform
   the action, assert the outcome.
4. Cover **all three bugs** above with explicit assertions that fail against
   the buggy implementation.
5. Use `@pytest.mark.parametrize` for at least one edge-case group (e.g.
   discount percentages, quantities, zero values).
6. Use a **pytest fixture** for shared setup (e.g. a fresh `Cart` per test) -
   no copy-pasted setup, no shared mutable state between tests.
7. Give every test a real assertion - an `assert` on a computed value or
   `pytest.raises(...)` for exceptions. No smoke tests, no `assert True`,
   no snapshot/golden-file comparisons.
8. Contain **no `time.sleep`** or any sleep-based waiting - these tests must
   run instantly and be deterministic.

The file must be syntactically valid Python and runnable with
`pytest test_orders.py` against the provided `orders.py`.
