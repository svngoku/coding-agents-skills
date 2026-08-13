# Rubric: pytest suite for the buggy orders module - score 0.0-1.0

Judge the agent's `test_orders.py` on the four dimensions below
(0.0-0.25 each) and sum, or judge holistically against the anchors.

## Bug coverage & correctness (0.0-0.25)
- All three bugs are pinned to their intended behavior: a negative quantity
  raises `ValueError`, an empty-cart average raises `ValueError`, and
  `apply_discount` is asserted to the exactly-once value
  (e.g. `apply_discount(100, 10) == 90`).
- Assertions FAIL against the buggy implementation - they are not tautological
  and not written to match the bug's output.
- `pytest.raises` used for exception paths; exact expected values for value
  paths (no loose `is not None` checks, no `assert True`).

## Test design quality (0.0-0.25)
- Clear arrange-act-assert structure in each test.
- Isolation: fresh state per test via fixtures; no shared mutable state, no
  order dependence, no test relying on another test's side effects.
- No sleep-based waits (`time.sleep`), no snapshot/golden-file tests.

## Parametrization & fixtures (0.0-0.25)
- `@pytest.mark.parametrize` used where the same shape repeats (discount
  percentages, quantities, zero/negative cases), with readable IDs where useful.
- A pytest fixture provides shared setup and is actually consumed by tests;
  setup is not copy-pasted across tests.
- Fixture scope is appropriate (a function-scoped fresh cart).

## Naming & readability (0.0-0.25)
- Test names follow given-when-then and read as specifications
  (`test_given_..._when_..._then_...`).
- Names distinguish the buggy behaviors they pin.
- Code is readable and consistent, without dead code or debug prints.

## Penalties
- -20% if the suite would not run as-is: Python syntax errors, a missing
  import, or asserts that always pass regardless of the module.
