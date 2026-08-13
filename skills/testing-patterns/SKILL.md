---
name: testing-patterns
description: >
  Plan, write, and maintain automated tests in any language, with worked
  examples in Python and JavaScript/TypeScript. Use this skill whenever the
  user wants to write or plan tests, apply TDD (red-green-refactor), structure
  unit tests with arrange-act-assert and given-when-then naming, choose between
  fakes, stubs, and mocks, set up dependency injection for testability, build
  integration tests with testcontainers, database seeding, or transaction
  rollback, write Playwright or Cypress end-to-end tests, use property-based
  testing with Hypothesis or fast-check, create pytest fixtures or factory_boy
  factories, review coverage or mutation testing, or triage flaky tests in CI.
  Also trigger for "test pyramid", "test trophy", "test isolation", "mocking",
  "snapshot tests", "golden files", "parametrized tests", "make the tests pass".
---

# Testing Patterns

Automated tests are how a codebase stays safe to change. This language-agnostic skill tells you what to test, where each test belongs, and how to keep the suite fast and reliable — with worked examples in Python (pytest, Hypothesis, factory_boy, testcontainers) and JavaScript/TypeScript (Vitest, Jest, fast-check, Playwright).

## Quick Reference

| Task | Reference |
|------|-----------|
| pytest fixtures & parametrization, factory_boy, Hypothesis, coverage.py, mutmut | [python-testing.md](references/python-testing.md) |
| Vitest/Jest mocking, fast-check, Playwright config & selectors, testcontainers-node | [js-ts-testing.md](references/js-ts-testing.md) |
| CI parallelization, flaky-test triage, rerun policies, quarantine | [flaky-tests-ci.md](references/flaky-tests-ci.md) |

## Test Pyramid

Many fast, isolated unit tests at the bottom; fewer slower integration tests in the middle; a handful of slow, brittle end-to-end tests at the top.

| Layer | Scope | Speed | Typical count | Catches |
|-------|-------|-------|---------------|---------|
| Unit | one function/class | ms | hundreds+ | logic errors, edge cases |
| Integration | modules + real deps (DB, cache, APIs) | s | dozens | schema/contract mismatches, wiring |
| E2E | whole system via UI/API | min | tens | broken journeys, config/deploy |

Map each feature to the lowest layer that can catch its failure:

```text
Feature                Layer        Tool
price calculation      unit         pytest / vitest
order persistence      integration  pytest + testcontainers Postgres
guest checkout         e2e          Playwright
```

### When the pyramid is wrong

- **Testing trophy** (Kent C. Dodds): front-end apps should be integration-heavy — render components with real hooks/context, assert on user-visible behavior. Reserve e2e for critical journeys.
- **Contract tests**: at service-to-service boundaries, test the API contract once per side (consumer-driven contracts) instead of full-stack e2e flows.

## TDD: Red-Green-Refactor

1. **Red** — write one failing test for the next behavior.
2. **Green** — minimal change to pass it; no extra features.
3. **Refactor** — clean up; tests stay green.

```python
# test_pricing.py
def test_discount_applies_at_100_or_more():
    assert apply_discount(100, 10) == 90   # RED: apply_discount does not exist yet

# pricing.py
def apply_discount(amount, percent):       # GREEN: minimal implementation
    return amount - amount * percent / 100
```

### When TDD is not the right call

- Exploratory code, spikes, and prototypes
- One-off scripts and throwaway analysis
- Mostly I/O glue with trivial logic — an integration test is cheaper
- UI/layout work where behavior is fuzzy — verify visually or with e2e smoke tests
- Legacy code with no tests — write *characterization tests* first, then refactor

## Unit Test Design

### Arrange-Act-Assert

Build state, perform the action, check the result:

```python
def test_cart_total_hits_free_shipping_threshold():
    # Arrange
    cart = Cart()
    cart.add(Item("book", 25), qty=4)
    # Act
    total = cart.total()
    # Assert
    assert total == 100
```

### Test isolation

Tests must not depend on each other or on order: fresh state per test, no shared mutable globals, no assumptions about prior tests.

### Naming: given-when-then

```python
def test_given_empty_cart_when_checkout_then_raises_value_error():
```

```ts
describe("checkout", () => {
  it("rejects an empty cart", () => {
    expect(() => checkout(new Cart())).toThrow("cart is empty");
  });
});
```

### Fakes, stubs, and mocks

| Kind | What it is | Use it when |
|------|-----------|-------------|
| Dummy | passed in, never used | filling parameters |
| Stub | returns canned answers | making a code path reachable |
| Fake | working simplified implementation (in-memory repo) | replacing a slow/external dependency with real behavior |
| Mock | records and asserts on calls | verifying an interaction happened |

Prefer real objects and fakes; mock only at seams where the dependency is expensive or external and interaction is the point. Mocking is an anti-pattern when it tests the mock: asserting internal calls, mocking classes you own, or deep mock chains that mirror the implementation.

### Dependency injection for testability

Inject dependencies instead of constructing them inside the unit:

```python
class OrderService:
    def __init__(self, repo: OrderRepository, notifier: Notifier):
        self._repo = repo
        self._notifier = notifier
```

```ts
export class OrderService {
  constructor(
    private repo: OrderRepository,
    private notifier: Notifier,
  ) {}
}
```

Tests inject a `FakeOrderRepository` (in-memory dict) and a no-op notifier — no mocks required.

## Integration Tests

Integration tests exercise real collaborations — database, cache, filesystem, HTTP boundary — and catch wiring/contract bugs unit tests cannot see. The cost is speed and setup.

- **Testcontainers** run real services in disposable Docker containers per suite, so CI matches local:

```python
# conftest.py
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def db_url():
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url()
```

- **Database seeding**: create the minimum data each test needs (factories or inline setup), not a giant shared dump.
- **Transaction rollback**: wrap each test in a transaction and roll back, or recreate schema per test:

```python
@pytest.fixture
def session(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    s = Session(engine)
    yield s
    s.close()
    Base.metadata.drop_all(engine)   # clean slate per test
```

## E2E Tests

Playwright and Cypress drive a real browser against the running app.

**Cover:** the happy path of the top 3–5 business journeys (checkout, signup, login, search), auth and permission flows, anything with no cheaper test.

**Avoid:** testing every UI detail, long journeys that duplicate integration coverage, `sleep()` waits — wait for elements, routes, or network idle instead.

```ts
// Playwright
import { test, expect } from "@playwright/test";

test("guest can complete checkout", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Add to cart" }).first().click();
  await page.getByRole("button", { name: "Checkout" }).click();
  await expect(page).toHaveURL(/\/checkout\/confirm/);
});
```

## Property-Based & Snapshot Tests

Property-based testing generates many inputs and asserts *invariants* that must hold for all of them — it finds edge cases you would never hand-write:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_is_idempotent(xs):
    assert sorted(sorted(xs)) == sorted(xs)
```

```ts
import fc from "fast-check";

test("sort is idempotent", () => {
  fc.assert(fc.property(fc.array(fc.integer()), (xs) => {
    expect([...xs].sort().sort()).toEqual([...xs].sort());
  }));
});
```

**Golden/snapshot files** record output to a committed file and diff future runs — good for serializers, CLI output, and rendering. Pitfalls: keep snapshots small and reviewable; normalize timestamps, IDs, and randomness before snapshotting; never auto-accept new snapshots in CI; prefer targeted assertions for stable behavior.

## Fixtures, Factories & Parametrization

pytest fixtures manage setup/teardown, scoped to function/class/module/session and reusable across tests (e.g. `@pytest.fixture def user(db): return UserFactory.create()`). Factories (factory_boy) build valid objects without hand-written setup:

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda u: f"{u.username}@example.com")
```

Parametrized tests run one body over many cases — each case fails independently:

```python
@pytest.mark.parametrize("amount,percent,expected", [
    (100, 10, 90),
    (50, 0, 50),
    (0, 50, 0),
])
def test_apply_discount(amount, percent, expected):
    assert apply_discount(amount, percent) == expected
```

```ts
test.each([
  [100, 10, 90],
  [50, 0, 50],
])("applyDiscount(%i, %i) === %i", (amount, percent, expected) => {
  expect(applyDiscount(amount, percent)).toBe(expected);
});
```

## Coverage & Mutation Testing

Coverage reports which *lines ran*, not whether they were *asserted* — a test that runs code and asserts nothing still counts as covered. Use it as a map of dead zones, not a score to maximize:

```text
Name              Stmts   Miss Branch BrPart  Cover
src/pricing.py       12      0      4      1    94%
src/cart.py          30      9      6      4    74%   # untested branch here
```

**Mutation testing** (mutmut, Stryker) injects small bugs — flipping `<` to `>=`, deleting a return — and checks whether tests catch them; surviving mutations mark weak tests. Slow, so run on changed code in CI or nightly. See [python-testing.md](references/python-testing.md) and [js-ts-testing.md](references/js-ts-testing.md).

## CI Integration

- **Fast feedback**: unit tests on every push; integration once services are up; e2e on merge or nightly.
- **Parallelize**: shard across workers (`pytest-xdist`, Vitest workers, Playwright shards).
- **Fail fast**: stop at the cheapest layer that catches the bug.

```bash
pytest -n auto                    # unit: parallel across CPUs, every push
npx playwright test --shard=1/4   # e2e: sharded across 4 workers on merge
```

**Flaky tests** are a bug, not a mystery: reproduce locally, check order/timing/environment dependence, fix the cause. To buy time, use a bounded retry (once) and move the test to a **quarantine** suite that reports separately. Full playbook: [flaky-tests-ci.md](references/flaky-tests-ci.md).

## Anti-Patterns to Avoid

- **Testing implementation details** — private methods, internal calls; breaks on any refactor
- **Mocking everything** — the test mirrors the implementation and passes for the wrong reasons
- **Asserting nothing** — running code without checks is a smoke test, not a test
- **Shared mutable state / order dependence** — the suite breaks when one test changes
- **Sleep-based waits** — always wait for a condition (element, route, response)
- **Blindly auto-updating snapshots** — hides real regressions
- **Chasing 100% coverage** — incentivizes tests that never fail
- **Defaulting to retry-on-failure** — masks flaky tests instead of fixing them
- **Testing the framework** — asserting on library internals or third-party behavior

## When to Use / Not Use

**Use this skill when** writing tests for new or existing code, choosing a testing strategy or tools, applying TDD, designing test doubles, setting up integration/e2e infrastructure, using property-based or snapshot testing, reviewing coverage, or triaging flaky tests in CI.

**Do not use when** the user just wants to *run* tests, is doing throwaway exploration with no behavior to assert, or asks for linting/type-checking/static analysis.
