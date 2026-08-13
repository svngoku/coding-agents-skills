# Python Testing Deep Dive (pytest)

pytest-specific depth for the testing-patterns skill. Read the main
[SKILL.md](../SKILL.md) first for the language-agnostic framework.

## pytest fixtures

### Scope

| Scope | Lifetime | Use for |
|-------|----------|---------|
| `function` (default) | per test | isolated state, mocks |
| `class` | per test class | class-level shared setup |
| `module` | per module | read-only data, persistent mocks |
| `session` | whole run | DB containers, app clients |

### Yield fixtures (setup + teardown)

Code after `yield` runs as teardown even when the test fails:

```python
@pytest.fixture
def session(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    s = Session(engine)
    yield s
    s.close()
    Base.metadata.drop_all(engine)
```

### autouse

```python
@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch):
    monkeypatch.setattr(utils, "now", lambda: datetime(2024, 1, 1))
```

## Parametrization

Each case runs (and fails) independently, with readable IDs:

```python
@pytest.mark.parametrize(
    "amount,percent,expected",
    [(100, 10, 90), (50, 0, 50), (0, 50, 0)],
    ids=["discount", "zero-percent", "zero-amount"],
)
def test_apply_discount(amount, percent, expected):
    assert apply_discount(amount, percent) == expected
```

- `pytest.param(..., marks=pytest.mark.xfail)` — mark a known-broken case
- `indirect=True` — feed a fixture from the parameter list instead of a value

## factory_boy factories

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda u: f"{u.username}@example.com")
    team = factory.SubFactory(TeamFactory)
    created_at = factory.Faker("date_time_this_year")

    @factory.post_generation
    def roles(self, create, extracted, **kwargs):
        if extracted:
            self.roles.add(*extracted)
```

| Helper | Purpose |
|--------|---------|
| `Sequence` | unique values (`f"user{n}"`) |
| `LazyAttribute` | derived from other attributes |
| `SubFactory` | build related objects |
| `Faker` | realistic values |
| `post_generation` | wire up after construction |

Build modes: `UserFactory.build()` (no persistence), `create()`
(persists), `build_batch(10)` / `create_batch(10)`. For Django use
`factory.DjangoModelFactory`; for SQLAlchemy,
`factory.alchemy.SQLAlchemyModelFactory`.

## Property-based testing with Hypothesis

```python
from hypothesis import given, assume, strategies as st

@given(st.lists(st.integers(), min_size=1))
def test_total_is_sum_of_items(xs):
    assume(all(x >= 0 for x in xs))      # filter invalid inputs
    assert sum(xs) == total(xs)
```

- Useful strategies: `st.integers()`, `st.floats()`,
  `st.text()`, `st.dates()`, `st.dictionaries()`,
  `st.from_type(User)`
- Custom data with `@st.composite` (decorate a function that takes
  `draw` and returns a value)
- `@example(...)` pins known regression inputs on top of generated ones
- On failure, Hypothesis shrinks to the minimal counterexample and prints a
  reproducible snippet — add it as an `@example`

## Coverage

```bash
pytest --cov=src --cov-branch --cov-report=term-missing --cov-report=html
```

- `--cov-branch` measures both sides of every `if`, not just lines
- `--cov-report=html` gives a browsable report
- Gate in CI: `--cov-fail-under=80`, or `fail_under` under
  `[tool.coverage.report]` in pyproject.toml
- Coverage `!` correctness: look for branches never exercised, not a number

## Mutation testing with mutmut

```bash
pip install mutmut
mutmut run --paths-to-mutate src/    # apply mutations to your code
mutmut results                       # surviving mutants = weak tests
mutmut show 3                        # inspect mutant #3 (the diff)
```

A surviving mutant means no test noticed the behavior change — write a test
that pins that behavior. Run mutmut on changed modules in CI or nightly; it is
too slow for every push on a large codebase.

## Parallel execution & retries

```bash
pip install pytest-xdist
pytest -n auto                 # one worker per CPU
pytest -n 4 --dist loadgroup  # keep tagged tests on the same worker
```

Use `@pytest.mark.group("db")` with `--dist loadgroup` so DB tests
do not run concurrently against one database. For a temporary stopgap while
triaging flaky tests: `pytest --reruns 1 --reruns-delay 1`
(pytest-rerunfailures) — never the permanent fix; see
[flaky-tests-ci.md](flaky-tests-ci.md).
