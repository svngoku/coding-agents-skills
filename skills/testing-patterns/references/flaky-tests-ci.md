# Flaky Tests & CI Integration Deep Dive

Depth for the testing-patterns skill's CI and flakiness guidance. Read the
main [SKILL.md](../SKILL.md) first for the language-agnostic framework.

## What makes tests flaky

| Cause | Symptom | Fix |
|-------|---------|-----|
| Order dependence | fails only after/before another test | isolate state per test |
| Timing | passes locally, fails on slow CI | wait for conditions, not sleeps |
| Environment | fails under a different locale/TZ/region | pin locale/TZ in setup |
| Race conditions | nondeterministic DB/network timing | synchronize or use real retries |
| Shared resources | parallel workers hit one database | per-worker schemas, load-group tags |
| Global coupling | test reads config another test mutated | reset globals in setup |

## Triage workflow

1. **Reproduce alone** — `pytest tests/x.py::test_y` or
   `npx playwright test --grep "..."`. Alone-and-green ⇒ order or
   environment dependence.
2. **Loop it locally** — rerun 20–50× to estimate the failure rate.
3. **Capture evidence** — enable Playwright trace on retry; run pytest with
   `-l --tb=long -rA`; compare CI logs across runs.
4. **Fix the cause** — reset state, remove timing assumptions, mock the
   nondeterministic dependency, or isolate data.
5. **Buy time only if needed** — bounded retry + quarantine (below).

## Retry policies — and their limits

| Tool | Config |
|------|--------|
| pytest | `pytest --reruns 1 --reruns-delay 1` (pytest-rerunfailures) |
| Playwright | `retries: 1` in config, or `process.env.CI ? 1 : 0` |
| Cypress | `retries: { runMode: 1, openMode: 0 }` |
| GitHub Actions | `continue-on-error: true` + issue-based triage, not blanket retry |

**A retried-then-passed run is a signal, not a pass.** Track it separately
and treat it as flaky until proven otherwise.

## Quarantine

When a test is flaky but you need a green main suite:

1. Move it to a `@pytest.mark.quarantine` marker, a `quarantine/`
   directory, or a dedicated Playwright project.
2. Quarantined tests run in a separate non-blocking job that reports pass
   rates to the team.
3. Fix or delete within a sprint — a permanently quarantined test is debt
   with a deadline.
4. Never let quarantined tests block merges, and never silently lose the
   coverage they provided.

## CI parallelization

| Layer | Technique | Config |
|-------|-----------|--------|
| Unit (Python) | pytest-xdist workers | `pytest -n auto` |
| Unit (JS/TS) | Vitest workers | `pool: "threads"`, `maxWorkers` |
| E2E (Playwright) | shards | `--shard=1/4` … `--shard=4/4` |
| E2E (Cypress) | parallel CI machines | Cypress Cloud / per-machine specs |
| Infra | per-suite service containers | testcontainers |

### GitHub Actions example

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        shard: [1, 2, 3, 4]
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npx playwright install --with-deps chromium
      - run: npx playwright test --shard=${{ matrix.shard }}/4
```

Use `fail-fast: false` so one flaky shard does not cancel the others,
and capture per-shard reports/artifacts for triage.

## Coverage gates in CI

- Enforce a floor: `--cov-fail-under=80` (pytest) or
  `coverageThreshold` (Vitest/Jest) — a floor, not a target.
- Better: fail the build when coverage *decreases* vs. the base branch —
  absolute gates reward padding tests.
- Upload reports (codecov / coveralls) and comment PRs with coverage diffs.

## Reporting & observability

- Summarize per layer: pass/fail, retried-then-passed count, quarantine pass
  rate, coverage delta.
- Keep e2e traces/screenshots from failing runs downloadable for ~7 days.
- Alert on *trends* (flakiness rising), not on single failures.
