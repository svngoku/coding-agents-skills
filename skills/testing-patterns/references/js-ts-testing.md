# JavaScript/TypeScript Testing Deep Dive

JS/TS-specific depth for the testing-patterns skill. Read the main
[SKILL.md](../SKILL.md) first for the language-agnostic framework.

## Vitest vs Jest

| | Jest | Vitest |
|--|------|--------|
| Config | `jest.config.(js|ts)` | `vite.config.ts` `test` block |
| Speed | slower, CJS-oriented | fast, Vite-based, native ESM |
| Mocking | `jest.fn/mock/spyOn` | `vi.fn/mock/spyOn` (same API) |
| Watch | `--watch` | `--watch` (default in dev) |

The APIs are near-identical (`jest.*` → `vi.*`); examples below use
Vitest syntax.

## Mocking

```ts
import { vi, describe, it, expect, beforeEach } from "vitest";

const notify = vi.fn();

beforeEach(() => notify.mockClear());   // reset call history between tests

it("sends an invoice notification", () => {
  const svc = new OrderService(new FakeOrderRepo(), { notify });
  svc.placeOrder({ id: "o1", total: 99 });
  expect(notify).toHaveBeenCalledWith("o1", 99);
});
```

| API | Purpose |
|-----|---------|
| `vi.fn()` | standalone mock |
| `vi.spyOn(obj, "method")` | wrap a real method |
| `vi.mock("./module")` | replace an entire module (hoisted to top) |
| `vi.mock("./db", () => ({ query: vi.fn() }))` | factory form |
| `vi.hoisted()` | values referenced by a `vi.mock` factory |
| `vi.clearAllMocks()` / `vi.resetAllMocks()` | between tests |

Rules: mock at the module boundary (network, DB, clock) — not every object;
prefer dependency injection + fakes for code you own; never mock the module
whose internals you are asserting.

## Async testing

```ts
it("resolves the order total", async () => {
  await expect(loadOrderTotal("o1")).resolves.toBe(99);
});

it("rejects for a missing order", async () => {
  await expect(loadOrderTotal("nope")).rejects.toThrow("not found");
});
```

## Property-based testing with fast-check

```ts
import fc from "fast-check";

test("sorting is idempotent", () => {
  fc.assert(
    fc.property(fc.array(fc.integer()), (xs) => {
      expect([...xs].sort().sort()).toEqual([...xs].sort());
    })
  );
});
```

Arbitraries: `fc.integer()`, `fc.string()`,
`fc.record({...})`, `fc.array(arb)`, `fc.oneof(...)`. Build
custom ones with `.map()` / `.filter()` / `fc.arbitrary`.
Raise the run count with `fc.configureGlobal({ numRuns: 1000 })` in a
setup file.

## Playwright

Config essentials:

```ts
// playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["iPhone 13"] } },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
  },
});
```

Selector guidance — prefer role/label selectors over CSS:

| Prefer | Avoid |
|--------|-------|
| `getByRole("button", { name: "Submit" })` | `page.locator(".btn-submit")` |
| `getByLabel("Email")` | `page.locator("input[type=email]")` |
| `getByText("Order confirmed")` | brittle text locators on dynamic pages |

Playwright auto-waits for actionability; use `await expect(locator).toBeVisible()`
instead of `waitForTimeout()`.

Shard in CI:

```bash
npx playwright test --shard=1/4
npx playwright test --shard=2/4
```

## testcontainers-node

```ts
import { PostgreSqlContainer } from "@testcontainers/postgresql";

test("order persists", async () => {
  const container = await new PostgreSqlContainer("postgres:16").start();
  try {
    const pool = new Pool({ connectionString: container.getConnectionUri() });
    // ... test against a real database ...
  } finally {
    await container.stop();
  }
});
```

Start one container per suite (beforeAll/afterAll), not per test, to keep the
suite fast.

## Mutation testing with Stryker

```bash
npx stryker run
```

Configure in `stryker.conf.json`: `mutate: ["src/**/*.{ts,js}"]`,
`testRunner: "vitest"` or `"jest"`, thresholds like
`{ high: 80, low: 60 }`. Like mutmut, run on changed code in CI or
nightly.
