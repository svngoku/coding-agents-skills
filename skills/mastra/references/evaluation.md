# Evaluation and Testing

Evaluate an agent as a system: deterministic contracts and security first, model-backed quality second,
then controlled production sampling. `@mastra/evals` and Vitest examples below were checked against
current documentation on 2026-11-08.

## Build the Evaluation Set From the Contract

Include representative and worst-case inputs, not only ideal prompts:

- Normal tasks and typed output correctness
- Ambiguous requests and required clarification
- Missing, stale, contradictory, and injected evidence
- Correct, incorrect, missing, and repeated tool calls
- Cross-tenant IDs and denied capabilities
- Maximum realistic context and tool result sizes
- Timeout, cancellation, provider failure, and malformed provider output
- Approval, decline, expiry, replay, restart, and resume
- Multi-agent misdelegation and nested failure

Keep ground truth and expected tool trajectories independent of the implementation's own output.
For security cases, assert exact denial and no business side effect rather than assigning a subjective
quality score.

## Deterministic Vitest Gates

```bash
npm install @mastra/evals vitest
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import { MastraEvalsReporter } from '@mastra/evals/vitest'

export default defineConfig({
  test: {
    reporters: ['default', new MastraEvalsReporter()],
    setupFiles: ['@mastra/evals/vitest/setup'],
  },
})
```

```typescript
import { expectEvals } from '@mastra/evals/vitest'
import { checks } from '@mastra/evals/checks'
import { test } from 'vitest'
import { mastra } from '../src/mastra'

test('weather agent follows its tool and answer contract', { timeout: 60_000 }, async () => {
  await expectEvals({
    target: mastra.getAgent('weatherAgent'),
    data: [{ input: "What's the weather in London?" }],
    gates: [
      checks.calledTool('weatherTool'),
      checks.noToolErrors(),
      checks.includes('London'),
    ],
    scorers: [{ scorer: mastra.getScorer('answerRelevancyScorer'), threshold: 0.7 }],
  }).toPass(0.8)
})
```

Always await `expectEvals` or `expectEval`. `toPass()` without an argument requires every item to pass
every gate; `toPass(0.8)` requires at least 80% to pass all gates. Scorer thresholds compare average
scores and must pass regardless of the gate pass rate. Give model-backed tests a suitable timeout.

Use `expectEval` with `test.for` or `test.each` for independently reported matrix cases. Limit both
Vitest file parallelism and `runEvals` concurrency to avoid multiplying provider rate-limit pressure.

## Direct `runEvals` Assertions

```typescript
import { runEvals } from '@mastra/core/evals'
import { checks } from '@mastra/evals/checks'
import { expect, test } from 'vitest'

test('agent quality', { timeout: 60_000 }, async () => {
  const result = await runEvals({
    target: mastra.getAgent('weatherAgent'),
    data: [{ input: "What's the weather in London?" }],
    gates: [checks.noToolErrors()],
    scorers: [mastra.getScorer('answerRelevancyScorer')],
  })

  expect(result).toHaveVerdict('passed')
  expect(result).toPassGates()
  expect(result).toHaveScoreAbove('answer-relevancy-scorer', 0.7)
})
```

Available matchers distinguish verdicts, gates, and scorer thresholds. `toPassGates()` and
`toPassThresholds()` fail when their respective configuration is absent, which helps catch an empty
safety check. Use workflow targets and workflow scorer configuration to evaluate output, steps, and
execution trajectories where needed.

## Separate the Evaluation Layers

| Layer | Examples | Release role |
| --- | --- | --- |
| Static/unit | Schema, policy, idempotency key, chunking, redaction | Every change; no model required |
| Integration | Fake tools/MCP, storage snapshots, resume, provider timeout | Every sensitive workflow change |
| Deterministic agent gates | Called tool, no tool errors, expected content/trajectory | Required task contract |
| Model-graded quality | Relevance, faithfulness, completeness, task completion | Calibrated quality signal |
| Human review | High-risk failures, ambiguity, reviewer agreement | Audit and rubric calibration |
| Production sampling | Live trace scoring by stable run/trace ID | Ongoing drift detection |

Built-in scorers come from `@mastra/evals/scorers/prebuilt`; deterministic checks come from
`@mastra/evals/checks`. Register scorers with Mastra when their scores must be persisted. Live scorer
sampling should be deterministic by trace/run ID and use eligibility filters so sensitive or unsuitable
runs are excluded.

## Model-Graded Quality Discipline

Define what each score means with concrete anchors and evidence requirements. Test the judge against
human-labeled examples, including subtle unsupported claims and prompt-injected outputs. Keep judge
input within its privacy and retention requirements. Do not let a model grade its own authorization
decision as a substitute for application assertions.

Track score distributions and worst cases, not just averages. A high mean relevance score cannot
offset one cross-tenant disclosure or duplicate refund. Set hard deterministic failure gates for those
events.

Measure models, prompts, retrieval, and workflow changes through the real entry point and production
configuration. Compare against a frozen baseline and include first-call/cold-start behavior where latency
matters. Save safe dataset/version/model/prompt metadata with results.

## Test Side Effects Without Live Providers

Use fake repositories and payment adapters that record attempted calls and enforce idempotency keys.
Simulate the critical crash point after recording an external success but before local receipt storage.
Then replay and assert one business effect. Use local fake MCP servers and deterministic embedding/vector
fixtures for retrieval contract tests.

Offline static evaluation of generated code, including this skill's harness, verifies artifact patterns
only. It does not compile Mastra code, call a model, or prove runtime safety. Label that limitation in
evaluation reports.

## Release Evidence

Report:

- Dataset and expected-output provenance
- Exact package, model, prompt, workflow, and retrieval versions
- Deterministic pass rate with every security gate at its worst case
- Scorer mean/distribution and failed examples
- p50/p95/p99 latency, token/cost, timeout, and rate-limit behavior
- Human disagreements, unresolved failure modes, and rollback criteria

## Evaluation Review Checklist

- [ ] Expected results come from independent contract evidence.
- [ ] Real entry points and production-like state are exercised.
- [ ] Denial, injection, replay, interruption, and no-evidence cases are included.
- [ ] Deterministic gates and model quality scores remain distinct.
- [ ] Assertions are awaited, timeouts are sufficient, and concurrency is bounded.
- [ ] Scorer persistence/sampling and privacy are configured deliberately.
- [ ] Offline/static checks are not misrepresented as runtime or live-model validation.
