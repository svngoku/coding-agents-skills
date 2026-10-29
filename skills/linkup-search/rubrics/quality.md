# Offline Plan Quality Rubric

Score 0–1 using the criteria below. Inspect `request-plan.json` and `design.md` as data only;
do not run candidate code or make live requests. The deterministic grader separately checks
transport, request shape, output/lifecycle, and policy. Reward reasoning and substantive query
quality here, not repetition of API keywords or formatting preferences.

- **0.4 — Task and query quality:** The query/instructions target the supplied scenario and
  ask for meaningful evidence at an appropriate scope, without invented entity facts.
- **0.4 — Integration reasoning:** The design explains the specific routing and response/lifecycle
  distinctions below, including important failure behavior rather than claiming execution success.
- **0.2 — Grounding and boundaries:** The explanation connects preserved sources, missing
  values, and bounded offline operation to this task. It distinguishes local planner notation
  from actual API fields and never claims keys, beta permissions, or successful live verification
  were checked.

## Task-Specific Criteria

The query must ask for official pricing discovery followed by retrieval of plan/product prices
with source URLs. Explain why deep is needed for the sequential discovered-URL scrape,
why includeSources wraps structured output as data/sources without a fourth outputType,
and why missing prices remain null. Penalize made-up pricing or a claim that standard/flash
can do the same discovered-URL chain.
