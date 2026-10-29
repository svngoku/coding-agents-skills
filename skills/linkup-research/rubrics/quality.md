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

The query should make comparison of primary 2024 cloud metrics tractable and acknowledge
non-comparability/unknowns. Explain task envelope versus completed output, failure error,
saved-id resumability, bounded backoff, and duplicate-spend risk after ambiguous submission.
Identify L as server default and M as the explicitly selected recommendation, not a default.
