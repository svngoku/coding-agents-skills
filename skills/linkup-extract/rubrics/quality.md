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

The query should extract public events from the supplied listing and pagination/detail
links, not replace the seed with a broad web search. Explain user-confirmed synthetic beta
access and stopping on 403, single-row schema versus result envelope, same-id polling,
NDJSON download before 24-hour expiry, and credential isolation at the result host.
Do not promise arbitrary-site exhaustiveness, fixed cost, or permanent result URLs.
