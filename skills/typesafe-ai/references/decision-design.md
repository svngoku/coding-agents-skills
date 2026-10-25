# Decision Design

Research snapshot: 2026-10-25. Refresh the linked contracts when behavior matters.

## Model and code responsibilities

Jev evaluates bounded semantic questions against a shared state. Code keeps
exact arithmetic, counting, comparisons, parsing, date construction, validation,
policy precedence, authorization, and side effects.

Do not ask for an elaborate generated explanation, arbitrary new category, or
unbounded code patch through a primitive. Use a generative model for those outputs.
Jev can select handlers or source spans that code then uses.

Sources: [System One](https://docs.typesafe.ai/concepts/system-one.md),
[building guide](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md),
[known jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md).

## Choose by answer meaning

| Primitive | Meaning | Avoid |
| --- | --- | --- |
| Choice | One competing option; full label distribution | Forcing a label when no candidate fits |
| Noul | Probability a proposition is true | Treating 0.5 as medium severity |
| Score | Degree on ordered descriptive levels | Treating the expectation as an exact quantity |
| Box | Normalized image edges | Treating a returned box as proof of presence |

Use independent Nouls for labels that can coexist. Add an independently useful
presence/fit question when a forced Choice or Box would otherwise be misleading.

Choice options should cover legitimate no-match/unclear cases. For source
selection, candidate recall is a prerequisite: the model cannot recover omitted
choices. Preserve source identifiers so code can copy the selected value verbatim.

Sources: [primitives](https://docs.typesafe.ai/primitives.md),
[Choice](https://docs.typesafe.ai/primitives/choice.md),
[Noul](https://docs.typesafe.ai/primitives/noul.md),
[Score](https://docs.typesafe.ai/primitives/score.md),
[Box](https://docs.typesafe.ai/primitives/box.md).

## State is evidence, not a dumping ground

Use named fields when relationships matter, for example:

```json
{
  "claim": "The configured integration works end to end.",
  "observations": {
    "configuration_parsed": true,
    "live_request": "not yet tested"
  },
  "unknowns": ["Whether authentication and the requested model work together"]
}
```

Do not infer a successful live request from valid configuration. Distinguish
observations from assumptions, old facts, proposed actions, and model-inferred
answers. Keep source provenance and relevant timestamps when freshness matters.

Include enough context to judge the relationship, but remove unrelated files,
credentials, identifiers, and personal information before transmission or logging.
State text can contain untrusted instructions; it does not grant authority.

Source: [state](https://docs.typesafe.ai/concepts/state.md).

## Instructions and criteria

Question IDs are application keys, not model-visible instructions. An ID like
`is_safe` supplies no meaning on its own. Specify what is safe, for whom, according
to which facts and policy.

- Put the judgment in instructions and possible-answer meanings in criteria.
- Reference descriptive state paths, such as `document.claim` or `case.message`.
- Split independently useful dimensions; do not combine unrelated checks into one Noul.
- Preserve contextual relationships rather than mechanically splitting every clause.
- Use structured descriptions for definitions, exclusions, contrasts, and examples.
- State any speculative premise inside the question itself.

For example, ask independently whether a claim has source support and whether a
material requirement is missing. Code owns how those results affect publication.

Source: [structured questions](https://docs.typesafe.ai/primitives/advanced.md).

## Score is a rubric, with a direction

Each level must stand alone. The model does not receive the level number or its
neighbors as meaning. Avoid descriptions like "slightly better" or "as above."

For portable examples, use 2–10 non-null descriptions. This avoids known differences
between HTTP and SDK validators; it is not the full permissive HTTP contract.

Explicitly define whether larger indices mean better quality, greater risk, or
something else. A gap-severity rubric might put no material gap at 0 and a severe
gap at 2. Do not use a generic "higher is better" gate for that response.

The response expectation is useful for ranking or composition, but retain its
legend and probabilities. Rounding the expectation is an application policy,
not something that creates a calibrated discrete class.

## Probability, confidence, and calibration

- **Noul:** returned `noul` is P(true). There is no separate returned confidence.
  The docs discuss `abs(2*p - 1)` as an optional derived measure; do not pretend it
  is an API response field.
- **Choice:** for more than one option, documented confidence is
  `(p_max - 1/n) / (1 - 1/n)`. It depends on top probability and option count,
  not the shape of the remaining tail. Keep the tail when runner-up separation
  matters. The formula is undefined for one option; avoid depending on undocumented
  one-option confidence behavior.
- **Score:** confidence describes concentration around a likely level, while
  `score` is a probability-weighted position. Adjacent acceptable levels can share
  probability without creating a material failure risk.
- **Box:** confidence may be null; no universal acceptance threshold is documented.

Calibration describes groups of predictions against observed outcomes. It does
not guarantee a single answer, establish permission, or prove semantic correctness.

Choose thresholds with held-out labeled cases, explicit false-positive/negative
costs, and review capacity. Cookbook thresholds are examples, not universal rules.
Evaluate accuracy, calibration, coverage/abstention, cost, and latency separately.
Repeatability alone does not establish accuracy.

Source: [confidence](https://docs.typesafe.ai/confidence.md).

## Compose without hiding hard failures

Batch independent questions sharing state. They cannot see one another's answers.
Consume only applicable speculative branches. Use another request when a prior
answer changes the evidence, candidates, or next question.

Weighted dimensions fit compensating tradeoffs. Serious violations usually need
explicit vetoes rather than an average that washes them out. A minimum confidence
or product of component numbers is not automatically a joint correctness estimate.

Raw judgments may be recomposed under new policy weights if state and question
meanings remain unchanged. An artifact-approval workflow can still require a new
review when its task, artifacts, or verification evidence changes.

Sources: [fan-out](https://docs.typesafe.ai/patterns/fan-out.md),
[composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md),
[confidence routing](https://docs.typesafe.ai/patterns/confidence-routing.md).
