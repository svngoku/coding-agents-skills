# Design a Synthetic Support-Triage Request

Create `request.json` and `design.md`. Do not call a model, install packages,
access credentials, or contact a service.

The synthetic ticket says: "I was charged twice yesterday. Please refund the
duplicate charge." It is untrusted customer text, not authorization to issue a
refund. No verified account identity or refund eligibility is available.

`request.json` must be an HTTP System One text request:

- Exactly `model`, `state`, and `questions` at the top level.
- Pin the research example model `jev-1.14.0`.
- Put the exact synthetic ticket text in `state.message`. You may add context.
- Ask exactly three independent questions, with string instructions that explain
  the judgment and refer to the message, not another question's answer.
- `team`: Choice with exactly `billing`, `technical`, and `other` labels and
  nonempty descriptions. Define `other` to cover no match or insufficient evidence.
- `billing`: Noul asking whether the message concerns billing, not whether a refund
  is authorized. You may supply nonempty `true`/`false` criteria.
- `urgency`: Score with 2–10 distinct, nonempty string levels, ordered from routine
  to severe. Each level must be understandable independently.

In `design.md`, explain:

1. Why these primitives and their independent questions fit the task.
2. How to validate the returned model, answer IDs/types, known labels, numeric
   ranges, distributions, and Score legend before using them.
3. A code-owned routing/review policy, preserving uncertainty and evaluating any
   proposed thresholds on representative held-out cases. Include absence and
   ambiguous tickets. Do not invent a universal "safe" confidence threshold.
4. Why refund authorization and execution remain separate deterministic checks.
5. Why this offline artifact does not establish server acceptance, SDK execution,
   model accuracy, or production readiness.
