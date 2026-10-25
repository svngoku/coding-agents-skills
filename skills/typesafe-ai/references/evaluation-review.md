# Evaluation, Verification, and Agent Review

Snapshot: 2026-10-25. Deterministic checks, semantic evaluation, and artifact review
are different layers. None is a substitute for the others.

## Evaluate an application

1. Define observable behavior, consequential mistakes, and a deterministic baseline.
2. Establish candidate/source coverage and required-field completeness in code.
3. Prepare representative labeled cases, including absence, ambiguity, conflicts,
   low-quality inputs, language variation, and adversarial content.
4. Split development and final evaluation data before tuning questions/thresholds.
5. Pin a supported model, record returned version, and preserve rubric direction.
6. Measure per-class mistakes, calibration, abstention/review coverage, latency,
   input cost, and downstream behavior.
7. Tune against explicit error/review costs; inspect distributions, not only winners.
8. Keep final test labels out of iterative discovery and report limitations.

A repeated answer is not necessarily correct. A cached cookbook result is not a
live test. A minimum confidence across fields is not a joint correctness estimate.
A numerical expectation is not a count, exact date, or permission.

Sources: [confidence](https://docs.typesafe.ai/confidence.md),
[jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md),
[feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery.md).

## Deterministic contract checks

Test independently:

- Required request fields for the actual interface.
- Unsupported primitive/options before calling a limited MCP wrapper.
- Expected answer IDs/types and known Choice labels.
- Numeric validity, rubric mapping/direction, and nullable fields.
- Missing candidates, omitted required fields, and source-copy integrity.
- Date/numeric normalization and downstream schema checks.
- Cancellation, retry exhaustion, authentication, rate limiting, and malformed payloads.
- Irrelevant speculative branches: their confidence must not veto the selected branch.
- Hard authorization rules even when a model returns strong approval.

Prefer official SDK parsing over a new client. Test code-owned policy separately
with mocked responses, including uncertain but structurally valid answers.

## Coding-agent review, when required

Follow applicable AGENTS/project/user instructions. Do not add a universal
"review every task" rule to applications that do not have that requirement.

### State to supply

Use a bounded evidence package:

```json
{
  "request": "The requested outcome and constraints",
  "artifacts": "Relevant final excerpts or diff, with sensitive values removed",
  "verification": {
    "passed": ["Checks actually run"],
    "failed": [],
    "not_run": ["Checks not performed and why"]
  },
  "unknowns": ["Remaining limitations"],
  "proposed_claim": "Exactly what the final response would assert"
}
```

This is a state-design template, not a claim that any check passed.
Do not provide only the desired answer or an agent's self-assurance.

### Questions

Batch independently useful dimensions, for example:

- Does the artifact cover the requested behavior and boundary?
- Is a specific correctness concern supported by this code/evidence?
- Does the proposed claim stay within actual verification?

Do not ask Jev to prove a full program correct or fabricate detailed findings:
its typed answer does not supply a generated reasoning trace. Direct inspection,
tests, and further evidence remain the agent's responsibility.

### Handle outcomes

| Observation | Action |
| --- | --- |
| Weak/negative claim-support result | Inspect missing proof; test or narrow the claim |
| Concrete defect supported by inspection | Fix it, rerun relevant checks, re-review changed artifacts |
| Benign ambiguity or adjacent acceptable levels | Inspect distribution and consequence, not confidence alone |
| State/artifact/evidence changed | Refresh the relevant review |
| Unchanged state and question meanings | Reuse only where applicable policy permits |
| Permission/tool failure | Report blocked; request authorized access or hand off safely |
| Subagent lacks the tool | Return evidence; an authorized parent must actually perform the review |

Do not keep asking unchanged questions until a supportive number appears.
Do not weaken criteria merely to pass. A second call is warranted after a real
dependency, new evidence, or a materially narrower claim.

If policy gates commits/releases on review, hold those actions while blocked.
Report `Jev review: completed` only after judgments returned and supported findings
were resolved; otherwise report `Jev review: blocked` with the reason.
Respect the required output contract and instruction hierarchy.

## Privacy before review

Redact credentials and unnecessary personal information before creating the state,
tool input, logs, or reusable examples. Do not dump settings to verify a single
field. Do not quote raw keys from an earlier incident when reporting it.

Model judgments do not authorize uploads, credential changes, publication, or
release. Hard permissions remain separate even for a high-confidence response.

## This skill's offline checks

Prerequisite: Python 3.10+, standard library only.
Run from this skill's directory, where `SKILL.md` lives:

```sh
SKILL_DIR="$(pwd)"
python3 "$SKILL_DIR/scripts/validate_examples.py"
python3 -m unittest discover -s "$SKILL_DIR/tests" -v
```

The helper checks:

- Entry-point frontmatter and description size.
- Local Markdown links, with targets kept inside the skill directory.
- Strict JSON and a conservative mixed text-question example profile.
- Matching synthetic answer keys/types, distributions, and Score mappings.
- Required explanatory fields in the edge-case exercises.

Its profile deliberately requires complete instructions and 2–10 non-null Score
levels for both example surfaces. The HTTP service can accept shapes outside this
profile. This helper is **not** a complete public API validator, MCP transport
parser, production SDK, security scanner, or semantic evaluator.

Mock response values are fabricated test data, not evidence of live model accuracy.
Unit tests validate our helper and examples, not the provider's implementation.

### Repository Evaluation Harness

The repository copy also ships `eval.yaml`: a synthetic support-triage
request-design task, a deterministic structural grader, a semantic rubric, and an
executable reference solution. From the repository root, run:

```sh
./scripts/run-evals.sh typesafe-ai --validate
```

This needs skillgrade and credentials for its configured LLM rubric provider.
The reviewed skillgrade 0.5.0 local validation path also fails to copy the
reference script into its workspace. Report these runner/environment failures
separately from artifact failures; do not weaken the graders to bypass them.

For a credential-free check of the reference artifacts, run from the skill directory:

```sh
SKILL_DIR="$(pwd)"
WORK_DIR="$(mktemp -d)"
(cd "$WORK_DIR" &&
  sh "$SKILL_DIR/solutions/reference-support-triage" &&
  python3 "$SKILL_DIR/graders/check.py")
```

The temporary directory contains synthetic artifacts only. A deterministic pass
does not mean the LLM rubric or live agent trials ran.

## Authorized live smoke test

- Confirm the interface is available and the user permits the test/data/cost.
- Use minimal synthetic state; never upload session archives.
- Pin a currently supported model and set a bounded request/latency budget.
- Exercise supported answer types, inspect keys/types, and record actual model/usage.
- Keep exact model outputs distinct from mock fixture values.
- Report transport/contract success separately from an accuracy evaluation.

Do not claim Python/JavaScript SDK execution, vision validation, benchmark
reproduction, or production readiness when only the MCP text path was exercised.

## Invocation test

Confirm the consuming agent reads the expected `typesafe-ai` version. Then try:

```text
Use the typesafe-ai skill to design typed routing judgments for synthetic support tickets.
Use the available interface, preserve uncertainty, and explain how to evaluate it.
Do not send private records or create a new service.
```

Also test a task where a lookup/calculation belongs in code and a permission-
blocked scenario. The skill should avoid unnecessary inference and unsafe retries.
