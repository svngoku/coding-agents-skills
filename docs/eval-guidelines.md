# Evaluating Skills with skillgrade (Code-Generation Archetype)

This repo evaluates its skills with [skillgrade](https://github.com/mgechev/skillgrade) — "unit
tests for your agent skills". It runs a real coding agent against a task, then scores the agent's
output with a combination of deterministic checks and an LLM rubric.

We standardize on the **code-generation archetype**: the agent is asked to produce an artifact
(code, config, schema, commands) that the skill teaches, and a **deterministic grader** statically
checks that artifact for the correct API surface and best practices. This keeps evaluations
fast, cheap, and hermetic — no network calls to real services are needed. An LLM rubric adds a
small (30%) judgment component for things static checks can't see.

## Layout convention

Every skill that ships an evaluation follows this layout:

```
skills/<name>/
├── SKILL.md                  # unchanged
├── eval.yaml                 # the skillgrade harness (small; uses file references)
├── instructions/             # agent task prompts (file-referenced from eval.yaml)
├── graders/                  # deterministic grader scripts (python3, stdlib only)
├── rubrics/                  # llm_rubric criteria (file-referenced from eval.yaml)
└── solutions/                # reference solutions — enables `skillgrade --validate`
```

Fixtures (input files copied into the agent workspace) are optional — add a `fixtures/` dir only
when the task genuinely needs an input file.

## eval.yaml shape

```yaml
version: "1"

defaults:
  agent: command          # opencode via `command` (matches this repo's setup)
  command: opencode run
  provider: local         # local execution; docker optional for isolation
  trials: 5               # smoke: 5 | reliable: 15 | regression: 30
  timeout: 300            # seconds before the agent is killed
  threshold: 0.8          # minimum pass rate for --ci mode

tasks:
  - name: write-agent
    instruction: instructions/write-agent.md   # file reference
    solution: solutions/reference-write-agent  # for --validate
    graders:
      - type: deterministic
        run: python3 graders/check.py          # file reference
        weight: 0.7
      - type: llm_rubric
        rubric: rubrics/quality.md             # file reference
        weight: 0.3
```

## Writing a task (rules of thumb)

1. **Grade outcomes, not steps.** Check that the artifact exists and is correct, not which
   commands the agent ran.
2. **Name the output files in the instruction.** If the grader checks `agent.py`, the task must
   say "save the result as agent.py". The grader runs in the workspace after the agent finishes.
3. **Keep deterministic checks meaningful, not pedantic.** Check the API surface and the
   best practices the SKILL.md emphasizes (idempotency keys, security boundaries,
   `model_id@provider` format, smoke-test-first, …). 3–5 strong checks beat 20 noisy ones.
4. **Deterministic graders are hermetic.** python3, standard library only, no network. They
   print JSON to stdout: `{"score": 0.67, "details": "2/3 passed", "checks": [...]}`
   (`score` 0–1 and `details` required, `checks` optional).
5. **Always ship a reference solution** in `solutions/` and run `--validate` — it proves the
   grader scores a known-good answer ~1.0 and catches graders that drift into false passes.
6. **Rubrics judge what static checks can't.** Workflow discipline, security posture,
   readability, trade-off reasoning. Keep criteria concrete and 0–1 scorable.

## Running evaluations (local, no CI)

```bash
npm i -g skillgrade

# all skills that have an eval.yaml (smoke: 5 trials)
./scripts/run-evals.sh

# one skill
./scripts/run-evals.sh langchain

# higher-confidence runs
./scripts/run-evals.sh --mode=reliable     # 15 trials
./scripts/run-evals.sh --mode=regression   # 30 trials

# verify every grader against its reference solution first
./scripts/run-evals.sh --validate

# pick a different agent (opencode is the default in eval.yaml)
GEMINI_API_KEY=... ./scripts/run-evals.sh langchain --agent=gemini
```

Reports are written to `.evals/<skill>/` (gitignored). `--ci` makes the run exit non-zero
when a skill drops below its `threshold`.

## Cost guidance

- **The dominant cost is agent trials**, not grading: the LLM grader defaults to a small
  Flash/Haiku-class model. Budget per run ≈ trials × tasks × agent cost per task.
- `--smoke` (5 trials) for day-to-day iteration; `--reliable`/`--regression` for release
  confidence. 3–5 well-designed tasks beat 50 noisy ones.
- Run `--validate` after any change to a task, grader, or solution.

## Current coverage

| Skill | Harness | Archetype | Status |
|-------|---------|-----------|--------|
| adaption-ai | ✅ | code-gen (deterministic 0.7 / rubric 0.3) | ready |
| langchain | ✅ | code-gen | ready |
| smolagents | ✅ | code-gen | ready |
| genai-tk | ✅ | code-gen | ready |
| unsloth-hf-jobs | ✅ | code-gen | ready |
| database-design | ✅ | code-gen | ready |
| api-design | ✅ | code-gen | ready |
| security-best-practices | ✅ | code-gen | ready |
| testing-patterns | ✅ | code-gen | ready |
| performance-optimization | ✅ | code-gen | ready |
| ddd | ✅ | code-gen | ready |
| microservices-patterns | ✅ | code-gen | ready |
| ui | ✅ | code-gen | ready |
| scalingo | ✅ | code-gen | ready |
