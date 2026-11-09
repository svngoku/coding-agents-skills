# Harness Repo Bootstrap Quality Rubric

Inspect only the files under `ledger-api/` as text. Do not execute candidate code, install
packages, or make network requests. Score **0.0–1.0** using the weighted criteria below. The
deterministic grader checks structure and hardening; use this rubric for judgment it cannot make.

- **0.25 — AGENTS.md as a map:** Short, imperative, and specific to a double-entry ledger API.
  Commands match `pyproject.toml`. It links to `docs/` instead of pasting them, and includes the
  integer-minor-units money boundary in the Always/Ask/Never rules rather than as vague advice.
- **0.20 — Honest tailoring:** Docs describe this project where facts were given and use a visible
  `TODO` where they were not. No invented architecture, customers, SLAs, or metrics.
- **0.25 — Security posture and honesty:** The baseline is intact and not weakened. `handoff.md`
  lists the admin-only settings as outstanding and never claims branch protection, secret scanning,
  or Dependabot alerts are already on.
- **0.15 — Tooling coherence:** `pyproject.toml`, the package, the smoke test, CI, and the
  AGENTS.md commands all use the same `install`/`check` contract and the same package name.
- **0.15 — Scope discipline:** Only the requested files were produced. No commits, leftover
  `.env`, `*.bak`, or `*.harness-new` files, and no unrelated edits.

A strong result scores 0.9–1.0. Claiming admin-only settings are enabled, a floating action ref,
or a missing active exec-plan should keep the score below 0.5. Output a single number from 0.0 to
1.0 as the final score.
