# Bootstrap an Agent-First Repository for `ledger-api`

Use the harness-repo skill to bootstrap a new **Python** project named `ledger-api` in
`./ledger-api` (relative to the current workspace). Its one-line description is:
`Double-entry ledger HTTP API for internal finance tools.`

This is an offline task. Do not push, call GitHub, enable any repository setting, install
packages, or make network requests. Running the bundled scaffold script and `node` locally is fine.

## Required result

1. The agent-first baseline from the skill exists under `ledger-api/`: a tailored `AGENTS.md`,
   a one-line `CLAUDE.md` pointer, the seven top-level docs under `docs/`, the design-docs,
   exec-plans and product-specs folders, an active initial exec-plan linked from `docs/PLANS.md`,
   the security baseline (`SECURITY.md`, `docs/SECURITY.md`, CODEOWNERS, dependabot, PR template,
   pre-commit config, `.env.example`), a hardened `.github/workflows/ci.yml`, and the
   golden-principles linter.
2. Create the Python tooling the commands need: `pyproject.toml` (project name `ledger-api`,
   ruff, mypy strict, pytest), a `ledger_api` package that runs with `python -m ledger_api`, and one
   smoke test under `tests/`.
3. Tailor `ledger-api/AGENTS.md` so it is true for this project. Keep it under 150 lines, keep the
   commands consistent with `pyproject.toml`, and add this repo-specific boundary: money amounts are
   integer minor units (for example cents) and must never be stored or computed as floats.
4. Every diagram is Mermaid. Leave a visible `TODO` instead of inventing project detail.
5. Write `ledger-api/handoff.md`: a short summary of what was created and a section listing the
   security settings that **still need a repository admin** (branch protection, secret scanning with
   push protection, Dependabot alerts, private vulnerability reporting, CODEOWNERS handles). Do not
   claim any of them is already enabled.

Do not commit, and do not leave `.env`, `*.bak` or `*.harness-new` files behind.
