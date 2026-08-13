Score the agent's solution 0.0-1.0:

- **Manifest correctness**: `scalingo.json` is valid JSON with a `name`, a
  `formation` declaring the `web` process at a sensible amount/size, a non-empty
  `env` with per-app secret generation, and a PostgreSQL addon in
  `provider:plan` form.
- **Addon plan sanity**: the PostgreSQL plan is a real plan name (Starter tier
  for dev, Business/Dedicated for production), not invented or malformed, and
  follows the `provider:plan` convention.
- **Procfile correctness**: a `web:` process starts the app and binds
  `$PORT`; the command is a realistic Flask/Gunicorn invocation.
- **Operational completeness**: `deploy.sh` covers the full lifecycle —
  region setup, `scalingo create`, `env-set`, `git remote add` +
  `git push scalingo main`, addon provisioning, `scale`, and a post-deploy
  verification (`logs` or `run`).
- **Region/compliance awareness**: targets `osc-fr1` explicitly (or
  `osc-secnum-fr1` with a stated compliance reason for sensitive data) and is
  region-aware rather than region-agnostic; `--region`/`config --region` used
  consistently.
- **Script robustness**: `deploy.sh` is a runnable bash script
  (`set -euo pipefail`), no hardcoded secrets committed, reasonable quoting
  and error handling.
