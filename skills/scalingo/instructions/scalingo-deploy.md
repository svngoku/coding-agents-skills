# Task: Prepare a Flask App for Deployment on Scalingo

You are preparing a small Python/Flask app (**flask-notes**) for deployment on
Scalingo, the European PaaS. Produce **three files in the current workspace**:

1. `scalingo.json` — the app manifest
2. `Procfile` — the process definition
3. `deploy.sh` — a bash script that deploys and verifies the app

There is no real Scalingo account and no network: the grader only inspects the
three files statically. Do not actually run `scalingo` or `git push`.

## App context

- **Stack**: Python 3 / Flask, app factory `app:app`, migrations via Flask-Migrate.
- **Managed database**: PostgreSQL, provisioned as a Scalingo addon.
- **Target region**: `osc-fr1` (Paris). The deploy script must be region-aware.
- **App name**: `flask-notes`.

## Requirements for `scalingo.json`

Valid JSON (no comments, no trailing commas) declaring at least:

1. A **`name`** field (display name, e.g. "Flask Notes").
2. **`formation`** with a **`web`** process: `amount >= 1` and a container
   `size` (e.g. `M`).
3. **`env`** — at least two variables: `SECRET_KEY` generated per app
   (`{ "generator": "secret" }`) and one app-level value such as
   `FLASK_ENV: production`.
4. **`addons`** — at least one addon: a PostgreSQL database with a valid plan
   in `provider:plan` form, e.g. `"postgresql:postgresql-starter-512"`.
5. Optionally `scripts` with a `first-deploy` command (e.g. run migrations
   once after the first deploy).

## Requirements for `Procfile`

A `web:` line that starts the web process and binds to the port Scalingo
injects — it **must reference `$PORT`**, e.g.:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

## Requirements for `deploy.sh`

A runnable bash script (`set -euo pipefail`) that performs the deployment
end-to-end. In order, it must:

1. **Set the region** — `scalingo config --region osc-fr1` and/or pass
   `--region osc-fr1` on the create command.
2. **Create the app** — `scalingo create flask-notes` (region-aware).
3. **Set environment variables** — `scalingo --app flask-notes env-set ...`
   (e.g. `SECRET_KEY` and `FLASK_ENV`).
4. **Add the git remote and push** — `git remote add scalingo <git-url>`
   then `git push scalingo main` (or `local-branch:main`).
5. **Provision the addon** — `scalingo --app flask-notes addons-add
   postgresql postgresql-starter-512`.
6. **Scale the web process** — `scalingo --app flask-notes scale web:1:M`.
7. **Post-deploy check** — verify the app came up, e.g. `scalingo --app
   flask-notes logs --lines 50` or `scalingo --app flask-notes run python -c
   "import app"`.

If anything is ambiguous, prefer the conventions the requirements above state
explicitly.
