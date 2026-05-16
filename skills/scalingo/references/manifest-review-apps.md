# scalingo.json Manifest and Review Apps

`scalingo.json` is the app manifest — a declarative file describing how an app should be configured when created from a one-click deploy button or as a review app. It's the Scalingo equivalent of Heroku's `app.json`, and the shapes are near-identical for common fields.

**Important scope**: the manifest is applied **only** when creating an app via one-click deploy or review apps. It is **not** re-read on ordinary `git push` deploys. Production addons, env vars, and formation are managed via the CLI/dashboard/Terraform after the initial creation.

## File location

At repo root, named `scalingo.json`. If both `scalingo.json` and `app.json` exist, `scalingo.json` wins.

If your code lives in a subdirectory (configured via the `PROJECT_DIR` env var), place the manifest in that subdirectory.

## Full example

```json
{
  "name": "Sample App",
  "description": "Example service illustrating the manifest",
  "logo": "https://example.com/logo.svg",
  "repository": "https://github.com/org/repo",
  "website": "https://example.com",
  "stack": "scalingo-22",
  "env": {
    "ENVIRONMENT": {
      "description": "Runtime environment identifier",
      "value": "staging"
    },
    "SECRET_KEY": {
      "description": "App secret for signing sessions",
      "generator": "secret"
    },
    "PUBLIC_URL": {
      "description": "URL of the deployed app",
      "generator": "url"
    },
    "ADMIN_URL": {
      "description": "Admin panel URL",
      "generator": "url",
      "template": "%URL%/admin"
    }
  },
  "formation": {
    "web":    { "amount": 2, "size": "L" },
    "worker": { "amount": 1, "size": "M" }
  },
  "addons": [
    {
      "plan": "postgresql:postgresql-starter-512",
      "options": { "version": "15" }
    },
    {
      "plan": "redis:redis-starter-128"
    }
  ],
  "scripts": {
    "first-deploy": "bundle exec rake db:migrate db:seed"
  }
}
```

## Field reference

### Top-level metadata

| Field | Type | Notes |
|---|---|---|
| `name` | string | Display name in the deploy page |
| `description` | string | Shown to the user before they confirm |
| `logo` | URL | Rendered on the deploy page |
| `repository` | URL | Source repo (used if URL param isn't provided) |
| `website` | URL | Project homepage |
| `stack` | string | Default stack — usually omit to let the platform pick |

### `env`

Keys are env var names. Values are objects describing how the variable is sourced:

- `value` (string) — hardcoded default
- `description` (string) — shown in the UI
- `required` (bool) — if true, user must supply a value
- `generator` — one of `secret` (random 64-char hex string), `url` (the app's Scalingo URL)
- `template` (string, with `generator: url`) — transform the URL; `%URL%` is the placeholder

**Review apps**: if a variable is listed in the manifest's `env`, it's populated from the generator/value rather than inherited from the parent app. This is the primary mechanism for preventing review apps from picking up production credentials.

### `formation`

A map of process type → `{ amount, size }`. Applied at creation time. Ordinary process types (`web`, `worker`, ...) are valid; size defaults to M if omitted.

### `addons`

Array of objects. Each addon is specified as `"<provider>:<plan>"`. Optional `options.version` pins a version.

Common plan identifiers:

- `postgresql:postgresql-starter-512`, `postgresql:postgresql-business-1024`
- `mysql:mysql-starter-512`
- `mongodb:mongo-starter-512`
- `redis:redis-starter-128`

Validate with `scalingo addons-providers` and `scalingo addons-plans <provider>`.

### `scripts`

- `first-deploy` (preferred) — runs once, after the first deployment of a review app or one-click app completes. Retried until it succeeds. Use for initial migrations and seed data.
- `postdeploy` (deprecated at manifest level; use `postdeploy` in the `Procfile` instead) — runs after every deployment of a review app except the first if `first-deploy` is defined.

For production apps, move `postdeploy` into the `Procfile`:

```
postdeploy: bundle exec rake db:migrate
```

This runs for every deploy of every linked app (parent + review apps). If you need review-app-only behavior, guard with an env var:

```json
{
  "env": { "IS_REVIEW_APP": { "value": "true" } }
}
```

Then in your script check `IS_REVIEW_APP`.

## Review apps

Review apps are ephemeral apps spawned from pull requests on a linked SCM. Enable on the parent app:

```bash
scalingo --app parent-app integration-link-update \
  --deploy-review-apps --destroy-on-close
```

A review app's default configuration:

- **Name**: `<parent>-pr<N>` (plus a hash for uniqueness)
- **Env**: parent's env **minus** any keys overridden in `scalingo.json`
- **Addons**: provisioned fresh per manifest; parent addons are **not** copied (data starts empty)
- **Formation**: from manifest, or 1 × M `web` default
- **Collaborators**: inherited from parent

### Lifecycle

- **Created**: when the PR is opened (auto) or manually via CLI
- **Redeployed**: on every push to the PR branch (auto-deploy)
- **Destroyed**: when the PR is closed (`--destroy-on-close`) or after a period of inactivity (`--hours-before-delete-on-stale`)

### Manually creating a review app

```bash
scalingo --app parent-app integration-link-manual-review-app <pr-number>
```

### Configuration flags

```bash
scalingo --app parent-app integration-link-update \
  --deploy-review-apps \
  --destroy-on-close \
  --hours-before-delete-on-close 2 \
  --destroy-stale \
  --hours-before-delete-stale 168 \
  --allow-review-apps-from-forks    # see security note below
```

### Fork safety

By default, PRs from forks **do not** spawn review apps — they would inherit the parent's env, which means secrets. Enable forks only if you trust the fork authors (rarely the case for public repos). A safer middle ground: use `scalingo.json`'s `env` to override every sensitive key with a generator, then enable fork PRs.

### Protecting review apps with basic auth

Review apps are publicly reachable by default. To gate them:

```json
{
  "env": {
    "HTTP_BASIC_AUTH_USER": { "value": "reviewer" },
    "HTTP_BASIC_AUTH_PASS": { "generator": "secret" }
  }
}
```

Your app code reads these vars and conditionally applies basic auth middleware. The presence of `HTTP_BASIC_AUTH_PASS` (generator returns a random secret per review app) is what distinguishes a review app from production, where the var is absent.

### Per-review-app behavior flag

Common pattern to branch app behavior:

```json
{
  "env": {
    "IS_REVIEW_APP": { "value": "true" }
  }
}
```

Production doesn't define `IS_REVIEW_APP`, review apps do. Use in your code:

```python
if os.environ.get("IS_REVIEW_APP"):
    # skip outbound webhooks to third parties
    # seed test data instead of pulling from prod S3
```

## One-click deploy button

The manifest powers the one-click deploy flow. Button URL:

```
https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo
```

With a specific branch:

```
https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo#develop
```

Embed in a README:

```markdown
[![Deploy on Scalingo](https://cdn.scalingo.com/deploy/button.svg)](https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo)
```

The button takes the user to a page that shows the manifest's `name`, `description`, `env` prompts, and `addons`. They pick a name and region, then the platform provisions everything and deploys.

Only public GitHub repos are supported for one-click deploy.

## Common patterns

### Generating a per-app secret

```json
{
  "env": {
    "JWT_SECRET": { "generator": "secret" }
  }
}
```

A different 64-char hex string per review app. Never commit secrets.

### Per-app canonical URL

```json
{
  "env": {
    "CANONICAL_HOST_URL": { "generator": "url" }
  }
}
```

`CANONICAL_HOST_URL` receives the review app's own URL (e.g. `https://my-app-pr42-abc.osc-fr1.scalingo.io`), useful for callback URLs and OAuth redirects.

### Skipping addons on review apps

If review apps don't need a full Business-tier database, override the manifest to request a Starter plan only in review apps. The manifest drives review-app addons, so it's already the simpler plan unless you explicitly set Business. Keep production addons on the Business plan via CLI/Terraform.

### Scoping a postdeploy to review apps only

In `scalingo.json`:

```json
{
  "env": { "RUN_TEST_SEED": { "value": "1" } },
  "scripts": { "first-deploy": "bundle exec rake db:migrate db:seed" }
}
```

In the Procfile:

```
postdeploy: [ -z "$RUN_TEST_SEED" ] && bundle exec rake db:migrate || true
```

Production apps (no `RUN_TEST_SEED`) run only `db:migrate`. Review apps get the seeded dataset via `first-deploy` once at creation.
