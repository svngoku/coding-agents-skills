# Deployment on Scalingo

Four deployment methods. Pick based on where your code lives and how much automation you want.

| Method | When to use | Trigger |
|---|---|---|
| [Git push](#git-push) | Local development, full control | `git push scalingo main` |
| [GitHub integration](#github--github-enterprise) | Team workflow, review apps per PR | Push to linked branch |
| [GitLab integration](#gitlab) | Same as GitHub, GitLab-hosted repos | Push to linked branch |
| [Archive](#archive-deploy) | CI-built artifacts, no git history exposed | API call with tarball URL |
| [JAR / WAR](#jar--war) | JVM apps with prebuilt artifacts | CLI upload |

## The deployment pipeline

Every method lands in the same pipeline:

1. **Ingest** — source code or pre-built artifact arrives at Scalingo
2. **Build** — a buildpack detects the stack, installs dependencies, produces a slug (compiled artifact)
3. **Release** — if a `release` process type exists in the `Procfile`, it runs here (e.g. migrations)
4. **Swap** — new containers start; routers wait for them to be healthy; old containers stop only once new ones serve traffic (zero-downtime)
5. **Postdeploy** — optional `postdeploy` process type from `Procfile` runs once after the new version is serving

The build is cached between deploys — `scalingo deployment-cache-delete` forces a full rebuild.

## Git push

Each app has its own git repo on Scalingo's side. Deploying is a push to the remote.

```bash
# Create the app
scalingo create my-app

# Get the remote URL (region-dependent)
scalingo --app my-app git-show
# → git@ssh.osc-fr1.scalingo.com:my-app.git

# Add the remote and push
git remote add scalingo git@ssh.osc-fr1.scalingo.com:my-app.git
git push scalingo main
```

Only `master` and `main` are accepted on the remote. To deploy a different local branch:

```bash
git push scalingo feature-x:main
```

To deploy a specific commit (not just HEAD):

```bash
git push scalingo <commit-sha>:main
```

To re-clone a Scalingo-hosted repo (e.g. after losing a local copy):

```bash
git clone --origin scalingo git@ssh.osc-fr1.scalingo.com:my-app.git
```

Only git-pushed code is in that repo — deploys made via GitHub integration, archive, or JAR/WAR don't populate it.

## GitHub / GitHub Enterprise

Linking an app to GitHub enables:

- Auto-deploy on push to a branch
- Manual deploy from any branch
- Review apps (ephemeral app per pull request)

### Dashboard setup (recommended for the initial link)

1. Account → Integrations → GitHub → Connect
2. App → Deploy → Configuration → choose repo and branch
3. Toggle Auto-deploy and/or Review apps

### CLI setup

You need a GitHub personal access token with `repo` scope:

```bash
# Link your Scalingo user to GitHub
scalingo integrations-add github --token <github-pat>

# Link an app to a repo (creates the integration-link)
scalingo --app my-app integration-link-create \
  --auto-deploy --branch main \
  https://github.com/org/repo

# Subsequent updates
scalingo --app my-app integration-link-update --branch develop
scalingo --app my-app integration-link-update --no-auto-deploy

# Enable review apps (with destroy-on-close)
scalingo --app my-app integration-link-update \
  --deploy-review-apps --destroy-on-close

# Manual deploy (from a branch, bypassing auto-deploy)
scalingo --app my-app integration-link-manual-deploy main
```

GitHub Enterprise works identically with a custom host: replace `github.com` in the URL with your GHE host.

### Fork review apps

Review apps from forked PRs are **disabled by default** — they inherit the parent app's environment, which would leak secrets to untrusted forks. Enable explicitly only if you trust the forks:

```bash
scalingo --app my-app integration-link-update --allow-review-apps-from-forks
```

A safer alternative: override sensitive env vars in `scalingo.json` so review apps don't inherit them.

## GitLab

Works the same as GitHub — link a GitLab user account under Account → Integrations → GitLab, then:

```bash
scalingo --app my-app integration-link-create \
  --auto-deploy --branch main \
  https://gitlab.com/org/repo
```

Self-hosted GitLab instances work by passing the full URL to your instance.

## Archive deploy

Deploy a `.tar.gz` of your source code fetched from a URL. Useful for CI pipelines that build artifacts independently.

Archive shape — code must live inside a subdirectory (commonly `master/`) at the archive root:

```
my-app.tar.gz
└── master/
    ├── package.json
    ├── Procfile
    └── src/...
```

One-liner to create such an archive from a git repo:

```bash
git archive --prefix=master/ main | gzip > my-app.tar.gz
```

Deploy from a URL:

```bash
scalingo --app my-app deploy https://example.com/my-app.tar.gz
# Optional second arg: a git ref for the deployment record
scalingo --app my-app deploy https://example.com/my-app.tar.gz v1.2.3
```

The URL must be reachable from Scalingo's build infrastructure (public, or behind auth encoded in the URL).

### GitHub tarball shortcut

GitHub serves repo tarballs under a predictable URL:

```bash
scalingo --app my-app deploy \
  https://github.com/org/repo/archive/refs/heads/main.tar.gz
```

Useful for the one-click deploy button flow.

## JAR / WAR

For JVM apps with prebuilt artifacts. Skip the build step entirely.

```bash
scalingo --app my-app deploy-jar ./target/app.jar
scalingo --app my-app deploy-war ./target/app.war
```

The platform wraps the artifact in a minimal runtime (Java 8/11/17/21 depending on env configuration — see `JAVA_VERSION` env var). Your `Procfile` still decides how `web` starts, typically:

```
web: java -jar target/app.jar
```

## Auto-deploy workflow notes

- Auto-deploy triggers on **every** push to the linked branch, not just the default branch
- A failed build does not roll back — containers keep running the previous version
- A failed `release` process type (migration) aborts the deploy — containers are not replaced
- Review apps created from PRs get `parent-app-name-pr<N>` naming by default

## One-click deploy buttons

A URL that deploys a `scalingo.json`-equipped public repo when clicked:

```
https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo
```

With a custom branch:

```
https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo#develop
```

Embed as a button in a README:

```markdown
[![Deploy](https://cdn.scalingo.com/deploy/button.svg)](https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo)
```

The `scalingo.json` manifest controls which addons and env vars the new app gets — see `references/manifest-review-apps.md`.

## Build cache

Caches go per-app. Buildpacks decide what to cache (npm deps, wheel files, compiled binaries). When a deploy fails inexplicably after working previously, and a dependency is suspect, clear it:

```bash
scalingo --app my-app deployment-cache-delete
```

Next deploy will rebuild from scratch.

## Troubleshooting common deployment errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `! [remote rejected]  main -> main (pre-receive hook declined)` | You don't have admin on the app | `scalingo --app <n> collaborators` to check |
| `Push rejected, missing Procfile` | No `Procfile` and no default process type from buildpack | Add a `Procfile` |
| `Detected buildpack: …` wrong | Framework-file present but minor | Set `BUILDPACK_URL` env to force a specific buildpack |
| Build succeeds, app crashes at boot | Missing env var or port binding | Check `logs`, verify app binds to `$PORT` |
| Build runs old code | Stale cache | `deployment-cache-delete`, redeploy |
| Review apps inherit prod credentials | No `scalingo.json` env overrides | Add `env` block to `scalingo.json` |
