# Deployment on Scalingo

Four deployment methods (there is no deploy-from-Docker-image option). Pick based on where your code lives and how much automation you want.

| Method | When to use | Trigger |
|---|---|---|
| [Git push](#git-push) | Local development, full control | `git push scalingo main` |
| [GitHub integration](#github--github-enterprise) | Team workflow, review apps per PR | Push to linked branch |
| [GitLab integration](#gitlab) | Same as GitHub, GitLab-hosted repos | Push to linked branch |
| [Archive](#archive-deploy) | CI-built artifacts, no git history exposed | `scalingo deploy` with a tarball path or URL |
| [WAR](#war) | JVM apps with a prebuilt WAR | `scalingo deploy --war` |

## The deployment pipeline

Every method lands in the same pipeline:

1. **Ingest** — source code or pre-built artifact arrives at Scalingo
2. **Build** — a buildpack detects the stack, installs dependencies, produces a slug (compiled artifact)
3. **Postdeploy** — if the `Procfile` has a `postdeploy` line, it runs once in a separate one-off container (size M, 20-minute limit) before traffic is switched (e.g. migrations)
4. **Swap** — new containers start; routers wait for them to be healthy; old containers stop only once new ones serve traffic (zero-downtime)

There is no `release` process type on Scalingo — a Heroku `release:` line must become `postdeploy:`.

Limits: the build runs in a container with 6 CPUs and 8 GB of RAM, must finish within 30 minutes, and the resulting image can't exceed 2 GiB.

The build is cached between deploys — `scalingo deployment-delete-cache` forces a full rebuild.

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

Only git-pushed code is in that repo — deploys made via GitHub integration, archive, or WAR don't populate it.

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

Deploy a `.tar.gz` of your source code, from a local file or a URL. Useful for CI pipelines that build artifacts independently.

Archive shape — code must live inside a single top-level folder (any name; `master/` is just the usual example):

```text
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

Deploy from a local file or a URL:

```bash
scalingo --app my-app deploy my-app.tar.gz
scalingo --app my-app deploy https://example.com/my-app.tar.gz
# Optional second arg: a version reference (tag, commit SHA) for the deployment record
scalingo --app my-app deploy https://example.com/my-app.tar.gz v1.2.3
# Return immediately instead of following the build
scalingo --app my-app deploy --no-follow my-app.tar.gz v1.2.3
```

The URL must be reachable from Scalingo's build infrastructure (public, or behind auth encoded in the URL).

### GitHub tarball shortcut

GitHub serves repo tarballs under a predictable URL:

```bash
scalingo --app my-app deploy \
  https://github.com/org/repo/archive/refs/heads/main.tar.gz
```

Useful for the one-click deploy button flow.

## WAR

For JVM apps with a prebuilt WAR, the `deploy` command takes a `--war` flag:

```bash
scalingo --app my-app deploy --war ./target/app.war
```

There is no `deploy-jar` or `deploy-war` command, and no JAR upload: for JAR-based apps, deploy the source and let the Java buildpack build it. The `Procfile` still decides how `web` starts, typically:

```text
web: java -jar target/app.jar
```

## Auto-deploy workflow notes

- Auto-deploy triggers on **every** push to the linked branch, not just the default branch
- A failed build does not roll back — containers keep running the previous version
- A failed `postdeploy` command (e.g. a migration) fails the deploy — the old containers keep serving
- Review apps created from PRs get `parent-app-name-pr<N>` naming by default

## One-click deploy buttons

A URL that deploys a `scalingo.json`-equipped public repo when clicked:

```text
https://dashboard.scalingo.com/create/app?source=https://github.com/org/repo
```

With a custom branch:

```text
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
scalingo --app my-app deployment-delete-cache
```

Next deploy will rebuild from scratch.

## Troubleshooting common deployment errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `! [remote rejected]  main -> main (pre-receive hook declined)` | You don't have admin on the app | `scalingo --app <n> collaborators` to check |
| `Push rejected, missing Procfile` | No `Procfile` and no default process type from buildpack | Add a `Procfile` |
| `Detected buildpack: …` wrong | Framework-file present but minor | Set `BUILDPACK_URL` env to force a specific buildpack |
| Build succeeds, app crashes at boot | Missing env var or port binding | Check `logs`, verify app binds to `$PORT` |
| Build runs old code | Stale cache | `deployment-delete-cache`, redeploy |
| Review apps inherit prod credentials | No `scalingo.json` env overrides | Add `env` block to `scalingo.json` |
