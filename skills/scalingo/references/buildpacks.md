# Buildpacks and the Scalingo Stack

Scalingo uses **buildpacks** — the same open-source ecosystem Heroku pioneered — to detect your app's stack, install dependencies, and produce a runnable slug. For most apps, detection is automatic and you never touch this. When it matters: a non-standard runtime, native dependencies, polyglot apps, or a custom build step.

## The stack

The stack is the Ubuntu-based base image your buildpack runs against. Current stack: `scalingo-22` (Ubuntu 22.04). Pinned at app-creation time; upgrade via:

```bash
scalingo --app my-app stacks-set scalingo-22
```

The stack image is public on Docker Hub as `scalingo/scalingo-22` — useful for local testing of buildpacks or reproducing build environments:

```bash
docker run --rm -it \
  -e STACK=scalingo-22 \
  -v "$(pwd):/build" \
  scalingo/scalingo-22:latest bash
```

## Officially supported buildpacks (auto-detected)

| Language/framework | Detection signal |
|---|---|
| Ruby | `Gemfile` |
| Python | `requirements.txt`, `Pipfile`, `pyproject.toml`, `setup.py` |
| Node.js | `package.json` |
| PHP | `composer.json` |
| Go | `go.mod` (or `Godeps/`) |
| Java | `pom.xml`, `build.gradle`, `*.jar` |
| Elixir | `mix.exs` |
| Clojure | `project.clj` |
| Scala | `build.sbt` |
| Meteor | `.meteor/` |
| Static | `Staticfile` |

If multiple signals exist, the first match in detection order wins. To override, use a custom or multi buildpack (see below).

## How detection picks a runtime version

Each buildpack reads the language-standard version file:

- **Node.js** — `"engines": { "node": "20.x" }` in `package.json`, or `.nvmrc`
- **Python** — `runtime.txt` (one line: `python-3.12.7`)
- **Ruby** — `ruby "3.2.2"` directive in `Gemfile`
- **PHP** — `"require": { "php": "^8.2" }` in `composer.json`
- **Java** — `system.properties` file with `java.runtime.version=21`
- **Go** — `go.mod`'s `go 1.22` directive

Pin these in your repo. Relying on defaults means silent upgrades when buildpacks update.

## Forcing a specific buildpack

Set the `BUILDPACK_URL` env var before deploying:

```bash
scalingo --app my-app env-set BUILDPACK_URL=https://github.com/Scalingo/rust-buildpack
scalingo --app my-app deployment-cache-delete
git push scalingo main
```

Use this when:

- The auto-detected buildpack is wrong (e.g. a repo has both `package.json` and `Gemfile` and you want Ruby)
- You need a community buildpack (Rust, Crystal, Deno, Bun, Nim, ...)
- You maintain your own buildpack for a non-standard runtime

## Multi buildpack (polyglot apps)

A single build that runs multiple buildpacks in order — useful when a Python app needs Node.js for frontend assets:

```bash
scalingo --app my-app env-set BUILDPACK_URL=https://github.com/Scalingo/multi-buildpack
```

Then create a `.buildpacks` file at repo root:

```
https://github.com/Scalingo/nodejs-buildpack
https://github.com/Scalingo/python-buildpack
```

Buildpacks run top-to-bottom. The **last** buildpack determines the process types (the `Procfile`-derived ones), so order the runtime that provides the `web` server last.

## Custom buildpacks

A buildpack is a git repo with three scripts:

```
bin/
├── detect      # Exit 0 if this buildpack applies
├── compile     # Install dependencies, compile the app
└── release     # Print YAML describing addons/env/default-Procfile
```

### `bin/detect`

Takes a build directory as argument. Non-zero exit = skip this buildpack:

```bash
#!/usr/bin/env bash
if [ -f "$1/Cargo.toml" ]; then
  echo "Rust"
  exit 0
fi
exit 1
```

### `bin/compile`

Takes `BUILD_DIR`, `CACHE_DIR`, `ENV_DIR` as arguments. Does the work:

```bash
#!/usr/bin/env bash
BUILD_DIR=$1
CACHE_DIR=$2

cd "$BUILD_DIR"
# Install Rust toolchain if not cached
if [ ! -d "$CACHE_DIR/rustup" ]; then
  curl -sSf https://sh.rustup.rs | RUSTUP_HOME=$CACHE_DIR/rustup sh -s -- -y
fi
export PATH="$CACHE_DIR/rustup/bin:$PATH"

cargo build --release
mkdir -p "$BUILD_DIR/bin"
cp target/release/my-app "$BUILD_DIR/bin/"
```

At the end, the contents of `BUILD_DIR` are moved to `/app` in the final slug.

### `bin/release`

Prints YAML metadata — addons, config vars, default Procfile entries. Applied only on first deploy:

```bash
#!/usr/bin/env bash
cat <<EOF
---
addons:
  - scalingo-postgresql
config_vars:
  PATH: /app/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
default_process_types:
  web: /app/bin/my-app
EOF
```

If `default_process_types` is provided, users don't need a `Procfile`. If omitted, they do.

### `.profile.d/`

Scripts dropped in `.profile.d/` by buildpacks are sourced at container startup — used to set env vars that depend on runtime state (e.g. compiler paths). Don't write user scripts here; use `.profile` at repo root for app-specific init.

### Testing a custom buildpack locally

```bash
# Get the stack image
docker pull scalingo/scalingo-22

# Mount the buildpack and the app, run the three scripts manually
docker run --rm -it \
  -e STACK=scalingo-22 \
  -v /path/to/my-buildpack:/buildpack \
  -v /path/to/my-app:/build \
  scalingo/scalingo-22:latest bash

# Inside the container:
/buildpack/bin/detect /build      # Should exit 0
/buildpack/bin/compile /build /cache /env
/buildpack/bin/release /build
```

Once the buildpack works, push it to a public git host and set `BUILDPACK_URL`.

## Common buildpack patterns

### Install a system package

Use the `apt` buildpack (Scalingo maintains a fork):

```bash
scalingo --app my-app env-set BUILDPACK_URL=https://github.com/Scalingo/multi-buildpack
```

`.buildpacks`:

```
https://github.com/Scalingo/apt-buildpack
https://github.com/Scalingo/python-buildpack
```

`Aptfile` at repo root:

```
imagemagick
libpq-dev
ffmpeg
```

Packages are installed into `/app/.apt` and added to PATH/LD_LIBRARY_PATH.

### Use a specific Node version not in the default list

The Node buildpack respects `engines.node` in `package.json`. Pin `"node": "20.11.1"` for exact version, `"20.x"` for latest 20.

### Python with C extensions

`requirements.txt` with packages like `psycopg2-binary`, `Pillow`, `lxml` usually "just works" on `scalingo-22`. For packages needing build-time system headers, add an `Aptfile` with `-dev` packages and use the multi buildpack with apt first.

### Compile-once binaries

For languages like Rust or Go where the build output is a single binary, consider the archive deploy method: build in CI, upload the binary and a minimal Procfile as a tarball, skip the buildpack step entirely.

## Private buildpacks

`BUILDPACK_URL` accepts any git URL. For a private repo, use the HTTPS form with an access token in the URL (less secure) or host on an internal git server reachable from Scalingo. SSH-based private git URLs aren't supported for buildpacks.

## Build failures — triage order

1. Check `scalingo logs` during the build — the buildpack prints what it's doing
2. Verify the detected buildpack matches your expectation (first line of build output: `-----> Detected buildpack: X`)
3. If a dependency install fails, check if the version is too new/old for the stack — pin explicitly
4. If a native build fails, check whether the needed system package is present (use `apt` buildpack if not)
5. If everything looks right but fails: clear the cache (`scalingo deployment-cache-delete`) and retry
6. Reproduce locally in the `scalingo/scalingo-22` Docker image

Common specific errors:

- `Could not determine Ruby version` → add `ruby "3.2.2"` to `Gemfile`
- `No such file or directory: pip` → the Python buildpack didn't detect; check for `requirements.txt` / `Pipfile`
- `Error while executing gyp` → a native Node dependency; may need the apt buildpack for `build-essential`, or prefer prebuilt wheels
- `heroku-` prefix on a community buildpack URL → Scalingo's buildpacks are usually at `github.com/Scalingo/...-buildpack`; Heroku buildpacks mostly work but may reference Heroku-specific paths
