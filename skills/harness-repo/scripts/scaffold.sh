#!/usr/bin/env bash
# Deterministic bootstrap of the harness-engineering starter.
# Usage: scaffold.sh --name my-api [--stack node|python|go|ruby|generic]
#        [--desc "one line"] [--path /parent/dir] [--overwrite] [--dry-run] [--commit]
# Policy per file: CREATE if absent, SKIP if identical, else KEEP and write <file>.harness-new
# (with --overwrite: back up to <file>.bak, then overwrite). Never touches files it did not render.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$SKILL_DIR/assets"
NAME=""; STACK=""; DESC=""; PARENT="."; OVERWRITE=0; DRY=0; COMMIT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --stack) STACK="$2"; shift 2;;
    --desc) DESC="$2"; shift 2;;
    --path) PARENT="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    --dry-run) DRY=1; shift;;
    --commit) COMMIT=1; shift;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$NAME" ] || { echo "--name is required" >&2; exit 2; }
case "$NAME" in *[!a-zA-Z0-9._-]*|"") echo "name must be a slug [a-zA-Z0-9._-]" >&2; exit 2;; esac

TARGET="$PARENT/$NAME"
[ "$DRY" = 1 ] || mkdir -p "$TARGET"
TARGET_ABS="$(cd "$TARGET" 2>/dev/null && pwd || echo "$TARGET")"

# --- stack resolution -------------------------------------------------------
if [ -z "$STACK" ]; then
  if   [ -f "$TARGET/package.json" ];   then STACK=node
  elif [ -f "$TARGET/pyproject.toml" ]; then STACK=python
  elif [ -f "$TARGET/go.mod" ];         then STACK=go
  elif [ -f "$TARGET/Gemfile" ];        then STACK=ruby
  else STACK=generic; fi
fi
case "$STACK" in
  node|typescript|ts|javascript|js|fullstack) STACK=node;;
  python|py) STACK=python;; go|golang) STACK=go;; ruby|rb) STACK=ruby;;
  *) STACK=generic;;
esac
case "$STACK" in
  node)    INSTALL="npm ci"; DEV="npm run dev"; BUILD="npm run build"; TEST="npm test"; LINT="npm run lint"; FORMAT="npm run format"; CHECK="npm run check"; ECO=npm;;
  python)  INSTALL="uv sync"; DEV="uv run python -m ${NAME//-/_}"; BUILD="uv build"; TEST="uv run pytest"; LINT="uv run ruff check ."; FORMAT="uv run ruff format ."; CHECK="uv run ruff check . && uv run mypy . && uv run pytest"; ECO=uv;;
  go)      INSTALL="go mod download"; DEV="go run ."; BUILD="go build ./..."; TEST="go test ./..."; LINT="go vet ./..."; FORMAT="gofmt -w ."; CHECK="make check"; ECO=gomod;;
  ruby)    INSTALL="bundle install"; DEV="bin/dev"; BUILD="bundle exec rake build"; TEST="bundle exec rspec"; LINT="bundle exec rubocop"; FORMAT="bundle exec rubocop -A"; CHECK="bundle exec rubocop && bundle exec rspec"; ECO=bundler;;
  generic) INSTALL="make install"; DEV="make dev"; BUILD="make build"; TEST="make test"; LINT="make lint"; FORMAT="make format"; CHECK="make check"; ECO="";;
esac
[ -n "$DESC" ] || DESC="TODO: one-sentence product purpose."
TODAY="$(date +%Y-%m-%d)"

esc() { printf '%s' "$1" | sed -e 's/[\\|&]/\\&/g'; }
render() { # stdin -> stdout
  sed -e "s|{{PROJECT_NAME}}|$(esc "$NAME")|g" -e "s|{{DESCRIPTION}}|$(esc "$DESC")|g" \
      -e "s|{{STACK}}|$(esc "$STACK")|g" -e "s|{{INSTALL}}|$(esc "$INSTALL")|g" \
      -e "s|{{DEV}}|$(esc "$DEV")|g" -e "s|{{BUILD}}|$(esc "$BUILD")|g" \
      -e "s|{{TEST}}|$(esc "$TEST")|g" -e "s|{{LINT}}|$(esc "$LINT")|g" \
      -e "s|{{FORMAT}}|$(esc "$FORMAT")|g" -e "s|{{CHECK}}|$(esc "$CHECK")|g" \
      -e "s|__DATE__|$TODAY|g"
}

TOUCHED=(); C_NEW=0; C_SKIP=0; C_KEEP=0; C_OVER=0
# place <rendered-tmpfile> <relative-dest> [mode]
place() {
  local tmp="$1" rel="$2" mode="${3:-}" dest="$TARGET/$2"
  if [ ! -e "$dest" ]; then
    echo "CREATE  $rel"; C_NEW=$((C_NEW+1))
    [ "$DRY" = 1 ] || { mkdir -p "$(dirname "$dest")"; cp "$tmp" "$dest"; [ -z "$mode" ] || chmod "$mode" "$dest"; TOUCHED+=("$rel"); }
  elif cmp -s "$tmp" "$dest"; then
    echo "SKIP    $rel"; C_SKIP=$((C_SKIP+1))
  elif [ "$OVERWRITE" = 1 ]; then
    echo "OVERWRT $rel (backup: $rel.bak)"; C_OVER=$((C_OVER+1))
    [ "$DRY" = 1 ] || { cp "$dest" "$dest.bak"; cp "$tmp" "$dest"; [ -z "$mode" ] || chmod "$mode" "$dest"; TOUCHED+=("$rel"); }
  else
    echo "KEEP    $rel (differs; proposed: $rel.harness-new)"; C_KEEP=$((C_KEEP+1))
    [ "$DRY" = 1 ] || cp "$tmp" "$dest.harness-new"
  fi
}

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# --- 1. git ---------------------------------------------------------------
if [ "$DRY" = 0 ] && ! git -C "$TARGET" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$TARGET" init -q -b main 2>/dev/null || { git -C "$TARGET" init -q; git -C "$TARGET" checkout -q -b main 2>/dev/null || true; }
  echo "GIT     initialised"
fi

# --- 2. mirror assets/repo/** (strip .tmpl, render) -------------------------
while IFS= read -r f; do
  rel="${f#"$ASSETS/repo/"}"; rel="${rel%.tmpl}"; rel="${rel//__DATE__/$TODAY}"
  render < "$f" > "$WORK/r"
  mode=""; [ -x "$f" ] && mode="755"
  place "$WORK/r" "$rel" "$mode"
done < <(find "$ASSETS/repo" -type f | LC_ALL=C sort)

# --- 3. stack-specific CI ---------------------------------------------------
render < "$ASSETS/ci/ci.$STACK.yml" > "$WORK/r"; place "$WORK/r" ".github/workflows/ci.yml"

# --- 4. dependabot ----------------------------------------------------------
{
  echo "version: 2"; echo "updates:"
  if [ -n "$ECO" ]; then
    printf '  - package-ecosystem: "%s"\n    directory: "/"\n    schedule:\n      interval: "weekly"\n    open-pull-requests-limit: 5\n' "$ECO"
  fi
  printf '  - package-ecosystem: "github-actions"\n    directory: "/"\n    schedule:\n      interval: "weekly"\n'
} > "$WORK/r"; place "$WORK/r" ".github/dependabot.yml"

# --- 5. .env.example --------------------------------------------------------
cat > "$WORK/r" <<EOF
# ${NAME}: environment variables. Copy to .env and fill in real values.
# NEVER commit .env. Use fake placeholders here, never real secrets.

# Application
APP_ENV=development   # development | test | production
PORT=3000

# DATABASE_URL=postgresql://user:password@localhost:5432/${NAME//-/_}_dev
EOF
place "$WORK/r" ".env.example"

# --- 6. Makefile (go + generic only) ---------------------------------------
if [ "$STACK" = go ] || [ "$STACK" = generic ]; then
  if [ "$STACK" = go ]; then
    printf '.PHONY: install dev build test lint format check\ninstall:\n\tgo mod download\ndev:\n\tgo run .\nbuild:\n\tgo build ./...\ntest:\n\tgo test ./...\nlint:\n\tgo vet ./...\nformat:\n\tgofmt -w .\ncheck: lint test\n' > "$WORK/r"
  else
    printf '.PHONY: install dev build test lint format check\ninstall: ## Install dependencies\n\t@echo "TODO: install"\ndev: ## Start dev server\n\t@echo "TODO: dev"\nbuild: ## Build\n\t@echo "TODO: build"\ntest: ## Run tests\n\t@echo "TODO: test"\nlint: ## Lint\n\t@echo "TODO: lint"\nformat: ## Format\n\t@echo "TODO: format"\ncheck: lint test ## Full gate\n' > "$WORK/r"
  fi
  place "$WORK/r" "Makefile"
fi

# --- 7. .gitignore: merge missing entries ----------------------------------
base='.env
.env.local
.env.*.local
!.env.example
*.log
*.tmp
.DS_Store
Thumbs.db
coverage/
.coverage
htmlcov/
.opencode/cache/
*.harness-new
*.bak'
case "$STACK" in
  node)    extra='node_modules/
dist/
build/
.next/
.output/';;
  python)  extra='__pycache__/
*.pyc
.venv/
dist/
build/
.mypy_cache/
.ruff_cache/
.pytest_cache/';;
  go)      extra='bin/
vendor/';;
  ruby)    extra='vendor/bundle/
.bundle/';;
  *)       extra='dist/
build/';;
esac
GI="$TARGET/.gitignore"; added=0
if [ "$DRY" = 0 ]; then touch "$GI"; fi
while IFS= read -r line; do
  [ -n "$line" ] || continue
  if [ ! -f "$GI" ] || ! grep -qxF -- "$line" "$GI"; then
    added=$((added+1)); [ "$DRY" = 1 ] || printf '%s\n' "$line" >> "$GI"
  fi
done <<< "$base"$'\n'"$extra"
if [ "$added" -gt 0 ]; then echo "MERGE   .gitignore (+$added entries)"; TOUCHED+=(".gitignore"); else echo "SKIP    .gitignore"; fi

# --- summary / optional commit ---------------------------------------------
echo "----"
echo "stack=$STACK path=$TARGET_ABS | created=$C_NEW skipped=$C_SKIP kept=$C_KEEP overwritten=$C_OVER"
[ "$C_KEEP" = 0 ] || echo "NOTE: $C_KEEP file(s) differ from the starter; review the *.harness-new proposals."

if [ "$COMMIT" = 1 ] && [ "$DRY" = 0 ] && [ "${#TOUCHED[@]}" -gt 0 ]; then
  git -C "$TARGET" add -- "${TOUCHED[@]}"
  git -C "$TARGET" commit -q -m "chore: bootstrap harness engineering structure" \
    -m "Humans steer. Agents execute. Starter generated by the harness-repo skill (stack: $STACK)."
  echo "COMMIT  $(git -C "$TARGET" rev-parse --short HEAD) (${#TOUCHED[@]} files staged, nothing else)"
fi
