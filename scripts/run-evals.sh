#!/usr/bin/env bash
#
# run-evals.sh — run skillgrade evaluations for this repo, locally (no CI).
#
# Usage:
#   ./scripts/run-evals.sh                      # smoke-test every skill that has an eval.yaml
#   ./scripts/run-evals.sh langchain            # smoke-test one skill
#   ./scripts/run-evals.sh --mode=reliable      # 15 trials (default: smoke / 5)
#   ./scripts/run-evals.sh --mode=regression    # 30 trials
#   ./scripts/run-evals.sh --validate           # verify graders against reference solutions
#   ./scripts/run-evals.sh langchain --mode=reliable --validate
#
# Requires: skillgrade (npm i -g skillgrade) and an agent. The agent comes from
# eval.yaml defaults (opencode) or --agent=gemini|claude|codex|acp overrides;
# set the matching API key env var (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/.evals"

SKILL=""
MODE="smoke"
VALIDATE=0

for arg in "$@"; do
  case "$arg" in
    --mode=*) MODE="${arg#--mode=}" ;;
    --validate) VALIDATE=1 ;;
    -h|--help) sed -n "2,15p" "${BASH_SOURCE[0]}"; exit 0 ;;
    *) SKILL="$arg" ;;
  esac
done

case "$MODE" in
  smoke|reliable|regression) ;;
  *) echo "unknown mode: $MODE (use smoke|reliable|regression)" >&2; exit 1 ;;
esac

command -v skillgrade >/dev/null 2>&1 || {
  echo "skillgrade not found — install it with: npm i -g skillgrade" >&2
  exit 1
}

if [ -n "$SKILL" ]; then
  DIR="$REPO_ROOT/skills/$SKILL"
  [ -f "$DIR/eval.yaml" ] || { echo "no eval.yaml in $DIR" >&2; exit 1; }
  SKILL_DIRS=("$DIR")
else
  mapfile -t SKILL_DIRS < <(find "$REPO_ROOT/skills" -name eval.yaml -not -path "*/node_modules/*" -exec dirname {} \; | sort)
  [ "${#SKILL_DIRS[@]}" -gt 0 ] || { echo "no skill has an eval.yaml yet"; exit 0; }
fi

mkdir -p "$OUT_DIR"
FAILED=0

for dir in "${SKILL_DIRS[@]}"; do
  name="$(basename "$dir")"
  args=(--"$MODE" --ci --provider=local --output="$OUT_DIR/$name")
  [ "$VALIDATE" -eq 1 ] && args+=(--validate)
  echo "==> $name ($MODE, provider=local)"
  if (cd "$dir" && skillgrade "${args[@]}"); then
    echo "    ✓ $name passed"
  else
    echo "    ✗ $name FAILED (see $OUT_DIR/$name)" >&2
    FAILED=1
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo "Some evaluations failed." >&2
  exit 1
fi
echo "All evaluations passed. Reports: $OUT_DIR/"
