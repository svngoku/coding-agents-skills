#!/usr/bin/env python3
"""Deterministic grader for the scalingo "scalingo-deploy" eval task.

Runs from the agent's workspace (cwd). Checks the three required artifacts:

  scalingo.json   app manifest: name, formation.web, env, postgresql addon
  Procfile        process definition with a web: line binding $PORT
  deploy.sh       bash deploy script: region, create, env-set, git remote
                  add + push main, addon provisioning, scale, post-deploy
                  verification (logs or run)

Python 3 standard library ONLY (no PyYAML, no network). JSON is parsed with
the stdlib json module; bash syntax is checked with `bash -n` when available.
Checks are lenient on cosmetics (key order, quoting, naming variations) but
strict on the skill's core API surface: the manifest must actually declare the
web formation, env, and a valid postgresql provider:plan addon; the Procfile
must bind $PORT; deploy.sh must be region-aware and cover the full deploy
lifecycle.

Prints JSON to stdout:
  {"score": 0.93, "details": "17/19 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import json
import os
import re
import subprocess
import sys

MANIFEST = "scalingo.json"
PROCFILE = "Procfile"
DEPLOY = "deploy.sh"
PASS_BAR = 0.8

POSTGRESQL_PLAN = re.compile(r"^postgresql:postgresql-[a-z0-9-]+$")


def _add(checks, name, passed, msg):
    checks.append({"name": name, "passed": bool(passed), "message": msg})


def _read(path):
    """Read a UTF-8 text file (BOM-tolerant). Returns text or None on OSError."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.read()
    except OSError:
        return None


def _deploy_lines(text):
    """Non-empty, non-comment lines (comments start with # after whitespace)."""
    return [raw for raw in text.splitlines()
            if raw.strip() and not raw.strip().startswith("#")]


# ---------------------------------------------------------------------------
# scalingo.json
# ---------------------------------------------------------------------------

def _check_manifest(checks):
    if not os.path.isfile(MANIFEST):
        _add(checks, "manifest-exists", False, "scalingo.json not found")
        _add(checks, "manifest-valid-json", False, "scalingo.json not found")
        _add(checks, "manifest-name", False, "scalingo.json not found")
        _add(checks, "manifest-formation-web", False, "scalingo.json not found")
        _add(checks, "manifest-env", False, "scalingo.json not found")
        _add(checks, "manifest-addon-postgresql", False, "scalingo.json not found")
        return

    text = _read(MANIFEST)
    if text is None:
        _add(checks, "manifest-exists", True, "scalingo.json present")
        for name in ("manifest-valid-json", "manifest-name",
                     "manifest-formation-web", "manifest-env",
                     "manifest-addon-postgresql"):
            _add(checks, name, False, f"cannot read {MANIFEST}")
        return

    _add(checks, "manifest-exists", True, "scalingo.json present")

    try:
        data = json.loads(text)
    except ValueError as e:
        _add(checks, "manifest-valid-json", False, f"invalid JSON: {e}")
        for name in ("manifest-name", "manifest-formation-web",
                     "manifest-env", "manifest-addon-postgresql"):
            _add(checks, name, False, "scalingo.json did not parse")
        return
    if not isinstance(data, dict):
        _add(checks, "manifest-valid-json", False, "scalingo.json is not a JSON object")
        for name in ("manifest-name", "manifest-formation-web",
                     "manifest-env", "manifest-addon-postgresql"):
            _add(checks, name, False, "scalingo.json is not a JSON object")
        return

    _add(checks, "manifest-valid-json", True, "valid JSON object")

    name = data.get("name")
    name_ok = isinstance(name, str) and name.strip() != ""
    _add(checks, "manifest-name", name_ok,
         f"name: {name!r}" if name_ok else "missing or empty 'name' field")

    formation = data.get("formation")
    web = formation.get("web") if isinstance(formation, dict) else None
    web_ok = False
    if isinstance(web, dict):
        amount = web.get("amount")
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and amount >= 1:
            web_ok = True
        elif isinstance(amount, str) and amount.strip().isdigit() and int(amount) >= 1:
            web_ok = True
    _add(checks, "manifest-formation-web", web_ok,
         f"formation.web: {web}" if web is not None
         else "formation missing or has no 'web' process")

    env = data.get("env")
    env_ok = isinstance(env, dict) and len(env) > 0
    _add(checks, "manifest-env", env_ok,
         f"env: {sorted(env)}" if env_ok else "env missing or empty")

    addons = data.get("addons")
    addon_ok = False
    addon_msg = "addons missing"
    if isinstance(addons, list) and addons:
        plans = []
        for entry in addons:
            if isinstance(entry, str):
                plans.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("plan"), str):
                plans.append(entry["plan"])
        good = [p for p in plans if POSTGRESQL_PLAN.match(p)]
        if good:
            addon_ok = True
            addon_msg = f"postgresql addon plan: {good[0]}"
        else:
            addon_msg = (
                "no valid postgresql plan (expected provider:plan form, e.g. "
                f"'postgresql:postgresql-starter-512'); got {plans}"
            )
    _add(checks, "manifest-addon-postgresql", addon_ok, addon_msg)


# ---------------------------------------------------------------------------
# Procfile
# ---------------------------------------------------------------------------

def _check_procfile(checks):
    if not os.path.isfile(PROCFILE):
        _add(checks, "procfile-exists", False, "Procfile not found")
        _add(checks, "procfile-web", False, "Procfile not found")
        _add(checks, "procfile-port", False, "Procfile not found")
        return

    text = _read(PROCFILE)
    if text is None:
        _add(checks, "procfile-exists", True, "Procfile present")
        _add(checks, "procfile-web", False, f"cannot read {PROCFILE}")
        _add(checks, "procfile-port", False, f"cannot read {PROCFILE}")
        return

    _add(checks, "procfile-exists", True, "Procfile present")

    web_cmd = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^web\s*:\s*(.+)$", line)
        if m:
            web_cmd = m.group(1).strip()
            break
    _add(checks, "procfile-web", bool(web_cmd),
         f"web: {web_cmd}" if web_cmd else "no 'web:' line with a command in Procfile")

    if web_cmd is None:
        _add(checks, "procfile-port", False, "no web: line to inspect")
        return
    port_ok = "$PORT" in web_cmd or "${PORT}" in web_cmd
    _add(checks, "procfile-port", port_ok,
         "web command binds $PORT" if port_ok
         else "web command does not reference $PORT (Scalingo routes traffic on the injected port)")


# ---------------------------------------------------------------------------
# deploy.sh
# ---------------------------------------------------------------------------

def _check_deploy(checks):
    if not os.path.isfile(DEPLOY):
        for name in ("deploy-exists", "deploy-bash-syntax", "deploy-create",
                     "deploy-region", "deploy-env-set", "deploy-remote-add",
                     "deploy-git-push", "deploy-scale", "deploy-addon-add",
                     "deploy-post-check"):
            _add(checks, name, False, "deploy.sh not found")
        return

    text = _read(DEPLOY)
    if text is None:
        _add(checks, "deploy-exists", True, "deploy.sh present")
        for name in ("deploy-bash-syntax", "deploy-create", "deploy-region",
                     "deploy-env-set", "deploy-remote-add", "deploy-git-push",
                     "deploy-scale", "deploy-addon-add", "deploy-post-check"):
            _add(checks, name, False, f"cannot read {DEPLOY}")
        return

    _add(checks, "deploy-exists", True, "deploy.sh present")

    syntax_ok = True
    syntax_msg = "valid bash syntax"
    try:
        r = subprocess.run(["bash", "-n", DEPLOY], capture_output=True, text=True)
        if r.returncode != 0:
            syntax_ok = False
            syntax_msg = f"bash -n: {(r.stderr or r.stdout).strip()}"
    except FileNotFoundError:
        syntax_msg = "bash not found on grader host; syntax check skipped"
    _add(checks, "deploy-bash-syntax", syntax_ok, syntax_msg)

    lines = _deploy_lines(text)
    blob = "\n".join(lines)

    create = re.search(r"scalingo[^\n]*\bcreate\b", blob)
    _add(checks, "deploy-create", bool(create),
         f"calls 'scalingo create': {create.group(0).strip()}" if create
         else "missing 'scalingo create' (scalingo ... create <app>)")

    region = re.search(r"scalingo[^\n]*--region\s+\S+", blob) or re.search(
        r"SCALINGO_REGION\s*=\s*[A-Za-z0-9_.-]+", blob)
    _add(checks, "deploy-region", bool(region),
         f"region mechanism: {region.group(0).strip()}" if region
         else "no region mechanism (expected 'scalingo ... --region <region>' or 'scalingo config --region')")

    _add(checks, "deploy-env-set", "env-set" in blob,
         "uses scalingo env-set" if "env-set" in blob else "missing 'env-set'")

    remote = any("git remote add" in line and "scalingo" in line for line in lines)
    _add(checks, "deploy-remote-add", remote,
         "git remote add ... scalingo present" if remote
         else "missing 'git remote add' pointing at the scalingo remote")

    push = next((line.strip() for line in lines
                 if "git push" in line and "scalingo" in line
                 and re.search(r"\b(main|master)\b", line)), None)
    _add(checks, "deploy-git-push", bool(push),
         push or "no 'git push scalingo main' (or local-branch:main)")

    scale = re.search(r"scalingo[^\n]*\bscale\b[^\n]*", blob)
    _add(checks, "deploy-scale", bool(scale),
         scale.group(0).strip() if scale else "missing 'scalingo ... scale'")

    addon = re.search(r"scalingo[^\n]*addons-add[^\n]*", blob)
    _add(checks, "deploy-addon-add", bool(addon),
         addon.group(0).strip() if addon else "missing 'scalingo ... addons-add'")

    post = re.search(r"scalingo[^\n]*\b(logs|run)\b[^\n]*", blob)
    _add(checks, "deploy-post-check", bool(post),
         post.group(0).strip() if post else "no post-deploy check ('scalingo ... logs' or 'scalingo ... run')")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checks = []
    _check_manifest(checks)
    _check_procfile(checks)
    _check_deploy(checks)

    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    score = (passed / total) if total else 0.0

    print(json.dumps({
        "score": round(score, 4),
        "details": f"{passed}/{total} checks passed",
        "checks": checks,
    }))
    sys.exit(0 if score >= PASS_BAR else 1)


if __name__ == "__main__":
    main()
