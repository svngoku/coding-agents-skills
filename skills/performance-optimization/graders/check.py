#!/usr/bin/env python3
"""Deterministic grader for the performance-optimization optimize-orders-endpoint task.

Runs inside the agent's workspace (cwd). Statically inspects the two required
artifacts:

  optimized.py   the fixed GET /orders endpoint (N+1 removed, batched or
                 eager-loaded queries, a TTL caching layer, no blocking sleeps)
  NOTES.md       the write-up (names the bottleneck, mentions EXPLAIN/index,
                 gives a before/after measurement)

Nothing is executed and nothing touches the network. Prints a skillgrade
result as JSON to stdout:

  {"score": 0.0-1.0, "details": "...", "checks": [{"name","passed","message"}, ...]}

Exit code 1 if the score is below the pass bar (0.8).
Python 3 standard library only.
"""

import ast
import glob
import json
import os
import re
import sys

OPTIMIZED = "optimized.py"
PASS_BAR = 0.8

# --- content signals -------------------------------------------------------

# A "DB query" call is a call to a helper whose name suggests a fetch/query/
# select/load, or a module-style attribute call (db.query, session.execute...).
DB_CALL_RE = re.compile(r"(fetch|query|select|find|load)", re.I)
_MODULE_ISH = {
    "db", "session", "conn", "connection", "cursor", "orm", "query",
    "models", "engine", "database", "repo", "repository",
}

# Eager loading / batching evidence (any common ORM or SQL form).
BATCH_RE = re.compile(
    r"\b(selectinload|eagerload|joinedload|subqueryload|select_related|prefetch_related|include)\b"
    r"|bulk"
    r"|batch"
    r"|\.in\s*\("
    r"|\bin\s*\("
    r"|\bin_\s*\(",
    re.I,
)

# Caching layer evidence (optimized.py or NOTES.md).
CACHE_RE = re.compile(r"\b(cache|lru_cache|ttl|redis|setex|memoiz)\w*\b", re.I)

# NOTES.md evidence.
BOTTLENECK_RE = re.compile(
    r"\bn\s*\+\s*1\b|n\s+plus\s+one|n-plus-one|one query per"
    r"|query per order|round trips? per|queries per request",
    re.I,
)
EXPLAIN_RE = re.compile(r"\b(explain|index(?:es|ed)?)\b", re.I)
MEASURE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|sec|%)"
    r"|\bp(?:50|95|99)\b"
    r"|\brps\b"
    r"|queries per request",
    re.I,
)
BEFORE_AFTER_RE = re.compile(
    r"\b(baseline|before|after|improved|reduced|decreased|dropped|speedup|faster)\b|→|->",
    re.I,
)

# Endpoint behavior preserved.
HANDLER_RE = re.compile(
    r"@(?:app\.)?(?:get|route)\b[^\n]*orders"
    r"|def\s+\w*order\w*\s*\("
    r"|get_orders_with_items",
    re.I,
)


def find_notes():
    """NOTES.md, case-insensitive (markdown casing is cosmetic)."""
    for hit in sorted(glob.glob("*[nN][oO][tT][eE][sS].md")):
        if os.path.isfile(hit):
            return hit
    return None


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


# --- AST helpers -----------------------------------------------------------

def _db_callee(call):
    """Resolve a call's callee name, or None for methods on local objects
    (e.g. items_by_order.get(...)) which are not DB queries."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            if func.value.id in _MODULE_ISH:
                return func.attr
            return None  # method on a local variable/dict - not a DB call
        return func.attr
    return None


def _target_names(target):
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _loop_scopes(tree):
    """Yield (target_names, body_nodes) for every for/while/comprehension."""
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            yield _target_names(node.target), node.body
        elif isinstance(node, ast.While):
            yield set(), node.body
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            for gen in node.generators:
                yield _target_names(gen.target), [node.elt]
        elif isinstance(node, ast.DictComp):
            for gen in node.generators:
                yield _target_names(gen.target), [node.key, node.value]


def _n_plus_one(tree):
    """True if a DB-lookup call sits inside a loop and receives the loop row
    (e.g. fetch_items(order["id"]) inside 'for order in orders')."""
    for targets, bodies in _loop_scopes(tree):
        if not targets:
            continue
        for body in bodies:
            for node in ast.walk(body):
                if not isinstance(node, ast.Call):
                    continue
                name = _db_callee(node)
                if name is None or not DB_CALL_RE.search(name):
                    continue
                args = list(node.args) + [
                    k.value for k in node.keywords if k.value is not None
                ]
                for arg in args:
                    arg_names = {n.id for n in ast.walk(arg)
                                 if isinstance(n, ast.Name)}
                    if arg_names & targets:
                        return True
    return False


def _blocking_sleep(tree):
    """True if optimized.py contains a synchronous blocking sleep
    (time.sleep, from-time-import sleep); asyncio.sleep is allowed."""
    from_asyncio_sleep = any(
        isinstance(n, ast.ImportFrom) and n.module == "asyncio"
        and any(a.name == "sleep" for a in n.names)
        for n in ast.walk(tree)
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "sleep":
            if isinstance(func.value, ast.Name) and func.value.id in (
                "asyncio", "loop", "event_loop",
            ):
                continue  # asyncio.sleep is non-blocking
            return True
        if isinstance(func, ast.Name) and func.id == "sleep":
            if not from_asyncio_sleep:
                return True
    return False


# --- checks ----------------------------------------------------------------

def main():
    checks = []

    def add(name, passed, message):
        checks.append({"name": name, "passed": bool(passed), "message": message})

    notes_path = find_notes()
    notes_src = read_file(notes_path) if notes_path else ""

    opt_src = read_file(OPTIMIZED)
    if opt_src is None:
        add("optimized-py-exists", False,
            "optimized.py not found in the workspace")
    else:
        add("optimized-py-exists", True, "optimized.py found")

    tree = None
    if opt_src is not None:
        try:
            tree = ast.parse(opt_src, filename=OPTIMIZED)
            add("optimized-py-syntax", True, "optimized.py parses as valid Python")
        except SyntaxError as exc:
            add("optimized-py-syntax", False,
                f"syntax error: line {exc.lineno}: {exc.msg}")

    if tree is None:
        for name, msg in (
            ("n-plus-one-removed", "cannot analyze optimized.py"),
            ("batched-or-eager-loading", "cannot analyze optimized.py"),
            ("caching-layer", "cannot analyze optimized.py"),
            ("no-blocking-sleep", "cannot analyze optimized.py"),
            ("handler-preserved", "cannot analyze optimized.py"),
        ):
            add(name, False, msg)
    else:
        n1 = _n_plus_one(tree)
        add("n-plus-one-removed", not n1,
            "no DB lookup call inside a loop (N+1 removed)"
            if not n1 else "DB lookup call found inside a loop (N+1 pattern)")

        batched = bool(BATCH_RE.search(opt_src))
        add("batched-or-eager-loading", batched,
            "batched or eager-loaded query present (IN list / selectinload / join / bulk)"
            if batched else "no batched or eager-loading query found (IN ..., selectinload, join, bulk)")

        cache_hit = bool(CACHE_RE.search(opt_src)) or bool(CACHE_RE.search(notes_src))
        add("caching-layer", cache_hit,
            "caching layer found (cache / TTL / Redis) in optimized.py or NOTES.md"
            if cache_hit else "no caching layer (cache/TTL/Redis) in optimized.py or NOTES.md")

        sleep = _blocking_sleep(tree)
        add("no-blocking-sleep", not sleep,
            "no synchronous blocking sleep in optimized.py"
            if not sleep else "synchronous blocking sleep (time.sleep) found in optimized.py")

        handler = bool(HANDLER_RE.search(opt_src))
        add("handler-preserved", handler,
            "GET /orders handler or orders function preserved"
            if handler else "no GET /orders handler or orders function found in optimized.py")

    if notes_path is None:
        add("notes-exists", False, "NOTES.md not found in the workspace")
        for name, msg in (
            ("notes-names-bottleneck", "NOTES.md missing"),
            ("notes-explain-or-index", "NOTES.md missing"),
            ("notes-measurement", "NOTES.md missing"),
            ("notes-before-after", "NOTES.md missing"),
        ):
            add(name, False, msg)
    else:
        add("notes-exists", True, f"found {notes_path}")

        bottleneck = bool(BOTTLENECK_RE.search(notes_src))
        add("notes-names-bottleneck", bottleneck,
            "NOTES.md names the original bottleneck (N+1 / per-order queries)"
            if bottleneck else "NOTES.md does not name the original bottleneck")

        explain = bool(EXPLAIN_RE.search(notes_src))
        add("notes-explain-or-index", explain,
            "NOTES.md mentions EXPLAIN or indexes"
            if explain else "NOTES.md does not mention EXPLAIN or indexes")

        measured = bool(MEASURE_RE.search(notes_src))
        add("notes-measurement", measured,
            "NOTES.md includes a quantitative measurement (ms / p95 / queries per request)"
            if measured else "NOTES.md has no quantitative measurement (latency / p95 / ms)")

        compared = bool(BEFORE_AFTER_RE.search(notes_src))
        add("notes-before-after", compared,
            "NOTES.md compares before and after (baseline → improved)"
            if compared else "NOTES.md lacks a before/after comparison")

    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    score = (passed / total) if total else 0.0

    print(json.dumps({
        "score": round(score, 3),
        "details": f"{passed}/{total} checks passed",
        "checks": checks,
    }))
    sys.exit(0 if score >= PASS_BAR else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # never crash the harness - emit a zero-score JSON
        print(json.dumps({
            "score": 0.0,
            "details": f"grader error: {exc}",
            "checks": [],
        }))
        sys.exit(1)
