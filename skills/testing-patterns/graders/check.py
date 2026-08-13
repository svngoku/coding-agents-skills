#!/usr/bin/env python3
"""Deterministic grader for the testing-patterns "orders-test-suite" task.

Runs inside the agent's workspace (the current working directory). Statically
inspects test_orders.py - the pytest suite the agent wrote for the buggy
fixtures/orders.py module - using only the standard-library ast module. It
never executes the tests and never imports orders.py (no network, no pytest
dependency, fully hermetic).

Checks that the suite:
  - exists and imports the orders module
  - has >= 3 test_* functions with given-when-then style names
  - gives every test a real assertion (assert or pytest.raises)
  - uses @pytest.mark.parametrize for edge cases
  - defines a pytest fixture for shared setup and actually consumes it
  - references all three buggy API surfaces (add_item, apply_discount,
    average_unit_price) so every bug is pinned
  - has at least one assertion on a computed orders value (no snapshot-only,
    no assert True)
  - contains no time.sleep / sleep-based waiting

Output contract (JSON on stdout):
  {"score": 0.0-1.0, "details": "N/M checks passed", "checks": [...]}
Exit code 0 if score >= PASS_BAR (0.8), else 1.
"""

import ast
import json
import os
import sys

TEST_FILE = "test_orders.py"
PASS_BAR = 0.8

# Public surface of fixtures/orders.py.
ORDERS_API = {
    "orders",
    "Cart",
    "add_item",
    "apply_discount",
    "average_unit_price",
    "total",
    "item_count",
}

# The three intentional bugs live on these surfaces.
BUG_SURFACES = ("add_item", "apply_discount", "average_unit_price")

GWT_MARKERS = ("given", "when", "then")

ALL_CHECKS = (
    "test-file-exists",
    "python-syntax",
    "imports-orders",
    "at-least-3-tests",
    "given-when-then-naming",
    "assertions-in-every-test",
    "parametrize-used",
    "fixture-for-shared-setup",
    "covers-buggy-behaviors",
    "behavioral-assertions",
    "no-time-sleep",
)


def test_functions(tree):
    """FunctionDefs pytest would collect: module-level test_* and Test* methods."""
    tests = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.lower().startswith("test_"):
            tests.append(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name.lower().startswith("test_")
                ):
                    tests.append(item)
    return tests


def is_parametrize(dec):
    """True for @pytest.mark.parametrize / @mark.parametrize / @parametrize."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Attribute):
        return dec.attr == "parametrize"
    if isinstance(dec, ast.Name):
        return dec.id == "parametrize"
    return False


def is_fixture(dec):
    """True for @pytest.fixture / @pytest.fixture(...)."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Attribute):
        return dec.attr == "fixture"
    if isinstance(dec, ast.Name):
        return dec.id == "fixture"
    return False


def fixture_is_autouse(dec):
    if not isinstance(dec, ast.Call):
        return False
    for kw in dec.keywords:
        if kw.arg == "autouse":
            return isinstance(kw.value, ast.Constant) and bool(kw.value.value)
    return False


def has_assertion(func):
    """True if the body contains an assert or a pytest.raises call."""
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "raises":
                return True
            if isinstance(f, ast.Name) and f.id == "raises":
                return True
    return False


def call_mentions_orders(expr):
    """True if expr contains a call to an orders API name."""
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in ORDERS_API:
                return True
            if isinstance(f, ast.Attribute) and f.attr in ORDERS_API:
                return True
    return False


def referenced_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def imports_orders(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "orders" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "orders":
            return True
    return False


def has_sleep(tree):
    """True if any call looks like a sleep-based wait (time.sleep, sleep())."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "sleep":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "sleep":
                return True
    return False


def check(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


def emit(checks):
    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    score = (passed / total) if total else 0.0
    print(json.dumps({
        "score": round(score, 2),
        "details": f"{passed}/{total} checks passed",
        "checks": checks,
    }))
    sys.exit(0 if score >= PASS_BAR else 1)


def main():
    if not os.path.isfile(TEST_FILE):
        emit([check("test-file-exists", False, "test_orders.py not found")]
             + [check(n, False, "test_orders.py not found")
                for n in ALL_CHECKS[1:]])

    try:
        with open(TEST_FILE, encoding="utf-8") as fh:
            src = fh.read()
    except OSError as exc:
        emit([check("test-file-exists", False, f"cannot read test_orders.py: {exc}")]
             + [check(n, False, "cannot read test_orders.py")
                for n in ALL_CHECKS[1:]])

    try:
        tree = ast.parse(src, filename=TEST_FILE)
        parse_ok, parse_msg = True, "valid Python syntax"
    except SyntaxError as exc:
        tree = None
        parse_ok, parse_msg = False, f"syntax error: line {exc.lineno}: {exc.msg}"

    checks = [check("test-file-exists", True, "test_orders.py present"),
              check("python-syntax", parse_ok, parse_msg)]

    if tree is None:
        checks += [check(n, False, "cannot parse test_orders.py")
                   for n in ALL_CHECKS[2:]]
        emit(checks)

    tests = test_functions(tree)
    names = referenced_names(tree)

    # imports the orders module
    imports_ok = imports_orders(tree)
    checks.append(check(
        "imports-orders",
        imports_ok,
        "imports the orders module"
        if imports_ok
        else "no import of the orders module (import orders / from orders import ...)",
    ))

    # at least 3 test functions
    checks.append(check(
        "at-least-3-tests",
        len(tests) >= 3,
        f"found {len(tests)} test function(s)"
        if len(tests) >= 3
        else f"only {len(tests)} test function(s); need at least 3",
    ))

    # given-when-then style names (>= 2 names contain given/when/then)
    gwt = [t.name for t in tests
           if any(m in t.name.lower() for m in GWT_MARKERS)]
    checks.append(check(
        "given-when-then-naming",
        len(gwt) >= 2,
        f"{len(gwt)} test name(s) use given/when/then markers"
        if len(gwt) >= 2
        else "fewer than 2 test names use given-when-then style (given/when/then)",
    ))

    # every test has a real assertion (assert or pytest.raises)
    no_assert = [t.name for t in tests if not has_assertion(t)]
    checks.append(check(
        "assertions-in-every-test",
        not no_assert,
        "every test contains an assert or pytest.raises"
        if not no_assert
        else f"tests without assertions: {', '.join(no_assert)}",
    ))

    # parametrize marker used
    param_ok = any(is_parametrize(dec) for t in tests for dec in t.decorator_list)
    checks.append(check(
        "parametrize-used",
        param_ok,
        "@pytest.mark.parametrize found on a test"
        if param_ok
        else "no @pytest.mark.parametrize decorator found",
    ))

    # a pytest fixture is defined and actually consumed
    fixtures = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(is_fixture(d) for d in node.decorator_list)
    ]
    used = []
    for fx in fixtures:
        autouse = any(fixture_is_autouse(d) for d in fx.decorator_list)
        as_param = any(fx.name in {a.arg for a in t.args.args} for t in tests)
        called = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == fx.name
            for node in ast.walk(tree)
        )
        if autouse or as_param or called:
            used.append(fx.name)
    checks.append(check(
        "fixture-for-shared-setup",
        len(fixtures) > 0 and bool(used),
        f"pytest fixture(s) defined and used: {', '.join(used)}"
        if used
        else ("no pytest fixture defined" if not fixtures
              else "pytest fixture(s) defined but never used by a test"),
    ))

    # all three buggy API surfaces referenced (bugs pinned)
    covered = [s for s in BUG_SURFACES if s in names]
    checks.append(check(
        "covers-buggy-behaviors",
        len(covered) == len(BUG_SURFACES),
        f"references all buggy surfaces: {', '.join(BUG_SURFACES)}"
        if len(covered) == len(BUG_SURFACES)
        else "missing references to: "
             + ", ".join(s for s in BUG_SURFACES if s not in covered),
    ))

    # at least one assertion on a computed orders value (no snapshot-only)
    behavioral = []
    for t in tests:
        for node in ast.walk(t):
            if isinstance(node, ast.Assert) and call_mentions_orders(node.test):
                behavioral.append(t.name)
                break
    checks.append(check(
        "behavioral-assertions",
        bool(behavioral),
        f"assertions pin computed orders values (e.g. {', '.join(behavioral[:3])})"
        if behavioral
        else "no assertion compares a computed orders value (snapshot-only or assert True)",
    ))

    # no time.sleep / sleep-based waiting
    sleep_ok = not has_sleep(tree)
    checks.append(check(
        "no-time-sleep",
        sleep_ok,
        "no sleep-based waits found"
        if sleep_ok
        else "time.sleep / sleep() found - tests must be instant and deterministic",
    ))

    emit(checks)


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
