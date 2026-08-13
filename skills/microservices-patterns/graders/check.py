#!/usr/bin/env python3
"""Deterministic grader for the microservices-patterns "checkout-choreographed-saga" task.

Runs inside the agent's workspace (the current working directory). Statically
inspects the two required artifacts:

  checkout-design.md   the saga design document (service boundaries, mermaid
                       choreography diagram, compensation, the outbox pattern,
                       idempotent consumers)
  events.py            Python dataclasses for the domain events

Python 3 standard library ONLY (no network). The markdown is checked with re;
the Python file is checked with the ast module. Class names are normalized by
stripping underscores so OrderPlaced / Order_Placed / order_placed all count.

Output contract (printed to stdout):
  {"score": 0.0-1.0, "details": "...", "checks": [{"name", "passed", "message"}, ...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import ast
import json
import re
import sys

DESIGN_FILE = "checkout-design.md"
EVENTS_FILE = "events.py"
PASS_BAR = 0.8

EVENT_SUFFIXES = ("Placed", "Succeeded", "Reserved", "Failed")
SERVICE_STEMS = ("order", "payment", "inventory", "notification")


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def check(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


# ---------------------------------------------------------------------------
# events.py checks
# ---------------------------------------------------------------------------

def _decorator_names(node):
    """Names of a class's decorators, resolving Call/Attribute wrappers."""
    names = []
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
        elif isinstance(d, ast.Call):
            f = d.func
            if isinstance(f, ast.Name):
                names.append(f.id)
            elif isinstance(f, ast.Attribute):
                names.append(f.attr)
    return names


def _event_classes(tree):
    """Return [(class_name, decorator_names), ...] for event classes.

    An event class is one whose name (underscores stripped) ends in
    Placed/Succeeded/Reserved/Failed.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        norm = node.name.replace("_", "").lower()
        if not any(norm.endswith(s.lower()) for s in EVENT_SUFFIXES):
            continue
        found.append((node.name, _decorator_names(node)))
    return found


def _check_events(checks):
    def add(name, passed, msg):
        checks.append(check(name, passed, msg))

    text = read_file(EVENTS_FILE)
    if text is None:
        for name in ("events-parses", "order-placed-event", "at-least-3-events",
                     "failure-event", "dataclass-decorated"):
            add(name, False, "events.py not found")
        return

    try:
        tree = ast.parse(text, filename=EVENTS_FILE)
    except SyntaxError as e:
        for name in ("events-parses", "order-placed-event", "at-least-3-events",
                     "failure-event", "dataclass-decorated"):
            add(name, False, f"events.py has a syntax error (line {e.lineno}: {e.msg})")
        return

    add("events-parses", True, "events.py parses as valid Python")

    events = _event_classes(tree)
    names = [n for n, _ in events]
    norm_names = [n.replace("_", "").lower() for n in names]

    has_order_placed = "orderplaced" in norm_names
    add(
        "order-placed-event",
        has_order_placed,
        "OrderPlaced event class found"
        if has_order_placed else f"no OrderPlaced class (found: {', '.join(names) or 'none'})",
    )

    at_least_3 = len(events) >= 3
    if at_least_3:
        msg = f"{len(events)} event class(es) named *Placed/*Succeeded/*Reserved/*Failed"
    else:
        msg = (
            f"only {len(events)} event class(es) named "
            "*Placed/*Succeeded/*Reserved/*Failed; need >= 3 (OrderPlaced + at least 2 more)"
        )
    add("at-least-3-events", at_least_3, msg)

    has_failure = any(n.endswith("failed") for n in norm_names)
    add(
        "failure-event",
        has_failure,
        "failure event(s) present (e.g. PaymentFailed / InventoryFailed)"
        if has_failure else "no *Failed event class - the saga needs explicit failure events",
    )

    dataclass_ok = sum(1 for _, decos in events if "dataclass" in decos) >= 2
    add(
        "dataclass-decorated",
        dataclass_ok,
        "at least 2 event classes are @dataclass decorated"
        if dataclass_ok else "fewer than 2 event classes are decorated with @dataclass",
    )


# ---------------------------------------------------------------------------
# checkout-design.md checks
# ---------------------------------------------------------------------------

def _mermaid_blocks(md):
    """Contents of every mermaid fenced block (triple-backtick)."""
    pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    return [m.group(1) for m in pattern.finditer(md)]


def _check_design(checks):
    def add(name, passed, msg):
        checks.append(check(name, passed, msg))

    md = read_file(DESIGN_FILE)
    if md is None:
        for name in ("mermaid-diagram", "saga-mentioned", "compensation-mentioned",
                     "outbox-mentioned", "idempotent-mentioned", "services-named"):
            add(name, False, "checkout-design.md not found")
        return

    blocks = _mermaid_blocks(md)
    diagram_ok = any(
        re.search(r"\b(?:sequenceDiagram|flowchart|graph)\b", b, re.I)
        for b in blocks
    )
    add(
        "mermaid-diagram",
        diagram_ok,
        "mermaid block with sequenceDiagram/flowchart found"
        if diagram_ok
        else "no mermaid block containing sequenceDiagram or flowchart",
    )

    saga_ok = bool(re.search(r"\bsaga", md, re.I))
    add("saga-mentioned", saga_ok,
        "design mentions the saga pattern" if saga_ok else "design never mentions the saga pattern")

    comp_ok = bool(re.search(r"\bcompensat", md, re.I))
    add("compensation-mentioned", comp_ok,
        "design mentions compensation / compensating actions"
        if comp_ok else "design never mentions compensation / compensating actions")

    outbox_ok = bool(re.search(r"\boutbox\b", md, re.I))
    add("outbox-mentioned", outbox_ok,
        "design mentions the outbox pattern" if outbox_ok else "design never mentions the outbox pattern")

    idem_ok = bool(re.search(r"\bidempoten", md, re.I))
    add("idempotent-mentioned", idem_ok,
        "design mentions idempotent consumers / idempotency"
        if idem_ok else "design never mentions idempotent consumers / idempotency")

    named = [s for s in SERVICE_STEMS if re.search(r"\b" + s + r"\w*\b", md, re.I)]
    services_ok = len(named) >= 3
    add(
        "services-named",
        services_ok,
        f"{len(named)}/4 services named ({', '.join(named)})"
        if services_ok
        else f"only {len(named)}/4 services named: {', '.join(named) or 'none'} (need orders, payment, inventory, notification)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checks = []
    _check_events(checks)
    _check_design(checks)

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
