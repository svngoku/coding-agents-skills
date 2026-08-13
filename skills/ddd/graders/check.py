#!/usr/bin/env python3
"""Deterministic grader for the ddd "checkout-domain-model" task.

Runs inside the agent's workspace (the current working directory). Statically
inspects the two required artifacts:

  domain.py       a Python tactical DDD model for an e-commerce checkout:
                  entities with identity (Order, Customer, Product), immutable
                  value objects (Money with amount+currency, Address), an
                  Order aggregate root that enforces invariants, domain events
                  (OrderPlaced, PaymentCaptured), and a repository port
                  (Protocol/ABC/abstractmethod).
  context-map.md  a strategic context map: >= 3 bounded contexts (Sales,
                  Inventory, Billing, Payment), a stated relationship pattern
                  (upstream/downstream, ACL, conformist, ...), and an explicit
                  statement of which context owns the Order aggregate.

Python 3 standard library ONLY (ast, json, re, sys) - no network, no PyYAML.

Prints JSON to stdout:
  {"score": 0.94, "details": "15/16 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import ast
import json
import re
import sys

PY_FILE = "domain.py"
MD_FILE = "context-map.md"
PASS_BAR = 0.8


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def check(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


# ---------------------------------------------------------------------------
# AST helpers for domain.py
# ---------------------------------------------------------------------------

def class_defs(tree):
    """Top-level class definitions in the module."""
    return [n for n in tree.body if isinstance(n, ast.ClassDef)]


def find_class(tree, name):
    """Return the ClassDef whose name matches `name` (case-insensitive)."""
    return next((c for c in class_defs(tree) if c.name.lower() == name.lower()), None)


def decorator_names(cls):
    """Simple names of all decorators applied to a class."""
    names = []
    for d in cls.decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
        elif isinstance(d, ast.Call):
            f = d.func
            names.append(f.id if isinstance(f, ast.Name) else f.attr)
    return names


def base_names(cls):
    """Simple names of all base classes of a class."""
    out = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def class_keywords(cls):
    """Keyword args of a class statement (e.g. frozen=True on a BaseModel)."""
    return [kw.arg for kw in cls.keywords if kw.arg]


def is_value_object(cls):
    """Dataclass, frozen Pydantic model, or BaseModel subclass."""
    if "dataclass" in decorator_names(cls):
        return True
    if "frozen" in class_keywords(cls) or any(b == "BaseModel" for b in base_names(cls)):
        return True
    return False


def is_frozen(cls):
    """True if the class is frozen: @dataclass(frozen=True), a frozen=True
    class kwarg (Pydantic), or a frozen-style decorator."""
    if "frozen" in class_keywords(cls):
        return True
    for d in cls.decorator_list:
        if isinstance(d, ast.Call):
            f = d.func
            fname = f.id if isinstance(f, ast.Name) else f.attr
            if fname == "frozen" or fname == "dataclass":
                for kw in d.keywords:
                    if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        return True
    return False


def class_field_names(cls):
    """Attribute names a class carries as fields: class-body annotations
    (dataclass/Pydantic fields) plus self.X assignments anywhere in the class
    (e.g. inside __init__)."""
    names = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for node in ast.walk(cls):
        if isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                names.add(t.attr)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    names.add(t.attr)
    return names


def class_subtree_names(cls):
    """Every identifier referenced anywhere inside a class body."""
    names = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def class_instantiates(cls, class_name):
    """True if the class body contains a call to `class_name(...)`."""
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == class_name:
                return True
    return False


def method_defs(cls):
    """Methods defined directly in the class body (incl. classmethod/staticmethod)."""
    return [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def method_has_raise(method):
    return any(isinstance(n, ast.Raise) for n in ast.walk(method))


MISSING_DOMAIN_CHECKS = (
    "python-syntax", "entity-classes", "entity-identity",
    "money-value-object", "address-value-object", "value-objects-immutable",
    "order-aggregate-behavior", "aggregate-invariant", "aggregate-refs-by-id",
    "domain-events", "repository-interface",
)


def check_domain(py, checks):
    exists = py is not None and py.strip() != ""
    checks.append(check(
        "domain-py-exists", exists,
        "domain.py present and non-empty" if exists else "domain.py not found or empty"))
    if not exists:
        for name in MISSING_DOMAIN_CHECKS:
            checks.append(check(name, False, "domain.py not found or empty"))
        return

    try:
        tree = ast.parse(py, filename=PY_FILE)
        syntax_ok, syntax_msg = True, "valid Python syntax"
    except SyntaxError as exc:
        tree = None
        syntax_ok, syntax_msg = False, "syntax error: line " + str(exc.lineno) + ": " + str(exc.msg)
    checks.append(check("python-syntax", syntax_ok, syntax_msg))

    if tree is None:
        for name in (
            "entity-classes", "entity-identity", "dataclass-value-objects",
            "money-value-object", "address-value-object", "order-aggregate-behavior",
            "aggregate-invariant", "aggregate-refs-by-id", "domain-events",
            "repository-interface",
        ):
            checks.append(check(name, False, "cannot parse domain.py"))
        return

    classes = {c.name.lower(): c for c in class_defs(tree)}

    # --- Entities with identity ---
    entity_names = ("order", "customer", "product")
    found = [n for n in entity_names if n in classes]
    checks.append(check(
        "entity-classes",
        len(found) == 3,
        ("entity classes found: " + str(found)) if found else "none of Order/Customer/Product defined",
    ))

    missing_id = [
        n for n in entity_names if n in classes and not any(
            f == "id" or f.endswith("_id") for f in class_field_names(classes[n]))
    ]
    checks.append(check(
        "entity-identity",
        not missing_id,
        "Order/Customer/Product each carry an identity field (id / *_id)"
        if not missing_id else "entities missing an id field: " + str(missing_id),
    ))

    # --- Value objects ---
    frozen_count = sum(1 for c in classes.values() if is_value_object(c) and is_frozen(c))
    checks.append(check(
        "value-objects-immutable",
        frozen_count >= 2,
        str(frozen_count) + " immutable value object(s) (frozen dataclass / frozen model)"
        if frozen_count else "value objects are not immutable - use @dataclass(frozen=True) or a frozen model",
    ))

    money = classes.get("money")
    money_fields = class_field_names(money) if money else set()
    money_ok = (money is not None and is_value_object(money)
                and "amount" in money_fields and "currency" in money_fields)
    checks.append(check(
        "money-value-object",
        money_ok,
        "Money value object with amount + currency fields"
        if money_ok else "missing Money value object carrying both amount and currency",
    ))

    address = classes.get("address")
    addr_hints = {"street", "city", "postal", "postal_code", "zip", "country", "state"}
    addr_hits = addr_hints & (class_field_names(address) if address else set())
    address_ok = address is not None and is_value_object(address) and len(addr_hits) >= 2
    checks.append(check(
        "address-value-object",
        address_ok,
        ("Address value object with address fields: " + str(sorted(addr_hits)))
        if address_ok else "missing Address value object (street/city/postal/country ...)",
    ))

    # --- Order aggregate ---
    order = classes.get("order")
    order_methods = method_defs(order) if order else []
    behavior = [m.name for m in order_methods if m.name != "__init__"]
    checks.append(check(
        "order-aggregate-behavior",
        len(behavior) > 0,
        ("Order exposes behavior method(s): " + str(behavior))
        if behavior else "Order is anemic - no behavior methods beyond __init__",
    ))

    invariant_ok = False
    for m in order_methods:
        if m.name.startswith("can_") or m.name in ("validate", "is_valid", "ensure_valid", "check_invariants"):
            invariant_ok = True
            break
        if method_has_raise(m):
            invariant_ok = True
            break
    checks.append(check(
        "aggregate-invariant",
        invariant_ok,
        "Order enforces invariants (validate/can_* method or raise on invalid state)"
        if invariant_ok else "Order has no invariant enforcement (no validate/can_* method, no raise)",
    ))

    order_names = class_subtree_names(order) if order else set()
    ref_by_id = any("customer_id" in n for n in order_names)
    embeds_customer = class_instantiates(order, "Customer") if order else False
    checks.append(check(
        "aggregate-refs-by-id",
        ref_by_id and not embeds_customer,
        "Order references customer by customer_id and does not embed a Customer object"
        if ref_by_id and not embeds_customer
        else "Order must reference the customer by ID (customer_id) and never instantiate Customer inside it",
    ))

    # --- Domain events ---
    events_ok = "orderplaced" in classes and "paymentcaptured" in classes
    events_found = [n for n in ("orderplaced", "paymentcaptured") if n in classes]
    missing_events = sorted({"orderplaced", "paymentcaptured"} - set(events_found))
    checks.append(check(
        "domain-events",
        events_ok,
        ("domain event classes found: " + str(events_found))
        if events_ok else "missing domain event class(es): " + str(missing_events),
    ))

    # --- Repository interface (port) ---
    repo_ok = False
    for c in classes.values():
        if c.name.lower().endswith("repository"):
            repo_ok = True
            break
        if any(b in ("ABC", "ABCMeta") or b.lower().endswith("protocol") for b in base_names(c)):
            repo_ok = True
            break
        if any("abstractmethod" in decorator_names(m) for m in method_defs(c)):
            repo_ok = True
            break
    checks.append(check(
        "repository-interface",
        repo_ok,
        "repository interface declared (XxxRepository / Protocol / ABC + abstractmethod)"
        if repo_ok else "missing repository interface (no XxxRepository, Protocol, or ABC/abstractmethod)",
    ))


def check_context_map(md, checks):
    if md is None or not md.strip():
        for name in ("context-map-exists", "bounded-contexts", "relationship-types", "order-ownership"):
            checks.append(check(name, False, "context-map.md not found or empty"))
        return

    checks.append(check("context-map-exists", True, "context-map.md present"))

    low = md.lower()
    contexts = ["sales", "inventory", "billing", "payment"]
    hits = [c for c in contexts if re.search(r"\b" + c + r"\b", low)]
    checks.append(check(
        "bounded-contexts",
        len(hits) >= 3,
        (str(len(hits)) + "/4 named contexts (Sales, Inventory, Billing, Payment): " + str(hits))
        if hits else "no named bounded contexts (Sales/Inventory/Billing/Payment)",
    ))

    patterns = [
        "anti-corruption", "acl", "conformist", "upstream", "downstream",
        "customer-supplier", "shared kernel", "open host", "published language",
        "partnership", "separate ways",
    ]
    pat_hits = [p for p in patterns if re.search(r"\b" + re.escape(p) + r"\b", low)]
    checks.append(check(
        "relationship-types",
        len(pat_hits) > 0,
        ("relationship pattern(s) stated: " + str(pat_hits))
        if pat_hits else "no relationship pattern stated (upstream/downstream, ACL, conformist, ...)",
    ))

    lines = md.splitlines()
    ownership = any("sales" in ln.lower() and "order" in ln.lower() for ln in lines)
    if not ownership:
        ownership = "sales" in low and "own" in low and re.search(r"\border\b", low)
    checks.append(check(
        "order-ownership",
        ownership,
        "states which context owns the Order aggregate (Sales owns Order)"
        if ownership else "context-map.md does not state which context owns Order",
    ))


def main():
    py = read_file(PY_FILE)
    md = read_file(MD_FILE)

    checks = []
    check_domain(py, checks)
    check_context_map(md, checks)

    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    score = (passed / total) if total else 0.0

    details = str(passed) + "/" + str(total) + " checks passed"
    missing = [f for f in (PY_FILE, MD_FILE) if read_file(f) is None]
    if missing:
        details += "; missing file(s): " + ", ".join(missing)

    print(json.dumps({
        "score": round(score, 4),
        "details": details,
        "checks": checks,
    }))
    sys.exit(0 if score >= PASS_BAR else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # never crash the harness - emit a zero-score JSON
        print(json.dumps({
            "score": 0.0,
            "details": "grader error: " + str(exc),
            "checks": [],
        }))
        sys.exit(1)
