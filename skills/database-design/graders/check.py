#!/usr/bin/env python3
"""Deterministic grader for the database-design schema-design-ecommerce task.

Runs inside the agent's workspace (the current working directory). Statically
inspects the two required artifacts - schema.sql and DESIGN.md - and prints a
skillgrade result as JSON to stdout.

Output contract:
  {"score": 0.0-1.0, "details": "...", "checks": [{"name", "passed", "message"}, ...]}

Python 3 standard library only, no network, no database.
"""

import json
import re
import sys

SQL_FILE = "schema.sql"
DESIGN_FILE = "DESIGN.md"
PASS_BAR = 0.8


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def split_create_tables(sql):
    """Return [(table_name, body), ...] for every CREATE TABLE statement.

    Uses a paren-depth scan so CHECK constraints containing parentheses
    (and single-quoted string literals) do not break block extraction.
    """
    tables = []
    pattern = re.compile(
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?([A-Za-z_][\w.]*)\"?\s*\(",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        name = match.group(1).strip('"')
        open_idx = match.end() - 1  # position of the '('
        depth = 0
        j = open_idx
        while j < len(sql):
            ch = sql[j]
            if ch == "'":  # skip single-quoted string literals
                j += 1
                while j < len(sql) and sql[j] != "'":
                    j += 1
                j += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        tables.append((name, sql[open_idx + 1:j]))
    return tables


def index_columns(sql, table):
    """Return the column-list strings of every CREATE INDEX on the given table."""
    cols = []
    pattern = re.compile(
        r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+[A-Za-z_][\w.]*\s+ON\s+(\w+)\s*\(([^)]*)\)",
        re.IGNORECASE,
    )
    for tbl, columns in pattern.findall(sql):
        if tbl.lower() == table.lower():
            cols.append(columns)
    return cols


def check(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


def main():
    sql = read_file(SQL_FILE) or ""
    design = read_file(DESIGN_FILE) or ""
    missing = [f for f in (SQL_FILE, DESIGN_FILE) if read_file(f) is None]

    tables = split_create_tables(sql)
    table_names = [t for t, _ in tables]
    order_items_body = next((b for t, b in tables if t.lower() == "order_items"), None)

    checks = []

    # 1. At least 4 CREATE TABLE statements (customers, products, orders, order_items).
    checks.append(check(
        "at-least-4-tables",
        len(tables) >= 4,
        f"found {len(tables)} CREATE TABLE statement(s): {', '.join(table_names) or 'none'}",
    ))

    # 2. Every table declares a PRIMARY KEY (requires at least one table).
    missing_pk = [t for t, b in tables if not re.search(r"\bPRIMARY\s+KEY\b", b, re.I)]
    checks.append(check(
        "primary-key-everywhere",
        len(tables) > 0 and not missing_pk,
        "every table declares a PRIMARY KEY"
        if not missing_pk else f"tables missing PRIMARY KEY: {', '.join(missing_pk)}",
    ))

    # 3. Foreign keys reference the core entities (customers, orders, products).
    missing_refs = [
        t for t in ("customers", "orders", "products")
        if not re.search(r"\bREFERENCES\s+\"?\s*" + t + r"\b", sql, re.I)
    ]
    ref_count = len(re.findall(r"\bREFERENCES\b", sql, re.I))
    checks.append(check(
        "foreign-keys-reference-entities",
        not missing_refs and ref_count >= 2,
        f"{ref_count} REFERENCES clause(s); missing REFERENCES to: "
        + (", ".join(missing_refs) if missing_refs else "none - all core entities are referenced"),
    ))

    # 4. order_items cascades with its order (owner-child lifecycle).
    oi_cascade = bool(order_items_body) and re.search(
        r"\bON\s+DELETE\s+CASCADE\b", order_items_body, re.I)
    checks.append(check(
        "order-items-cascade",
        oi_cascade,
        "ON DELETE CASCADE found on order_items (items die with their order)"
        if oi_cascade else "order_items must declare ON DELETE CASCADE on its order foreign key",
    ))

    # 5. At least one foreign key uses the safe default RESTRICT / NO ACTION.
    restrict_ok = bool(re.search(r"\bON\s+DELETE\s+(?:RESTRICT|NO\s+ACTION)\b", sql, re.I))
    checks.append(check(
        "restrict-default-present",
        restrict_ok,
        "found ON DELETE RESTRICT / NO ACTION"
        if restrict_ok else "no ON DELETE RESTRICT / NO ACTION found on any foreign key",
    ))

    # 6. CHECK enforces positive quantity.
    qty_ok = bool(re.search(r"\bCHECK\s*\(\s*quantity\b[^)]*\>\s*=?\s*0\b", sql, re.I))
    checks.append(check(
        "check-quantity-positive",
        qty_ok,
        "CHECK (quantity > 0) found" if qty_ok else "missing CHECK enforcing quantity > 0",
    ))

    # 7. CHECK enforces non-negative unit price (money in integer cents).
    price_ok = bool(re.search(r"\bCHECK\s*\(\s*unit_price[\w]*\s*\>\s*=?\s*0\b", sql, re.I))
    checks.append(check(
        "check-unit-price",
        price_ok,
        "CHECK (unit_price... >= 0) found" if price_ok else "missing CHECK enforcing unit_price >= 0",
    ))

    # 8. customers.email is NOT NULL and UNIQUE (business key).
    email_not_null = bool(re.search(r"\bemail\b[^;,()]*\bNOT\s+NULL\b", sql, re.I))
    email_unique = bool(re.search(r"\bemail\b[^;,()]*\bUNIQUE\b", sql, re.I)) or bool(
        re.search(r"\bUNIQUE\s*\(\s*email\s*\)", sql, re.I))
    checks.append(check(
        "email-unique-not-null",
        email_not_null and email_unique,
        "customers.email is NOT NULL and UNIQUE"
        if email_not_null and email_unique
        else f"customers.email needs NOT NULL ({email_not_null}) and UNIQUE ({email_unique})",
    ))

    # 9. Composite index on orders (customer_id, created_at) - equality first, then sort.
    orders_indexes = index_columns(sql, "orders")
    composite_ok = any(
        re.search(r"\bcustomer_id\b", c) and re.search(r"\bcreated_at\b", c)
        for c in orders_indexes
    )
    checks.append(check(
        "orders-composite-index",
        composite_ok,
        "CREATE INDEX ... ON orders (customer_id, created_at) found"
        if composite_ok else "missing composite index on orders (customer_id, created_at)",
    ))

    # 10. Index on order_items (order_id) - order-detail reads.
    oi_indexes = index_columns(sql, "order_items")
    oi_ok = any(re.search(r"\border_id\b", c) for c in oi_indexes)
    checks.append(check(
        "order-items-index",
        oi_ok,
        "CREATE INDEX ... ON order_items (order_id) found"
        if oi_ok else "missing index on order_items (order_id)",
    ))

    # 11. DESIGN.md discusses normalization.
    norm_ok = bool(re.search(r"\b(?:normaliz\w*|1NF|2NF|3NF)\b", design, re.I))
    checks.append(check(
        "design-normalization",
        norm_ok,
        "DESIGN.md discusses normalization (1NF/2NF/3NF)"
        if norm_ok else "DESIGN.md does not discuss normalization",
    ))

    # 12. DESIGN.md explains the surrogate vs natural key choice.
    surrogate_ok = bool(re.search(
        r"\bsurrogate\b|\bGENERATED\s+ALWAYS\s+AS\s+IDENTITY\b", design, re.I))
    checks.append(check(
        "design-key-choice",
        surrogate_ok,
        "DESIGN.md explains the surrogate key choice"
        if surrogate_ok else "DESIGN.md does not explain surrogate vs natural keys",
    ))

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    score = passed / total

    details = f"{passed}/{total} checks passed"
    if missing:
        details += f"; missing file(s): {', '.join(missing)}"

    print(json.dumps({
        "score": round(score, 2),
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
            "details": f"grader error: {exc}",
            "checks": [],
        }))
        sys.exit(1)
