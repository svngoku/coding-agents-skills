#!/usr/bin/env python3
"""Deterministic grader for the security-best-practices secure-login-rewrite task.

Runs inside the agent's workspace (the current working directory). Statically
inspects secure_app.py - the hardened rewrite of the vulnerable login endpoint
(fixtures/vulnerable.py) - with the ast module plus a few regexes over the
source. Python 3 standard library only: no third-party imports, no network,
no execution of the artifact.

Graded checks (the skill's core API surface / anti-patterns):
  1. secure-app-created   file exists and parses as Python
  2. password-hashing     argon2/bcrypt/passlib/werkzeug imported AND used
  3. parameterized-queries no string-built SQL; placeholders or ORM used
  4. no-user-enumeration  generic error, no "user not found" / 404-vs-401 leak
  5. login-rate-limit     per-IP limiter or in-memory attempt counter
  6. secrets-from-env     os.environ/os.getenv used, no hardcoded secret literals
  7. cookie-flags         every set_cookie() sets HttpOnly + Secure + SameSite
  8. no-dangerous-eval    no eval/exec/pickle.load/yaml.load

Output contract (printed as JSON to stdout):
  {"score": 0.0-1.0, "details": "...", "checks": [{"name","passed","message"}, ...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import ast
import json
import os
import re
import sys

APP_FILE = "secure_app.py"
PASS_BAR = 0.8

SQL_KEYWORD = re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", re.IGNORECASE)
PLACEHOLDER = re.compile(r"(\?|%s|%\(\w+\)s|:\w+|\$\d)")
HAS_FILTER = re.compile(r"\b(?:WHERE|VALUES|SET)\b", re.IGNORECASE)

HASHING_MODULES = ("argon2", "bcrypt", "passlib", "werkzeug")
HASHING_USAGE = re.compile(
    r"PasswordHasher\s*\(|hashpw\s*\(|generate_password_hash\s*\(|"
    r"check_password_hash\s*\(|CryptContext\s*\(|\.\s*(?:hash|verify|checkpw)\s*\("
)

ENUM_RE = re.compile(
    r"user\s+(?:not\s+found|unknown|not\s+registered|not\s+recognized|"
    r"not\s+in\s+our\s+system|doesn'?t\s+exist|does\s+not\s+exist)|"
    r"no\s+such\s+user|account\s+not\s+found|username\s+not\s+found|"
    r"email\s+not\s+found|user\s+doesn'?t\s+exist|"
    r"invalid\s+username\b(?!\s+or\s+password)|incorrect\s+username\b(?!\s+or\s+password)",
    re.IGNORECASE,
)
GENERIC_RE = re.compile(
    r"invalid\s+credentials|invalid\s+(?:username|email)\s+or\s+password|"
    r"incorrect\s+(?:username|email)\s+or\s+password|incorrect\s+(?:password|credentials)|"
    r"authentication\s+failed|login\s+failed|unauthorized|access\s+denied",
    re.IGNORECASE,
)

SECRET_NAME_RE = re.compile(
    r"\b(?:secret[_-]?key|api[_-]?key|access[_-]?token|jwt[_-]?secret|auth[_-]?token|"
    r"password|passwd|credential|token)\s*[:=]\s*[\"'][^\"']{8,}[\"']",
    re.IGNORECASE,
)
ENV_FALLBACK_RE = re.compile(
    r"os\.(?:environ\.get|getenv)\s*\(\s*[\"'][^\"']*(?:secret|key|token|password|"
    r"passwd|credential|auth)[^\"']*[\"']\s*,\s*[\"'][A-Za-z0-9_\-./+]{10,}[\"']\s*\)",
    re.IGNORECASE,
)
ENV_USE_RE = re.compile(r"os\.(?:environ|getenv)\b")

COOKIE_FLAG_RE = re.compile(r"\b(httponly|http_only|secure|samesite|same_site)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def add(checks, name, passed, message):
    checks.append({"name": name, "passed": bool(passed), "message": message})


def docstring_ids(tree):
    """ids of Constant nodes that are docstrings (skipped by message scans)."""
    ids = set()

    def visit(body):
        if body:
            stmt = body[0]
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                ids.add(id(stmt.value))

    visit(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            visit(node.body)
    return ids


def message_strings(tree):
    """String literals in the module that are messages, excluding docstrings."""
    skip = docstring_ids(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            out.append(node.value)
    return out


def calls_named(tree, names):
    """True if any call in the module resolves to one of the given names."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname in names:
                return True
    return False


def module_names(tree):
    """Top-level module/package names this file imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def set_cookie_regions(src):
    """Argument text of every set_cookie(...) call, found with a paren scan."""
    regions = []
    for m in re.finditer(r"\bset_cookie\s*\(", src, re.IGNORECASE):
        j = m.end() - 1  # position of '('
        depth = 0
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        regions.append(src[m.end():j])
    return regions


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_password_hashing(src, tree):
    """argon2/bcrypt/passlib/werkzeug imported AND actually used."""
    imports = module_names(tree)
    has_lib = any(mod in HASHING_MODULES for mod in imports)
    if not has_lib:
        return False, ("no argon2/bcrypt/passlib/werkzeug import found - "
                       "passwords must be hashed with a dedicated KDF")
    if not HASHING_USAGE.search(src):
        return False, ("hashing library imported but never used to hash/verify "
                       "a password (look for PasswordHasher(), hashpw(), "
                       "generate_password_hash(), or .verify())")
    return True, "argon2/bcrypt/passlib/werkzeug imported and used for password hashing"


def check_sql(src, tree):
    """No string-built SQL; parameterized queries or ORM present."""
    # 1. f-strings that build SQL (only when they interpolate).
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and node.values:
            text = "".join(
                v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if SQL_KEYWORD.search(text):
                return False, "SQL built with an f-string (injection risk)"

    # 2. .format() on a SQL string.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"):
            base = node.func.value
            if (isinstance(base, ast.Constant) and isinstance(base.value, str)
                    and SQL_KEYWORD.search(base.value)):
                return False, "SQL built with str.format()"

    # 3. %-interpolation on a SQL string.
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            left = node.left
            if (isinstance(left, ast.Constant) and isinstance(left.value, str)
                    and SQL_KEYWORD.search(left.value)):
                return False, "SQL built with %-formatting"

    # 4. string concatenation that mixes SQL with a variable.
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = node.left, node.right
            a_const = isinstance(a, ast.Constant) and isinstance(a.value, str)
            b_const = isinstance(b, ast.Constant) and isinstance(b.value, str)
            if (a_const and not b_const and SQL_KEYWORD.search(a.value)) or (
                b_const and not a_const and SQL_KEYWORD.search(b.value)
            ):
                return False, "SQL built via string concatenation with a variable"

    # 5. execute() with a literal SQL string that filters but has no placeholder.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute" and node.args):
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                q = arg0.value
                if SQL_KEYWORD.search(q) and HAS_FILTER.search(q) and not PLACEHOLDER.search(q):
                    return False, "non-parameterized SQL query (no ?/%s placeholder) in execute()"

    # Parameterized mechanism: placeholder in a SQL literal, params passed
    # separately, or an ORM/query-builder call. "SQL present" is judged from
    # string literals and execute()/ORM calls only - never from the raw source
    # (the Python keyword 'from' would otherwise match the SQL keyword FROM).
    parameterized = False
    sql_literal_seen = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SQL_KEYWORD.search(node.value):
                sql_literal_seen = True
                if PLACEHOLDER.search(node.value):
                    parameterized = True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                sql_literal_seen = True
                if len(node.args) >= 2:
                    parameterized = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("filter", "where", "add"):
                parameterized = True

    if parameterized:
        return True, "queries are parameterized (placeholders) or use an ORM/query-builder"
    if not sql_literal_seen:
        return True, "no SQL statements present - injection surface absent"
    return False, "SQL is present but never parameterized - use placeholders or an ORM"


def check_enumeration(src, tree):
    """Identical generic error; no user-existence leak or 404-vs-401 oracle."""
    # Leaking error-message phrases (from string literals, not comments/docstrings).
    for msg in message_strings(tree):
        if ENUM_RE.search(msg):
            return False, f"user-enumeration leak in message: {msg!r}"
    # 404-style leaks: HTTPException(404), abort(404), status_code=404,
    # or return (..., 404) - the "unknown user" vs "wrong password" oracle.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if fname in ("HTTPException", "abort"):
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 404:
                    return False, "404 used for the login failure path - leaks whether the user exists"
            for kw in node.keywords:
                if kw.arg == "status_code" and isinstance(kw.value, ast.Constant) and kw.value.value == 404:
                    return False, "status_code=404 used on the login failure path - user-existence oracle"
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and elt.value == 404:
                    return False, "login failure returns HTTP 404 - leaks whether the user exists"
    generic = any(GENERIC_RE.search(m) for m in message_strings(tree))
    if not generic:
        return False, ("no generic error message found (e.g. 'Invalid credentials') - "
                       "unknown-user and wrong-password must return the same message")
    return True, "single generic login error; no user-existence leak"


def check_rate_limit(src, tree):
    """Rate limiting must exist as CODE (imports, identifiers, literals);
    comments and docstrings cannot satisfy this check."""
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
            tokens.add(str(node.value).lower())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                tokens.add(alias.name.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                tokens.add(node.module.lower())
                tokens.add(node.module.split(".")[0].lower())
            for alias in node.names:
                tokens.add(alias.name.lower())

    markers = {
        "slowapi", "flask_limiter", "flask-limiter", "limiter", "rate_limit",
        "rate_limiter", "rate_limiting", "too_many_requests", "too many requests",
        "throttle", "throttling", "429", "login_attempts", "failed_attempts",
        "attempt_counts", "attempts", "requests_per_minute",
    }
    hit = sorted(markers & tokens)
    if hit:
        return True, f"login rate limiting present ({', '.join(hit)})"
    return False, ("no login rate limiting found in code "
                   "(e.g. slowapi/Limiter or a per-IP attempt counter)")


def check_secrets(src, tree):
    if ENV_FALLBACK_RE.search(src):
        return False, "a secret-ish env var has a hardcoded fallback value - remove it"
    if SECRET_NAME_RE.search(src):
        return False, "hardcoded secret string assigned in the source (use os.environ/os.getenv)"
    if not ENV_USE_RE.search(src):
        return False, "secrets are not loaded from the environment (no os.environ/os.getenv)"
    return True, "secrets loaded from os.environ/os.getenv; no hardcoded secret literals"


def check_cookies(src, tree):
    regions = set_cookie_regions(src)
    if not regions:
        return True, "no session cookie set - check skipped (safe if no sessions are used)"
    for region in regions:
        flags = set(COOKIE_FLAG_RE.findall(region))
        flags = {f.lower() for f in flags}
        missing = []
        if not ({"httponly", "http_only"} & flags):
            missing.append("HttpOnly")
        if "secure" not in flags:
            missing.append("Secure")
        if not ({"samesite", "same_site"} & flags):
            missing.append("SameSite")
        if missing:
            return False, f"set_cookie() missing flag(s): {', '.join(missing)}"
    return True, "every set_cookie() sets HttpOnly + Secure + SameSite"


def check_no_eval(src, tree):
    if calls_named(tree, {"eval", "exec"}):
        return False, "eval()/exec() found - arbitrary code execution on untrusted input"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            obj = node.func.value
            obj_name = obj.id if isinstance(obj, ast.Name) else (
                obj.attr if isinstance(obj, ast.Attribute) else None
            )
            if obj_name in ("pickle", "yaml") and node.func.attr in ("load", "loads"):
                return False, f"{obj_name}.{node.func.attr}() on (possibly untrusted) input found"
    return True, "no eval/exec/pickle.load/yaml.load found"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

CHECKS = [
    ("secure-app-created", None),
    ("password-hashing", check_password_hashing),
    ("parameterized-queries", check_sql),
    ("no-user-enumeration", check_enumeration),
    ("login-rate-limit", check_rate_limit),
    ("secrets-from-env", check_secrets),
    ("cookie-flags", check_cookies),
    ("no-dangerous-eval", check_no_eval),
]


def main():
    src = read_file(APP_FILE)
    checks = []
    if src is None:
        for name, _ in CHECKS:
            add(checks, name, False, f"{APP_FILE} not found")
        passed = 0
        total = len(checks)
        score = 0.0
        details = f"{passed}/{total} checks passed; missing file: {APP_FILE}"
    else:
        try:
            tree = ast.parse(src, filename=APP_FILE)
        except SyntaxError as exc:
            tree = None
            for i, (name, _) in enumerate(CHECKS):
                if i == 0:
                    add(checks, name, False,
                        f"{APP_FILE} exists but has a syntax error: line {exc.lineno}: {exc.msg}")
                else:
                    add(checks, name, False,
                        "cannot analyze - syntax error in secure_app.py")
            passed = sum(1 for c in checks if c["passed"])
            total = len(checks)
            score = passed / total
            details = f"{passed}/{total} checks passed; {APP_FILE} has a syntax error"
        else:
            for name, fn in CHECKS:
                if name == "secure-app-created":
                    add(checks, name, True, f"{APP_FILE} exists and parses as valid Python")
                else:
                    ok, msg = fn(src, tree)
                    add(checks, name, ok, msg)
            passed = sum(1 for c in checks if c["passed"])
            total = len(checks)
            score = passed / total
            details = f"{passed}/{total} checks passed"

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
            "details": f"grader error: {exc}",
            "checks": [],
        }))
        sys.exit(1)
