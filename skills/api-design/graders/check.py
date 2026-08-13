#!/usr/bin/env python3
"""Deterministic grader for the api-design "orders-api-openapi" eval task.

Runs from the agent's workspace (cwd). Statically inspects the single required
artifact, openapi.yaml, an OpenAPI 3.x spec for an orders API:

  /orders           GET  cursor-paginated list (status filter + sort)
                    POST create with a REQUIRED Idempotency-Key header, 201 + Location
  /orders/{id}      GET + PATCH (partial update)
  /orders/{id}/cancel  POST domain action
  responses 201 + Location, 200, 422, 429 referencing an RFC 7807
  application/problem+json Problem schema (components.schemas.Problem)
  security: OAuth2 client-credentials flow, read:orders / write:orders scopes
  info.version set; a pagination schema with next_cursor / has_more

Python 3 standard library ONLY (no PyYAML, no network). YAML is checked with a
small hand-rolled subset parser (block mappings, block sequences, inline flow)
sufficient for OpenAPI spec files.

Prints JSON to stdout:
  {"score": 0.93, "details": "13/14 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import json
import os
import sys

SPEC_FILE = "openapi.yaml"
PASS_BAR = 0.8

CHECK_NAMES = [
    "yaml-parses",
    "openapi-3x",
    "paths-present",
    "orders-id-get-patch",
    "cancel-action-post",
    "orders-get-cursor-limit",
    "orders-get-status-sort",
    "orders-post-idempotency-header",
    "orders-post-201-location",
    "orders-post-422",
    "error-response-429",
    "problem-json-envelope",
    "pagination-schema",
    "oauth2-client-credentials",
    "oauth2-scopes",
    "info-version",
]


# ---------------------------------------------------------------------------
# Minimal YAML-subset parser (block mappings, block sequences, inline flow)
# ---------------------------------------------------------------------------

def _strip_comment(line):
    """Remove a trailing comment, respecting quotes."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line.rstrip()


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _split_key(content):
    """Split 'key: value' at the first top-level colon. Returns (key, rest)."""
    in_single = in_double = False
    depth = 0
    for i, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "[{" and not in_single and not in_double:
            depth += 1
        elif ch in "]}" and not in_single and not in_double:
            depth -= 1
        elif ch == ":" and not in_single and not in_double and depth == 0:
            if i + 1 == len(content) or content[i + 1].isspace():
                return _unquote(content[:i]), content[i + 1:].strip()
    return None, None


def _split_flow(s, delim=","):
    """Split a flow section on delim at top nesting level only."""
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == delim and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


def _parse_scalar(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_flow(s):
    s = s.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError("unclosed flow sequence")
        inner = s[1:-1].strip()
        if inner == "":
            return []
        return [_parse_flow(p) for p in _split_flow(inner)]
    if s.startswith("{"):
        if not s.endswith("}"):
            raise ValueError("unclosed flow mapping")
        inner = s[1:-1].strip()
        out = {}
        if inner == "":
            return out
        for part in _split_flow(inner):
            key, rest = _split_key(part)
            if key is not None:
                out[key] = _parse_flow(rest) if rest else None
        return out
    return _parse_scalar(s)


def _build_tree(lines):
    """Group (indent, content) lines into an indentation tree."""
    stack = [(-1, [])]
    for indent, content in lines:
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            return None
        node = {"indent": indent, "content": content, "children": []}
        stack[-1][1].append(node)
        stack.append((indent, node["children"]))
    return stack[0][1]


def _has_key(content):
    key, _ = _split_key(content)
    return key is not None


def _interpret(nodes):
    """Interpret a tree node list as a Python value (dict/list/scalar)."""
    if not nodes:
        return None
    first = nodes[0]["content"]
    if first == "-" or first.startswith("- "):
        out = []
        for node in nodes:
            rest = node["content"][1:].strip()
            if rest == "":
                out.append(_interpret(node["children"]))
            elif rest.startswith("{") or rest.startswith("["):
                out.append(_parse_flow(rest))
            elif _has_key(rest):
                out.append(_mapping_from(rest, node["children"]))
            else:
                out.append(_parse_scalar(rest))
        return out
    out = {}
    for node in nodes:
        key, rest = _split_key(node["content"])
        if key is None:
            continue
        if rest == "":
            out[key] = _interpret(node["children"])
        elif rest.startswith("{") or rest.startswith("["):
            out[key] = _parse_flow(rest)
        else:
            out[key] = _parse_scalar(rest)
    return out


def _mapping_from(first_rest, child_nodes):
    """A sequence item that begins a mapping, e.g. '- name: X' + deeper keys."""
    m = {}
    key, rest = _split_key(first_rest)
    if key is not None:
        if rest == "":
            m[key] = _interpret(child_nodes)
        elif rest.startswith("{") or rest.startswith("["):
            m[key] = _parse_flow(rest)
        else:
            m[key] = _parse_scalar(rest)
    child_map = _interpret(child_nodes)
    if isinstance(child_map, dict):
        m.update(child_map)
    return m


def parse_yaml(text):
    """Parse the YAML subset. Returns a dict, or None if unparseable."""
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        line = _strip_comment(line)
        if not line.strip():
            continue
        leading = line[: len(line) - len(line.lstrip())]
        if "\t" in leading:
            return None  # tabs are invalid YAML indentation
        indent = len(leading)
        lines.append((indent, line[len(leading):]))
    if not lines:
        return None
    tree = _build_tree(lines)
    if tree is None:
        return None
    value = _interpret(tree)
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# OpenAPI helpers
# ---------------------------------------------------------------------------

def _operations(spec):
    """Yield (path, method, operation) for every operation in the spec."""
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in ("get", "post", "put", "patch", "delete",
                       "head", "options", "trace"):
            op = item.get(method)
            if isinstance(op, dict):
                yield path, method, op


def _operation(spec, path, method):
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return None
    item = paths.get(path)
    if not isinstance(item, dict):
        return None
    op = item.get(method)
    return op if isinstance(op, dict) else None


def _params(op):
    params = op.get("parameters") if isinstance(op, dict) else None
    return params if isinstance(params, list) else []


def _param(params, name, location):
    """Find a parameter by name (case-insensitive) and location."""
    for p in params:
        if not isinstance(p, dict):
            continue
        if str(p.get("in", "")).lower() == location and                 str(p.get("name", "")).lower() == name.lower():
            return p
    return None


def _responses(op):
    resp = op.get("responses") if isinstance(op, dict) else None
    return resp if isinstance(resp, dict) else {}


def _deep_has_key(obj, target):
    """True if any dict in obj has a key equal to target (recursive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target:
                return True
            if _deep_has_key(v, target):
                return True
    elif isinstance(obj, list):
        return any(_deep_has_key(v, target) for v in obj)
    return False


def _deep_has_str(obj, substr):
    """True if substr appears in any string key or value in obj (recursive)."""
    if isinstance(obj, str):
        return substr in obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _deep_has_str(k, substr) or _deep_has_str(v, substr):
                return True
    elif isinstance(obj, list):
        return any(_deep_has_str(v, substr) for v in obj)
    return False


def _resolve_ref(spec, ref):
    """Resolve '#/a/b/c' style refs against the spec. Returns node or None."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node = spec
    for part in ref[2:].split("/"):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _refs_problem(spec, resp, depth=0):
    """True if a response references the RFC 7807 Problem schema.

    Accepts either an 'application/problem+json' content type on the response,
    any '#/components/schemas/Problem' (or containing 'Problem') reference in
    the response subtree, or a shared-response $ref that resolves to one.
    """
    if depth > 4 or not isinstance(resp, dict):
        return False
    content = resp.get("content")
    if isinstance(content, dict) and any("problem+json" in str(k) for k in content):
        return True
    if _deep_has_str(resp, "Problem"):
        return True
    ref = resp.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_ref(spec, ref)
        if isinstance(resolved, dict):
            return _refs_problem(spec, resolved, depth + 1)
    return False


def _first_response(spec, status):
    """Return the first operation response with the given status key, or None."""
    for _path, _method, op in _operations(spec):
        resp = _responses(op).get(str(status))
        if isinstance(resp, dict):
            return resp
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(name, passed, message):
    return {"name": name, "passed": bool(passed), "message": message}


def _fail_all(checks, reason):
    for name in CHECK_NAMES:
        checks.append(check(name, False, reason))


def main():
    checks = []

    if not os.path.isfile(SPEC_FILE):
        _fail_all(checks, "openapi.yaml not found")
        _emit(checks)
        return

    try:
        with open(SPEC_FILE, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        _fail_all(checks, f"cannot read openapi.yaml: {exc}")
        _emit(checks)
        return

    spec = parse_yaml(text)
    checks.append(check(
        "yaml-parses",
        spec is not None,
        "parsed as YAML dict" if spec is not None else "unparseable YAML",
    ))
    if spec is None:
        for name in CHECK_NAMES[1:]:
            checks.append(check(name, False, "YAML did not parse"))
        _emit(checks)
        return

    # 2. OpenAPI 3.x declaration.
    ov = spec.get("openapi")
    ov_ok = isinstance(ov, str) and ov.startswith("3")
    checks.append(check(
        "openapi-3x", ov_ok,
        f"openapi: {ov!r}" if ov_ok
        else f"top-level openapi must be a 3.x version string, got {ov!r}",
    ))

    # 3. Required paths present.
    paths = spec.get("paths")
    required_paths = ("/orders", "/orders/{id}", "/orders/{id}/cancel")
    missing_paths = [
        p for p in required_paths
        if not (isinstance(paths, dict) and isinstance(paths.get(p), dict))
    ]
    checks.append(check(
        "paths-present", not missing_paths,
        "paths /orders, /orders/{id}, /orders/{id}/cancel declared"
        if not missing_paths
        else f"missing path(s): {', '.join(missing_paths)}",
    ))

    # 4. /orders/{id} is GET + PATCH (partial update, not PUT).
    op_get = _operation(spec, "/orders/{id}", "get")
    op_patch = _operation(spec, "/orders/{id}", "patch")
    get_patch_ok = op_get is not None and op_patch is not None
    methods = []
    item = paths.get("/orders/{id}") if isinstance(paths, dict) else None
    if isinstance(item, dict):
        methods = [m for m in ("get", "put", "patch", "post", "delete") if item.get(m)]
    checks.append(check(
        "orders-id-get-patch", get_patch_ok,
        "GET and PATCH defined on /orders/{id}"
        if get_patch_ok else f"/orders/{{id}} methods found: {methods or 'none'}; need GET + PATCH",
    ))

    # 5. /orders/{id}/cancel is a POST action.
    op_cancel = _operation(spec, "/orders/{id}/cancel", "post")
    cancel_methods = []
    citem = paths.get("/orders/{id}/cancel") if isinstance(paths, dict) else None
    if isinstance(citem, dict):
        cancel_methods = [m for m in ("get", "put", "patch", "post", "delete") if citem.get(m)]
    checks.append(check(
        "cancel-action-post", op_cancel is not None,
        "POST /orders/{id}/cancel declared"
        if op_cancel is not None
        else f"/orders/{{id}}/cancel methods found: {cancel_methods or 'none'}; expected a POST action",
    ))

    # 6. GET /orders has cursor + limit query params.
    op_list = _operation(spec, "/orders", "get")
    list_params = _params(op_list)
    cur = _param(list_params, "cursor", "query")
    lim = _param(list_params, "limit", "query")
    cur_lim_ok = cur is not None and lim is not None
    checks.append(check(
        "orders-get-cursor-limit", cur_lim_ok,
        "GET /orders has cursor and limit query params"
        if cur_lim_ok
        else f"cursor: {'found' if cur else 'missing'}, limit: {'found' if lim else 'missing'}",
    ))

    # 7. GET /orders has status filter + sort params.
    st = _param(list_params, "status", "query")
    so = _param(list_params, "sort", "query")
    st_so_ok = st is not None and so is not None
    checks.append(check(
        "orders-get-status-sort", st_so_ok,
        "GET /orders has status filter and sort query params"
        if st_so_ok
        else f"status filter: {'found' if st else 'missing'}, sort: {'found' if so else 'missing'}",
    ))

    # 8. POST /orders requires the Idempotency-Key header.
    op_create = _operation(spec, "/orders", "post")
    create_params = _params(op_create)
    idem = _param(create_params, "Idempotency-Key", "header")
    idem_ok = idem is not None and (
        idem.get("required") is True
        or str(idem.get("required", "")).lower() == "true"
    )
    checks.append(check(
        "orders-post-idempotency-header", idem_ok,
        "POST /orders has a required Idempotency-Key header param"
        if idem_ok
        else "POST /orders is missing a required Idempotency-Key header parameter",
    ))

    # 9. POST /orders -> 201 with a Location header.
    create_resp = _responses(op_create) if op_create else {}
    resp201 = create_resp.get("201")
    has_loc = (
        isinstance(resp201, dict)
        and isinstance(resp201.get("headers"), dict)
        and any(str(k).lower() == "location" for k in resp201["headers"])
    )
    checks.append(check(
        "orders-post-201-location", has_loc,
        "POST /orders declares 201 with a Location header"
        if has_loc
        else "POST /orders must declare a 201 response with a Location header",
    ))

    # 10. POST /orders -> 422 for semantically invalid input.
    has_422 = isinstance(create_resp.get("422"), dict)
    checks.append(check(
        "orders-post-422", has_422,
        "POST /orders declares a 422 response"
        if has_422 else "POST /orders is missing a 422 response",
    ))

    # 11. Some operation declares a 429 (rate limit) response.
    resp429 = _first_response(spec, "429")
    checks.append(check(
        "error-response-429", resp429 is not None,
        "a 429 Too Many Requests response is declared"
        if resp429 is not None
        else "no 429 response declared on any operation",
    ))

    # 12. Error envelope: 422 + 429 reference RFC 7807 problem+json Problem.
    post422 = create_resp.get("422") if isinstance(create_resp.get("422"), dict) else None
    refs_422 = _refs_problem(spec, post422) if post422 is not None else False
    refs_429 = _refs_problem(spec, resp429) if resp429 is not None else False
    comps = spec.get("components")
    schemas = comps.get("schemas") if isinstance(comps, dict) else None
    problem_schema = (
        isinstance(schemas, dict) and isinstance(schemas.get("Problem"), dict)
    )
    env_ok = refs_422 and refs_429 and problem_schema
    checks.append(check(
        "problem-json-envelope", env_ok,
        "422/429 reference application/problem+json Problem schema"
        if env_ok
        else "error responses must use application/problem+json and reference components.schemas.Problem "
             f"(422: {'ok' if refs_422 else 'missing'}, 429: {'ok' if refs_429 else 'missing'}, "
             f"Problem schema: {'ok' if problem_schema else 'missing'})",
    ))

    # 13. Pagination schema with next_cursor + has_more.
    has_next = _deep_has_key(schemas, "next_cursor") if isinstance(schemas, dict) else False
    has_more = _deep_has_key(schemas, "has_more") if isinstance(schemas, dict) else False
    page_ok = has_next and has_more
    checks.append(check(
        "pagination-schema", page_ok,
        "components.schemas has a pagination schema with next_cursor and has_more"
        if page_ok
        else f"pagination schema missing next_cursor ({has_next}) and/or has_more ({has_more})",
    ))

    # 14. OAuth2 client-credentials security scheme.
    schemes_dict = comps.get("securitySchemes") if isinstance(comps, dict) else None
    oauth2 = schemes_dict.get("oauth2") if isinstance(schemes_dict, dict) else None
    flows = oauth2.get("flows") if isinstance(oauth2, dict) else None
    cc = flows.get("clientCredentials") if isinstance(flows, dict) else None
    oauth2_ok = (
        isinstance(oauth2, dict)
        and oauth2.get("type") == "oauth2"
        and isinstance(cc, dict)
    )
    checks.append(check(
        "oauth2-client-credentials", oauth2_ok,
        "securitySchemes.oauth2 uses type oauth2 with flows.clientCredentials"
        if oauth2_ok
        else "missing securitySchemes.oauth2 with type: oauth2 and flows.clientCredentials",
    ))

    # 15. Client-credentials scopes include read:orders and write:orders.
    scopes = cc.get("scopes") if isinstance(cc, dict) else None
    if isinstance(scopes, dict):
        scope_names = list(scopes.keys())
    elif isinstance(scopes, list):
        scope_names = [str(s) for s in scopes]
    else:
        scope_names = []
    scopes_ok = "read:orders" in scope_names and "write:orders" in scope_names
    checks.append(check(
        "oauth2-scopes", scopes_ok,
        "clientCredentials scopes include read:orders and write:orders"
        if scopes_ok
        else f"clientCredentials scopes: {scope_names or 'none'}; need read:orders and write:orders",
    ))

    # 16. info.version set and non-empty.
    info = spec.get("info")
    version = info.get("version") if isinstance(info, dict) else None
    ver_ok = version is not None and str(version).strip() != ""
    checks.append(check(
        "info-version", ver_ok,
        f"info.version: {version!r}" if ver_ok else "info.version is missing or empty",
    ))

    _emit(checks)


def _emit(checks):
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    score = (passed / total) if total else 0.0
    details = f"{passed}/{total} checks passed"
    if not os.path.isfile(SPEC_FILE):
        details += "; missing file: openapi.yaml"
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
