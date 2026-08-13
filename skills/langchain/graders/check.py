#!/usr/bin/env python3
"""
Deterministic grader for the langchain skill's customer-support-agent task.

Runs in the workspace where the agent wrote its output and statically analyzes
the produced artifact (agent.py, or any *.py files present) using only the
Python standard library - no network, no langchain imports required.

Checks the API surface and best practices the langchain SKILL.md emphasizes:
current-API imports, tool docstrings + type hints, ToolRuntime[Context]
plumbing, checkpointer wiring, ToolStrategy(Response) structured output, and
an example invocation that passes a thread_id and a context instance.

Prints a JSON report to stdout:
    {"score": 0.93, "details": "14/15 checks passed", "checks": [...]}
Exit code 0 if score >= 0.8, else 1.
"""

import ast
import glob
import json
import os
import sys

PASS_BAR = 0.8


def find_sources():
    """Locate the agent artifact in the workspace."""
    if os.path.isfile("agent.py"):
        return ["agent.py"]
    py_files = sorted(glob.glob("*.py"))
    if py_files:
        return py_files
    # --validate runs may stage the reference solution under its own name.
    refs, seen = [], set()
    for f in sorted(glob.glob("reference-*")) + sorted(glob.glob("*agent*")):
        if f not in seen:
            seen.add(f)
            refs.append(f)
    if refs:
        return refs
    return []


def load_sources():
    files = find_sources()
    if not files:
        return "", [], "No agent.py (or other Python artifact) found in the workspace."
    parts = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())
    return "\n\n".join(parts), files, None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def imported(tree, name):
    """True if `name` is imported (possibly aliased) from any module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == name or (a.asname or a.name) == name:
                    return True
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == name or a.name.split(".")[0] == name:
                    return True
    return False


def name_used(tree, name):
    """True if `name` appears as an identifier or attribute anywhere."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def is_tool_decorator(deco):
    if isinstance(deco, ast.Name):
        return deco.id == "tool"
    if isinstance(deco, ast.Attribute):
        return deco.attr == "tool"
    if isinstance(deco, ast.Call):
        return is_tool_decorator(deco.func)
    return False


def tool_functions(tree):
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(is_tool_decorator(d) for d in n.decorator_list)
    ]


def tool_params(fn):
    return list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)


def fully_typed(fn):
    params = [a for a in tool_params(fn) if a.arg not in ("self", "cls")]
    return fn.returns is not None and all(a.annotation is not None for a in params)


def meaningful_docstring(fn):
    doc = ast.get_docstring(fn)
    return bool(doc and len(doc.strip()) >= 5)


def dataclasses(tree):
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decos = node.decorator_list
        is_dc = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in decos
        )
        if is_dc:
            out[node.name] = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    return out


def call_kwargs(tree, func_name):
    """Keyword args of every call to func_name (plain name or attribute)."""
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        hit = (isinstance(f, ast.Name) and f.id == func_name) or (
            isinstance(f, ast.Attribute) and f.attr == func_name
        )
        if hit:
            results.append({k.arg: k for k in node.keywords if k.arg})
    return results


def invoke_calls(tree):
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("invoke", "ainvoke")
    ]


def subscript_context(ann):
    """True if annotation is ToolRuntime[Context] (handles ast.Index wrapper)."""
    if not isinstance(ann, ast.Subscript):
        return False
    value_ok = (
        (isinstance(ann.value, ast.Name) and ann.value.id == "ToolRuntime")
        or (isinstance(ann.value, ast.Attribute) and ann.value.attr == "ToolRuntime")
    )
    if not value_ok:
        return False
    sl = ann.slice
    if isinstance(sl, ast.Index):  # Python 3.8
        sl = sl.value
    return isinstance(sl, ast.Name) and sl.id == "Context"


def source_segment(src, node):
    seg = ast.get_source_segment(src, node)
    if seg is not None:
        return seg
    lines = src.splitlines()
    start = lines[node.lineno - 1]
    if node.lineno == node.end_lineno:
        return start[node.col_offset:node.end_col_offset]
    return start[node.col_offset:]  # good enough for a substring probe


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_checks(src, tree):
    checks = []
    found = lambda n: imported(tree, n) or name_used(tree, n)

    # 1-6: required current-API imports
    for name in ("create_agent", "init_chat_model", "tool", "ToolRuntime",
                 "InMemorySaver", "ToolStrategy"):
        ok = found(name)
        checks.append({
            "name": "imports-" + name,
            "passed": ok,
            "message": ("imported or referenced " + name) if ok else ("missing " + name),
        })

    # 7-8: tools documented and typed
    tools = tool_functions(tree)
    with_docs = [t for t in tools if meaningful_docstring(t)]
    typed = [t for t in tools if fully_typed(t)]
    checks.append({
        "name": "tools-documented",
        "passed": len(with_docs) >= 2,
        "message": "{}/{} @tool functions have real docstrings (need >= 2)".format(
            len(with_docs), len(tools)),
    })
    checks.append({
        "name": "tools-typed",
        "passed": len(typed) >= 2,
        "message": "{}/{} @tool functions have full type hints (need >= 2)".format(
            len(typed), len(tools)),
    })

    # 9: ToolRuntime[Context] plumbing + Context dataclass with user_id
    dcs = dataclasses(tree)
    ctx_fields = dcs.get("Context", [])
    tr_ctx_tool = any(
        subscript_context(a.annotation)
        for t in tools for a in tool_params(t) if a.annotation is not None
    )
    checks.append({
        "name": "toolruntime-context",
        "passed": tr_ctx_tool and "user_id" in ctx_fields,
        "message": (
            "@tool using ToolRuntime[Context] with @dataclass Context(user_id) found"
            if tr_ctx_tool and "user_id" in ctx_fields
            else (
                "@dataclass Context(user_id) found but no @tool takes ToolRuntime[Context]"
                if "user_id" in ctx_fields
                else "missing @tool with ToolRuntime[Context] and/or @dataclass Context with user_id"
            )
        ),
    })

    # create_agent wiring
    agent_calls = call_kwargs(tree, "create_agent")
    has_kw = lambda kw: any(kw in k for k in agent_calls)

    checks.append({
        "name": "checkpointer-wired",
        "passed": has_kw("checkpointer"),
        "message": "create_agent(..., checkpointer=...) found"
        if has_kw("checkpointer") else "create_agent is missing the checkpointer= argument",
    })
    checks.append({
        "name": "context-schema-wired",
        "passed": has_kw("context_schema"),
        "message": "create_agent(..., context_schema=...) found"
        if has_kw("context_schema") else "create_agent is missing context_schema= (needed for ToolRuntime[Context])",
    })

    resp_fields = dcs.get("Response", [])
    so_ok = has_kw("response_format") and len(resp_fields) >= 1
    checks.append({
        "name": "structured-output",
        "passed": so_ok,
        "message": "create_agent(response_format=ToolStrategy(Response)) + @dataclass Response found"
        if so_ok else "missing response_format=ToolStrategy(Response) and/or a @dataclass Response with fields",
    })

    # example invocation: thread_id + context instance
    invokes = invoke_calls(tree)
    has_config = any(any(k.arg == "config" for k in n.keywords if k.arg) for n in invokes)
    has_context = any(any(k.arg == "context" for k in n.keywords if k.arg) for n in invokes)
    thread_in_invoke = any(
        "thread_id" in source_segment(src, n) for n in invokes
    )
    thread_ok = thread_in_invoke or (has_config and "thread_id" in src)
    checks.append({
        "name": "invoke-thread-id",
        "passed": thread_ok,
        "message": "example invoke passes a configurable thread_id" if thread_ok
        else "example invoke is missing a thread_id in config",
    })
    checks.append({
        "name": "invoke-context",
        "passed": has_context,
        "message": "example invoke passes context=Context(...)" if has_context
        else "example invoke is missing a context= instance",
    })

    # negative: no deprecated chains
    deprecated = [t for t in ("LLMChain", "SequentialChain") if t in src]
    checks.append({
        "name": "no-deprecated-chains",
        "passed": not deprecated,
        "message": "no deprecated LLMChain/SequentialChain" if not deprecated
        else "deprecated API referenced: " + ", ".join(deprecated),
    })

    return checks


def main():
    src, files, err = load_sources()
    report = {"score": 0.0, "details": "", "checks": []}
    if err:
        report["details"] = err
        print(json.dumps(report))
        sys.exit(1)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        report["details"] = "artifact could not be parsed: {} (line {})".format(
            exc.msg, exc.lineno)
        print(json.dumps(report))
        sys.exit(1)

    checks = run_checks(src, tree)
    passed = sum(1 for c in checks if c["passed"])
    score = round(passed / len(checks), 4) if checks else 0.0
    report["score"] = score
    report["details"] = "{}/{} checks passed ({})".format(passed, len(checks), ", ".join(files))
    report["checks"] = checks
    print(json.dumps(report))
    sys.exit(0 if score >= PASS_BAR else 1)


if __name__ == "__main__":
    main()
