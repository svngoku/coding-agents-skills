#!/usr/bin/env python3
"""Deterministic grader for the smolagents skill task: financial-analysis-codeagent.

The agent must produce a self-contained module `agent.py` that builds a smolagents
CodeAgent. This grader statically checks the produced artifact with the `ast` module
(never executes it, no network, Python stdlib only). It expects to run from the
workspace directory where the agent wrote its output.

Checks (all binary; score = passed / total):
  1. smolagents-imports             CodeAgent, tool and a model class imported
  2. tool-docstring-and-type-hints  at least one @tool fn with docstring + full type hints
  3. model-explicitly-configured    model built with concrete model_id/provider, wired to CodeAgent
  4. max-steps-15                   CodeAgent(max_steps=15)
  5. planning-interval-3            CodeAgent(planning_interval=3)
  6. verbosity-level-set            CodeAgent(verbosity_level >= 1)
  7. restricted-import-whitelist    additional_authorized_imports: 1..6 explicit names, no wildcards
  8. run-example                    an agent.run("<prompt>") example call exists

Prints JSON {"score": float, "details": str, "checks": [...]} to stdout.
Exits 0 when score >= PASS_BAR, else 1.
"""

import ast
import glob
import json
import sys

PASS_BAR = 0.8
MODEL_CLASSES = ("InferenceClientModel", "LiteLLMModel")
CHECK_NAMES = (
    "smolagents-imports",
    "tool-docstring-and-type-hints",
    "model-explicitly-configured",
    "max-steps-15",
    "planning-interval-3",
    "verbosity-level-set",
    "restricted-import-whitelist",
    "run-example",
)


def load_source() -> str:
    """Read the produced module from the workspace: agent.py first, then any *.py,
    then a reference-* file (so --validate against the reference solution works)."""
    if glob.glob("agent.py"):
        with open("agent.py", encoding="utf-8") as fh:
            return fh.read()
    py_files = sorted(glob.glob("*.py"))
    if py_files:
        return "\n".join(open(path, encoding="utf-8").read() for path in py_files)
    refs = sorted(glob.glob("reference-*"))
    if refs:
        with open(refs[0], encoding="utf-8") as fh:
            return fh.read()
    return ""


def tool_functions(tree):
    """Yield FunctionDef nodes decorated with @tool (handles smolagents.tool too)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            name = ""
            if isinstance(dec, ast.Name):
                name = dec.id
            elif isinstance(dec, ast.Attribute):
                name = dec.attr
            if name == "tool":
                yield node
                break


def has_docstring(fn: ast.FunctionDef) -> bool:
    if not fn.body:
        return False
    first = fn.body[0]
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and bool(first.value.value.strip())
    )


def has_full_type_hints(fn: ast.FunctionDef) -> bool:
    args = fn.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if not params:
        return False
    return all(p.annotation is not None for p in params) and fn.returns is not None


def codeagent_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CodeAgent"
        ):
            yield node


def model_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in MODEL_CLASSES
        ):
            yield node


def primary_agent_call(tree):
    """Prefer the CodeAgent(...) call assigned to `agent`; else the last one."""
    calls = list(codeagent_calls(tree))
    if not calls:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if (
                isinstance(node.value.func, ast.Name)
                and node.value.func.id == "CodeAgent"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "agent":
                        return node.value
    return calls[-1]


def get_kwarg(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def resolve_list(tree, value):
    """Resolve a list literal, or a name bound to one, to the ast.List node."""
    if isinstance(value, ast.List):
        return value
    if isinstance(value, ast.Name):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == value.id:
                        return node.value
    return None


def has_explicit_model_id(call: ast.Call) -> bool:
    v = get_kwarg(call, "model_id")
    if (
        v is not None
        and isinstance(v, ast.Constant)
        and isinstance(v.value, str)
        and v.value.strip()
    ):
        return True
    return get_kwarg(call, "provider") is not None


def emit(checks) -> int:
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    score = passed / total if total else 0.0
    print(
        json.dumps(
            {
                "score": round(score, 4),
                "details": f"{passed}/{total} checks passed",
                "checks": checks,
            }
        )
    )
    return 0 if score >= PASS_BAR else 1


def main() -> int:
    src = load_source()
    if not src.strip():
        checks = [
            {"name": name, "passed": False, "message": "no agent.py found in workspace"}
            for name in CHECK_NAMES
        ]
        return emit(checks)

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        checks = [
            {
                "name": name,
                "passed": False,
                "message": f"module is not valid Python: {exc.msg}",
            }
            for name in CHECK_NAMES
        ]
        return emit(checks)

    checks = []

    # 1. imports: CodeAgent, tool and a model class from smolagents
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "smolagents":
            imported.update(alias.name for alias in node.names)
    missing = sorted({"CodeAgent", "tool"} - imported)
    model_class = imported & set(MODEL_CLASSES)
    ok = not missing and bool(model_class)
    checks.append(
        {
            "name": "smolagents-imports",
            "passed": ok,
            "message": (
                f"imported CodeAgent, tool, {sorted(model_class)[0]} from smolagents"
                if ok
                else "missing imports: "
                + ", ".join(missing + sorted(set(MODEL_CLASSES) - model_class))
            ),
        }
    )

    # 2. at least one @tool function with docstring + full type hints
    tools = list(tool_functions(tree))
    good_tools = [f for f in tools if has_docstring(f) and has_full_type_hints(f)]
    ok = bool(good_tools)
    checks.append(
        {
            "name": "tool-docstring-and-type-hints",
            "passed": ok,
            "message": (
                f"{len(good_tools)} @tool function(s) with docstring and full type hints"
                if ok
                else f"{len(tools)} @tool function(s) found, none with docstring AND param/return type hints"
            ),
        }
    )

    # 3. model explicitly configured (model_id/provider) and wired into the agent
    agent_call = primary_agent_call(tree)
    models = list(model_calls(tree))
    explicit = [c for c in models if has_explicit_model_id(c)]
    if not explicit:
        msg = "no InferenceClientModel/LiteLLMModel call with explicit model_id/provider found"
    elif agent_call is None:
        msg = "no CodeAgent(...) construction found"
    elif get_kwarg(agent_call, "model") is None:
        msg = "model built but not passed to CodeAgent via model="
    else:
        msg = "model built with explicit model_id/provider and passed to CodeAgent"
    checks.append(
        {
            "name": "model-explicitly-configured",
            "passed": bool(explicit) and agent_call is not None and get_kwarg(agent_call, "model") is not None,
            "message": msg,
        }
    )

    def const_int(call, name):
        v = get_kwarg(call, name) if call is not None else None
        if isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool):
            return v.value
        return None

    # 4. max_steps=15
    max_steps = const_int(agent_call, "max_steps")
    checks.append(
        {
            "name": "max-steps-15",
            "passed": max_steps == 15,
            "message": (
                f"max_steps={max_steps}" if max_steps is not None else "max_steps not set"
            ),
        }
    )

    # 5. planning_interval=3
    planning_interval = const_int(agent_call, "planning_interval")
    checks.append(
        {
            "name": "planning-interval-3",
            "passed": planning_interval == 3,
            "message": (
                f"planning_interval={planning_interval}"
                if planning_interval is not None
                else "planning_interval not set"
            ),
        }
    )

    # 6. verbosity_level explicitly set (non-silent)
    verbosity = const_int(agent_call, "verbosity_level")
    ok = verbosity is not None and verbosity >= 1
    checks.append(
        {
            "name": "verbosity-level-set",
            "passed": ok,
            "message": (
                f"verbosity_level={verbosity}"
                if verbosity is not None
                else "verbosity_level not set (0 is silent)"
            ),
        }
    )

    # 7. additional_authorized_imports: small explicit whitelist, no wildcards
    v = (
        get_kwarg(agent_call, "additional_authorized_imports")
        if agent_call is not None
        else None
    )
    lst = resolve_list(tree, v) if v is not None else None
    if lst is None:
        ok = False
        message = "additional_authorized_imports missing or not a list literal"
    else:
        names = [
            e.value
            for e in lst.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        ok = (
            1 <= len(names) <= 6
            and len(names) == len(lst.elts)
            and all(n != "*" for n in names)
        )
        message = (
            f"additional_authorized_imports={names}"
            if ok
            else (
                f"additional_authorized_imports={names} too broad: wildcards or >6 entries"
                if names
                else "additional_authorized_imports empty or non-string entries"
            )
        )
    checks.append(
        {
            "name": "restricted-import-whitelist",
            "passed": ok,
            "message": message,
        }
    )

    # 8. agent.run(...) example with a string prompt
    ok = False
    message = "no agent.run(...) example call found"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.strip()
        ):
            ok = True
            message = "agent.run('<prompt>') example present"
            break
    checks.append(
        {
            "name": "run-example",
            "passed": ok,
            "message": message,
        }
    )

    return emit(checks)


if __name__ == "__main__":
    sys.exit(main())
