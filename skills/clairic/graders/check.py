"""Hermetic invoice grader: a narrow AST profile plus synthetic pure-helper outcomes.

Never import candidate code. Only accept_total's whitelisted body is executed, without
annotations/decorators/defaults, with empty builtins and a single numeric predicate.
This is not a general Python sandbox or evidence of live SDK/model correctness.
"""

from __future__ import annotations

import ast
import copy
from itertools import product
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MAX_BYTES = 65_536
IMPORTS = {
    "__future__": {"annotations"},
    "datetime": {"date"},
    "math": {"isfinite"},
    "typing": {"Annotated"},
    "pydantic": {"BaseModel", "Field"},
    "clairic": {"Clairic", "DateAnswer", "Number", "NumberAnswer", "Result", "TextAnswer"},
}
REQUIRED = {"date", "Annotated", "BaseModel", "Clairic", "DateAnswer", "Number",
            "NumberAnswer", "TextAnswer"}


def docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)


def statements(body: list[ast.stmt]) -> list[ast.stmt]:
    return [node for node in body if not docstring(node)]


def named(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def literal(node: ast.AST, value: Any) -> bool:
    # bool is an int subclass but not a valid substitute for a date component/decimals.
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def keywords(call: ast.Call, expected: set[str]) -> dict[str, ast.AST] | None:
    keys = [keyword.arg for keyword in call.keywords]
    if None in keys or len(set(keys)) != len(keys) or set(keys) != expected:
        return None
    return {keyword.arg: keyword.value for keyword in call.keywords}


def signature(node: ast.FunctionDef, parameter: str) -> bool:
    args = node.args
    positional = args.posonlyargs + args.args
    return (
        len(positional) == 1 and positional[0].arg == parameter
        and not args.defaults and not args.kwonlyargs and not args.kw_defaults
        and args.vararg is None and args.kwarg is None and not node.decorator_list
        and not getattr(node, "type_params", [])
    )


def safe_annotation(node: ast.AST | None) -> bool:
    if node is None or isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in {"str", "float", "Result", "InvoiceExtraction"}
    if isinstance(node, ast.Subscript):
        return named(node.value, "Result") and named(node.slice, "InvoiceExtraction")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return safe_annotation(node.left) and safe_annotation(node.right)
    return False


def module_profile(tree: ast.Module) -> bool:
    imports: set[str] = set()
    classes = []
    functions = []
    for node in tree.body:
        if docstring(node):
            continue
        if isinstance(node, ast.ImportFrom):
            if classes or functions:
                return False
            if node.level or node.module not in IMPORTS:
                return False
            for item in node.names:
                if item.asname or item.name not in IMPORTS[node.module]:
                    return False
                imports.add(item.name)
        elif isinstance(node, ast.Import):
            if classes or functions:
                return False
            if any(item.name != "math" or item.asname for item in node.names):
                return False
            imports.add("math")
        elif isinstance(node, ast.ClassDef):
            if functions:
                return False
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            if classes != ["InvoiceExtraction"]:
                return False
            functions.append(node.name)
            parameter = {"extract_invoice": "text", "accept_total": "result"}.get(node.name)
            if parameter is None or not signature(node, parameter):
                return False
            annotations = [arg.annotation for arg in node.args.posonlyargs + node.args.args]
            if not all(safe_annotation(a) for a in [node.returns, *annotations]):
                return False
        else:
            return False
    return (
        REQUIRED <= imports and classes == ["InvoiceExtraction"]
        and sorted(functions) == ["accept_total", "extract_invoice"]
    )


def field_description(value: ast.AST | None) -> bool:
    if value is None:
        return True
    if not isinstance(value, ast.Call) or not named(value.func, "Field") or value.args:
        return False
    options = keywords(value, {"description"})
    return bool(
        options is not None and isinstance(options["description"], ast.Constant)
        and isinstance(options["description"].value, str)
        and options["description"].value.strip()
    )


def invoice_schema(tree: ast.Module) -> bool:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        return False
    model = classes[0]
    if (
        model.name != "InvoiceExtraction" or len(model.bases) != 1
        or not named(model.bases[0], "BaseModel") or model.keywords or model.decorator_list
        or getattr(model, "type_params", [])
    ):
        return False
    fields: dict[str, ast.AnnAssign] = {}
    for node in statements(model.body):
        if (
            not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name)
            or node.simple != 1 or node.target.id in fields or not field_description(node.value)
        ):
            return False
        fields[node.target.id] = node
    if set(fields) != {"invoice_number", "total", "due_date"}:
        return False
    if not named(fields["invoice_number"].annotation, "TextAnswer"):
        return False
    if not named(fields["due_date"].annotation, "DateAnswer"):
        return False
    total = fields["total"].annotation
    if not isinstance(total, ast.Subscript) or not named(total.value, "Annotated"):
        return False
    if not isinstance(total.slice, ast.Tuple) or len(total.slice.elts) != 2:
        return False
    answer, spec = total.slice.elts
    if not (
        isinstance(answer, ast.Subscript) and named(answer.value, "NumberAnswer")
        and named(answer.slice, "float") and isinstance(spec, ast.Call)
        and named(spec.func, "Number") and not spec.args
    ):
        return False
    options = keywords(spec, {"unit", "decimals"})
    return bool(
        options is not None and literal(options["unit"], "USD")
        and literal(options["decimals"], 2)
    )


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    return matches[0] if len(matches) == 1 else None


def live_contract(tree: ast.Module) -> bool:
    function = find_function(tree, "extract_invoice")
    if function is None or not signature(function, "text"):
        return False
    body = statements(function.body)
    if len(body) != 1 or not isinstance(body[0], ast.With):
        return False
    context = body[0]
    if len(context.items) != 1:
        return False
    item = context.items[0]
    constructor = item.context_expr
    if (
        not isinstance(item.optional_vars, ast.Name)
        or item.optional_vars.id.startswith("_")
        or item.optional_vars.id in REQUIRED | {"text", "InvoiceExtraction", "Result", "Field", "math", "isfinite"}
        or not isinstance(constructor, ast.Call) or not named(constructor.func, "Clairic")
        or constructor.args
    ):
        return False
    options = keywords(constructor, {"timeout"})
    if options is None or not (
        literal(options["timeout"], 30.0) or literal(options["timeout"], 30)
    ):
        return False
    body = statements(context.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    call = body[0].value
    if not (
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and named(call.func.value, item.optional_vars.id) and call.func.attr == "extract"
        and len(call.args) == 2 and named(call.args[0], "text")
        and named(call.args[1], "InvoiceExtraction")
    ):
        return False
    options = keywords(call, {"reference_date"})
    if options is None:
        return False
    anchor = options["reference_date"]
    return (
        isinstance(anchor, ast.Call) and named(anchor.func, "date")
        and not anchor.keywords and len(anchor.args) == 3
        and all(literal(arg, value) for arg, value in zip(anchor.args, (2026, 10, 26)))
    )


def restricted_helper(function: ast.FunctionDef):
    """Compile only a finite, non-mutating expression/branch subset, never the module.

    Attribute names are closed, inputs are grader-owned SimpleNamespaces/floats, and
    call targets are fixed. No loops, imports, subscripts, arithmetic, indirect calls,
    exception handling, mutation or dunder access are admitted.
    """
    args = function.args
    if (
        len(args.posonlyargs + args.args) != 1
        or (args.posonlyargs + args.args)[0].arg != "result"
        or args.vararg or args.kwarg or args.kwonlyargs
    ):
        raise ValueError("accept_total must have one result parameter")
    body_tree = ast.Module(body=function.body, type_ignores=[])
    nodes = list(ast.walk(body_tree))
    if len(nodes) > 160:
        raise ValueError("Pure helper exceeds the bounded AST profile")
    locals_ = set()
    for node in nodes:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Only simple local-name assignments are allowed")
            name = node.targets[0].id
            if name.startswith("_") or name in {"result", "math", "isfinite"}:
                raise ValueError("Cannot rebind inputs or permitted numeric predicate")
            locals_.add(name)
    allowed_nodes = (
        ast.Module, ast.Assign, ast.If, ast.Return, ast.Expr,
        ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Constant,
        ast.Compare, ast.BoolOp, ast.UnaryOp, ast.Call,
        ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        ast.Is, ast.IsNot, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    attrs = {"truncated", "data", "total", "value", "p_null", "confidence"}
    for node in nodes:
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported pure-helper syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"result", "math", "isfinite", *locals_}:
            raise ValueError("Unbound or forbidden pure-helper name")
        if isinstance(node, ast.Attribute):
            numeric_predicate = named(node.value, "math") and node.attr == "isfinite"
            if not numeric_predicate and node.attr not in attrs:
                raise ValueError("Forbidden attribute (including guess and all dunders)")
        if isinstance(node, ast.Constant) and (
            type(node.value) not in {type(None), bool, int, float, str}
            or isinstance(node.value, str) and len(node.value) > 400
        ):
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.Expr) and not docstring(node):
            raise ValueError("Expression statements must be docstrings")
        if isinstance(node, ast.Call):
            permitted = named(node.func, "isfinite") or (
                isinstance(node.func, ast.Attribute)
                and named(node.func.value, "math") and node.func.attr == "isfinite"
            )
            if not permitted or len(node.args) != 1 or node.keywords:
                raise ValueError("Only isfinite(value) or math.isfinite(value) is callable")

    # Rebuild signature rather than executing any candidate annotations/defaults/decorators.
    clean = copy.deepcopy(function)
    clean.decorator_list = []
    clean.returns = None
    clean.type_comment = None
    clean.args = ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="result")], kwonlyargs=[], kw_defaults=[], defaults=[]
    )
    if hasattr(clean, "type_params"):
        clean.type_params = []
    program = ast.fix_missing_locations(ast.Module(body=[clean], type_ignores=[]))
    namespace = {
        "__builtins__": {},
        "isfinite": math.isfinite,
        "math": SimpleNamespace(isfinite=math.isfinite),
    }
    exec(compile(program, "<restricted accept_total>", "exec"), namespace)
    return namespace["accept_total"]


def helper_outcomes(tree: ast.Module) -> tuple[bool, str]:
    function = find_function(tree, "accept_total")
    if function is None:
        return False, "Missing or duplicate accept_total"
    try:
        helper = restricted_helper(function)
        count = 0
        for truncated, value, p_null, confidence in product(
            (False, True),
            (None, 0.0, 0.01, 12.34, 1e15, -0.01, -300.0, math.nan, math.inf, -math.inf),
            (0.0, 0.499999, 0.5, 0.99),
            (0.0, 0.899999, 0.9, 1.0),
        ):
            total = SimpleNamespace(value=value, p_null=p_null, confidence=confidence, guess=999.0)
            result = SimpleNamespace(truncated=truncated, data=SimpleNamespace(total=total))
            expected = None
            if (
                not truncated and value is not None and p_null < 0.5 and confidence >= 0.9
                and math.isfinite(value) and value >= 0
            ):
                expected = value
            actual = helper(result)
            if expected is None:
                correct = actual is None
            else:
                correct = type(actual) is float and actual == expected
            if not correct:
                return False, (
                    "Synthetic mismatch: "
                    f"truncated={truncated}, value={value}, p_null={p_null}, confidence={confidence}"
                )
            count += 1
    except Exception as error:
        return False, f"Restricted helper rejected/failed: {type(error).__name__}: {error}"
    return True, f"{count} synthetic boundary/absence/truncation/uncertainty cases passed"


def read_artifact(root: Path, name: str) -> str:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular, non-symlink file")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{name} exceeds {MAX_BYTES} bytes")
    return data.decode("utf-8")


def grade(root: Path) -> dict[str, Any]:
    try:
        source = read_artifact(root, "invoice.py")
        requirements = read_artifact(root, "requirements.txt")
        design = read_artifact(root, "design.md")
        tree = ast.parse(source, filename="invoice.py")
        # Validate Python semantics without executing imports or any candidate code.
        compile(tree, "invoice.py", "exec")
        if len(list(ast.walk(tree))) > 1200:
            raise ValueError("invoice.py exceeds the bounded AST profile")
    except (OSError, UnicodeError, ValueError, SyntaxError, RecursionError) as error:
        return {"score": 0.0, "details": f"Missing/malformed artifact: {type(error).__name__}: {error}"}

    pins = [line.split("#", 1)[0].strip() for line in requirements.splitlines()]
    pins = [line for line in pins if line]
    imports = {
        item.name for node in tree.body if isinstance(node, ast.ImportFrom)
        for item in node.names
    }
    # Ensure optional Field and finiteness names are genuinely imported, not just spelled.
    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    helper_ok, helper_details = helper_outcomes(tree)
    checks = [
        {
            "name": "Artifacts, version pin and import-time module boundary",
            "passed": pins == ["clairic==0.1.0"] and bool(design.strip()) and module_profile(tree)
            and ("Field" not in used_names or "Field" in imports)
            and ("Result" not in used_names or "Result" in imports)
            and ("isfinite" not in used_names or "isfinite" in imports)
            and ("math" not in used_names or any(
                isinstance(node, ast.Import) and any(item.name == "math" for item in node.names)
                for node in tree.body
            )),
        },
        {"name": "Required typed invoice schema", "passed": invoice_schema(tree)},
        {"name": "Function-only supported SDK lifecycle and date anchor", "passed": live_contract(tree)},
        {"name": "Restricted pure-helper outcomes", "passed": helper_ok, "details": helper_details},
    ]
    passed = sum(check["passed"] for check in checks)
    # All four groups are essential. Partial credit must not pass at threshold 0.8:
    # a deterministic failure yields 0, so even a perfect 0.3-weight rubric cannot pass.
    return {
        "score": 1.0 if passed == len(checks) else 0.0,
        "details": f"{passed}/{len(checks)} essential groups passed; design semantics use the rubric",
        "checks": checks,
    }


if __name__ == "__main__":
    result = grade(Path.cwd())
    print(json.dumps(result))
    raise SystemExit(0 if result["score"] >= 0.8 else 1)
