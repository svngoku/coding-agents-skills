#!/usr/bin/env python3
"""Hermetic static checks for the two Mastra support-agent artifacts."""

import json
import re
import stat
from pathlib import Path

CODE_PATH = "src/mastra/index.ts"
DESIGN_PATH = "design.md"
CODE_LIMIT = 128 * 1024
DESIGN_LIMIT = 64 * 1024
PASS_BAR = 0.8


def artifact(root, name, limit):
    """Read a bounded UTF-8 regular file; never follow a symlink."""
    path = root / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink")
    if info.st_size > limit:
        raise ValueError(f"{name} exceeds the {limit}-byte limit")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError(f"{name} exceeds the {limit}-byte limit")
    text = raw.decode("utf-8")
    if "\x00" in text:
        raise ValueError(f"{name} contains a NUL byte")
    return text


def strip_ts_comments(source):
    """Blank comments while preserving every string/template literal and newline."""
    out = []
    index = 0
    quote = None
    escaped = False
    while index < len(source):
        char = source[index]
        pair = source[index:index + 2]
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in ("'", '"', "`"):
            quote = char
            out.append(char)
            index += 1
        elif pair == "//":
            while index < len(source) and source[index] != "\n":
                out.append(" ")
                index += 1
        elif pair == "/*":
            out.extend((" ", " "))
            index += 2
            while index < len(source) and source[index:index + 2] != "*/":
                out.append("\n" if source[index] == "\n" else " ")
                index += 1
            if index < len(source):
                out.extend((" ", " "))
                index += 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


def matching(source, start, opening, closing):
    """Find a delimiter's mate, ignoring delimiters inside string literals."""
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def calls(source, expression):
    """Return argument text for calls selected by a regular expression."""
    found = []
    for match in re.finditer(expression, source):
        start = source.rfind("(", match.start(), match.end())
        if start < 0:
            continue
        end = matching(source, start, "(", ")")
        if end is not None:
            found.append(source[start + 1:end])
    return found


def split_top_level(source):
    parts = []
    start = 0
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == "," and not stack:
            parts.append(source[start:index])
            start = index + 1
    parts.append(source[start:])
    return parts


def top_level_colon(source):
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"', "`"):
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == ":" and not stack:
            return index
    return None


def fields(call_source):
    """Parse simple top-level key/value fields from the first object argument."""
    start = call_source.find("{")
    if start < 0:
        return {}
    end = matching(call_source, start, "{", "}")
    if end is None:
        return {}
    result = {}
    for part in split_top_level(call_source[start + 1:end]):
        colon = top_level_colon(part)
        if colon is None:
            shorthand = part.strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", shorthand):
                result[shorthand] = shorthand
            continue
        key = part[:colon].strip().strip("'\"")
        if re.fullmatch(r"[A-Za-z_$][\w$]*", key):
            result[key] = part[colon + 1:].strip()
    return result


def imported(source, names, module):
    pattern = r"\bimport\s*\{([^}]*)\}\s*from\s*['\"]" + re.escape(module) + r"['\"]"
    for match in re.finditer(pattern, source, re.DOTALL):
        imported_names = set(re.findall(r"\b[A-Za-z_$][\w$]*\b", match.group(1)))
        if all(name in imported_names for name in names):
            return True
    return False


def zod_object(value, source):
    if not value:
        return False
    if re.search(r"\bz\s*\.\s*object\s*\(", value):
        return True
    name = value.strip()
    if re.fullmatch(r"[A-Za-z_$][\w$]*", name):
        definition = r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*z\s*\.\s*object\s*\("
        return bool(re.search(definition, source))
    return False


def valid_tool(call_source, source):
    config = fields(call_source)
    execute = config.get("execute", "")
    signature = re.match(
        r"^async\s*\(\s*inputData\s*,\s*(?:context|\{[^{}]*\})\s*\)", execute
    )
    return bool(
        signature
        and zod_object(config.get("inputSchema", ""), source)
        and zod_object(config.get("outputSchema", ""), source)
    )


def current_agent_api(source):
    imports_ok = all((
        imported(source, ("Agent",), "@mastra/core/agent"),
        imported(source, ("createTool",), "@mastra/core/tools"),
        imported(source, ("createWorkflow", "createStep"), "@mastra/core/workflows"),
        imported(source, ("z",), "zod"),
    ))
    agents = [fields(call) for call in calls(source, r"\bnew\s+Agent\s*\(")]
    for config in agents:
        model = config.get("model", "").strip()
        model_ok = bool(re.fullmatch(
            r"['\"][A-Za-z0-9_.-]+/[A-Za-z0-9][A-Za-z0-9_.:-]*['\"]", model
        ))
        structured_configs = [config.get("structuredOutput", "")]
        structured_configs.extend(
            fields(call).get("structuredOutput", "")
            for call in calls(source, r"\.\s*generate\s*\(")
        )
        schemas = [fields(value).get("schema", "") for value in structured_configs]
        if model_ok and any(zod_object(schema, source) for schema in schemas):
            return imports_ok
    return False


def validated_tools(source):
    tool_calls = calls(source, r"\bcreateTool\s*\(")
    return len(tool_calls) >= 2 and sum(valid_tool(call, source) for call in tool_calls) >= 2


def refund_safety(source):
    for call in calls(source, r"\bcreateTool\s*\("):
        config = fields(call)
        if config.get("requireApproval") != "true" or "refund" not in call.lower():
            continue
        execute = config.get("execute", "")
        actor = re.search(
            r"\bcontext\s*\.\s*requestContext\s*\.\s*get\s*\(\s*['\"][^'\"]*actor[^'\"]*['\"]",
            execute, re.IGNORECASE,
        )
        tenant = re.search(
            r"\bcontext\s*\.\s*requestContext\s*\.\s*get\s*\(\s*['\"][^'\"]*tenant[^'\"]*['\"]",
            execute, re.IGNORECASE,
        )
        client_identity = re.search(
            r"\binputData\s*\.\s*(?:actor|actorId|tenant|tenantId)\b", execute
        )
        if (actor and tenant and not client_identity
                and re.search(r"authoriz", execute, re.IGNORECASE)
                and re.search(r"idempoten", execute, re.IGNORECASE)):
            return valid_tool(call, source)
    return False


def workflow_lifecycle(source):
    workflows = [fields(call) for call in calls(source, r"\bcreateWorkflow\s*\(")]
    steps = [fields(call) for call in calls(source, r"\bcreateStep\s*\(")]
    typed_workflow = any(
        zod_object(config.get("inputSchema", ""), source)
        and zod_object(config.get("outputSchema", ""), source)
        for config in workflows
    )
    typed_step = any(
        zod_object(config.get("inputSchema", ""), source)
        and zod_object(config.get("outputSchema", ""), source)
        and "inputData" in config.get("execute", "")
        for config in steps
    )
    composed = bool(re.search(
        r"\)\s*\.\s*then\s*\(\s*[A-Za-z_$][\w$]*\s*\)\s*\.\s*commit\s*\(\s*\)",
        source, re.DOTALL,
    ))
    run_match = re.search(
        r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*await\s+"
        r"[^;\n]+?\.\s*createRun\s*\(\s*\)", source
    )
    if not run_match:
        return False
    run_name = run_match.group(1)
    start_match = re.search(
        r"\b(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*await\s+"
        + re.escape(run_name) + r"\s*\.\s*start\s*\(\s*\{", source
    )
    if not start_match:
        return False
    result_name = start_match.group(1)
    status = re.escape(result_name) + r"\s*\.\s*status"
    statuses = all(re.search(status + r"\s*===\s*['\"]" + state + r"['\"]", source)
                   for state in ("success", "failed", "suspended"))
    first_status = re.search(status, source)
    before_fields = bool(first_status)
    for field in ("result", "error"):
        access = re.search(re.escape(result_name) + r"\s*\.\s*" + field + r"\b", source)
        if access and (not first_status or access.start() < first_status.start()):
            before_fields = False
    start_args = calls(source, re.escape(run_name) + r"\s*\.\s*start\s*\(")
    return bool(typed_workflow and typed_step and composed and statuses and before_fields
                and any("inputData" in fields(call) or re.match(r"\s*inputData\s*,", call)
                        for call in start_args))


def bounded_execution(source):
    values = [int(value) for value in re.findall(r"\bmaxSteps\s*:\s*(\d+)", source)]
    return any(1 <= value <= 8 for value in values)


def persistent_memory(source):
    imports_ok = (imported(source, ("Memory",), "@mastra/memory")
                  and imported(source, ("LibSQLStore",), "@mastra/libsql"))
    stores = [fields(call) for call in calls(source, r"\bnew\s+LibSQLStore\s*\(")]
    persistent_store = any(
        config.get("url") and ":memory:" not in config.get("url", "")
        and re.search(r"file:|process\s*\.\s*env", config.get("url", ""))
        for config in stores
    )
    memories = [fields(call) for call in calls(source, r"\bnew\s+Memory\s*\(")]
    storage_wired = any(config.get("storage") for config in memories)
    stable_identity = bool(
        re.search(r"\brequestContext\s*\.\s*get\s*\(\s*['\"]tenantId['\"]", source)
        and re.search(r"\brequestContext\s*\.\s*get\s*\(\s*['\"]threadId['\"]", source)
    )
    stable_memory = False
    for call in calls(source, r"\.\s*generate\s*\("):
        memory = fields(fields(call).get("memory", ""))
        if (memory.get("resource") and memory.get("thread")
                and "tenant" in memory.get("resource", "").lower()):
            stable_memory = True
    return bool(imports_ok and persistent_store and storage_wired and stable_identity and stable_memory)


def design_reasoning(design):
    blocks = [block for block in re.split(r"\n\s*\n", design) if len(block.split()) >= 18]
    topics = (
        (("approval",), ("bind", "bound"), ("chang",), ("new approval", "re-approv")),
        (("tenant",), ("authoriz", "actor"), ("trusted", "server"), ("model", "client")),
        (("idempoten",), ("replay", "retry", "duplicate"), ("key",)),
        (("snapshot", "checkpoint"), ("restart", "recover", "resume"), ("persist", "durable", "libsql")),
        (("maxsteps", "bounded", "step limit"), ("exhaust", "limit", "four", "4")),
        (("untrusted",), ("tool", "rag", "retriev"), ("inject", "instruction", "data")),
        (("evaluat", "eval"), ("trac", "observ"), ("metadata", "secret", "personal")),
    )
    for alternatives in topics:
        if not any(all(any(term in block.lower() for term in group) for group in alternatives)
                   for block in blocks):
            return False
    return True


def grade(root):
    try:
        raw_code = artifact(root, CODE_PATH, CODE_LIMIT)
        design = artifact(root, DESIGN_PATH, DESIGN_LIMIT)
        if len(raw_code.strip()) < 40 or len(design.strip()) < 100:
            raise ValueError("Artifacts are too short to contain the requested solution")
    except (OSError, UnicodeError, ValueError) as error:
        return {"score": 0.0, "details": str(error), "checks": []}

    code = strip_ts_comments(raw_code)
    specifications = (
        ("current-agent-api", lambda: current_agent_api(code)),
        ("validated-tools", lambda: validated_tools(code)),
        ("refund-approval-authorization-idempotency", lambda: refund_safety(code)),
        ("typed-workflow-and-status-lifecycle", lambda: workflow_lifecycle(code)),
        ("bounded-agent-execution", lambda: bounded_execution(code)),
        ("persistent-memory-and-trusted-identity", lambda: persistent_memory(code)),
        ("substantive-design-reasoning", lambda: design_reasoning(design)),
    )
    checks = []
    for name, callback in specifications:
        try:
            passed = bool(callback())
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, re.error):
            passed = False
        checks.append({"name": name, "passed": passed})
    count = sum(check["passed"] for check in checks)
    score = 1.0 if count == len(checks) else min(0.6, count / len(checks))
    return {
        "score": score,
        "details": f"{count}/{len(checks)} essential groups passed; all are required",
        "checks": checks,
    }


def main():
    try:
        result = grade(Path.cwd())
    except Exception as error:  # Malformed artifacts must still produce the JSON contract.
        result = {"score": 0.0, "details": f"Unable to grade artifacts: {error}", "checks": []}
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["score"] >= PASS_BAR else 1)


if __name__ == "__main__":
    main()
