#!/usr/bin/env python3
"""Deterministic grader for the genai-tk "langchain-agent-profile" eval task.

Runs from the agent's workspace (cwd). Checks the two required artifacts:

  config/agents/langchain.yaml   a genai-tk langchain_agents profile with a
                                 "Research" deep agent (model_id@provider llm,
                                 planning, file system, web_search tool,
                                 mcp_servers, skills.directories, memory
                                 checkpointer)
  use_agent.py                   a snippet importing LangchainAgent and
                                 constructing LangchainAgent("Research", ...)
                                 with at least one override

Python 3 standard library ONLY (no PyYAML, no network). YAML is checked with a
small hand-rolled subset parser sufficient for genai-tk profile files; the
Python file is checked with the ast module.

Prints JSON to stdout:
  {"score": 0.93, "details": "13/14 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import ast
import json
import os
import sys

YAML_PATH = os.path.join("config", "agents", "langchain.yaml")
PY_PATH = "use_agent.py"
PASS_BAR = 0.8


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
# Checks
# ---------------------------------------------------------------------------

def _find_profile(parsed, name):
    profiles = parsed.get("profiles") if isinstance(parsed, dict) else None
    if not isinstance(profiles, list):
        return None
    for p in profiles:
        if isinstance(p, dict) and p.get("name") == name:
            return p
    return None


def _tools_list(profile):
    tools = profile.get("tools") if isinstance(profile, dict) else None
    return tools if isinstance(tools, list) else []


def _langchain_calls(tree):
    """Yield Call nodes whose func resolves to LangchainAgent."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "LangchainAgent":
                yield node


def _find_import(tree):
    """True if LangchainAgent is imported from genai_tk.agents.langchain."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "genai_tk.agents.langchain" or node.module.endswith(
                ".agents.langchain"
            ):
                for alias in node.names:
                    if alias.name == "LangchainAgent":
                        return True
    return False


def _call_profile(call):
    """Return the profile name used in a LangchainAgent(...) call, or None."""
    if call.args:
        arg = call.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in call.keywords:
        if kw.arg == "profile_name" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _call_overrides(call):
    """Return the list of override keyword names present in the call."""
    return [kw.arg for kw in call.keywords if kw.arg in ("llm", "sandbox")]


def _check_python(py_path, checks):
    def add(name, passed, msg):
        checks.append({"name": name, "passed": bool(passed), "message": msg})

    if not os.path.isfile(py_path):
        add("python-syntax", False, "use_agent.py not found")
        add("import-langchain-agent", False, "use_agent.py not found")
        add("constructs-research-profile", False, "use_agent.py not found")
        return

    try:
        with open(py_path, encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        add("python-syntax", False, f"cannot read use_agent.py: {e}")
        add("import-langchain-agent", False, "cannot read use_agent.py")
        add("constructs-research-profile", False, "cannot read use_agent.py")
        return

    try:
        tree = ast.parse(src, filename=py_path)
        syntax_ok = True
        syntax_msg = "valid Python syntax"
    except SyntaxError as e:
        tree = None
        syntax_ok = False
        syntax_msg = f"syntax error: line {e.lineno}: {e.msg}"
    add("python-syntax", syntax_ok, syntax_msg)

    if tree is None:
        add("import-langchain-agent", False, "cannot parse use_agent.py")
        add("constructs-research-profile", False, "cannot parse use_agent.py")
        return

    imported = _find_import(tree)
    add(
        "import-langchain-agent",
        imported,
        "found from genai_tk.agents.langchain import LangchainAgent"
        if imported
        else "missing import of LangchainAgent from genai_tk.agents.langchain",
    )

    calls = list(_langchain_calls(tree))
    if not calls:
        add(
            "constructs-research-profile",
            False,
            "no LangchainAgent(...) call found in use_agent.py",
        )
        return

    profile_hit = next(
        (c for c in calls if _call_profile(c) == "Research"), None
    )
    if profile_hit is None:
        used = [_call_profile(c) for c in calls]
        add(
            "constructs-research-profile",
            False,
            f"LangchainAgent called, but no call targets the Research profile (got: {used})",
        )
        return

    overrides = _call_overrides(profile_hit)
    if overrides:
        add(
            "constructs-research-profile",
            True,
            f"LangchainAgent('Research', ...) with override(s): {', '.join(overrides)}",
        )
    else:
        add(
            "constructs-research-profile",
            False,
            "LangchainAgent('Research') found, but no llm=/sandbox= override",
        )


def _check_yaml(yaml_path, checks):
    def add(name, passed, msg):
        checks.append({"name": name, "passed": bool(passed), "message": msg})

    missing_names = (
        "yaml-parses", "langchain-agents-key", "profile-research", "type-deep",
        "llm-format", "enable-planning", "enable-file-system", "web-search-tool",
        "mcp-servers-list", "skills-directories", "checkpointer-memory",
    )
    if not os.path.isfile(yaml_path):
        for name in missing_names:
            add(name, False, "config/agents/langchain.yaml not found")
        return

    try:
        with open(yaml_path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        for name in missing_names:
            add(name, False, f"cannot read {yaml_path}: {e}")
        return

    parsed = parse_yaml(text)
    add(
        "yaml-parses",
        parsed is not None,
        "parsed as YAML (dict)" if parsed is not None else "unparseable YAML",
    )

    if parsed is None:
        for name in missing_names[1:]:
            add(name, False, "YAML did not parse")
        return

    has_key = "langchain_agents" in parsed
    add(
        "langchain-agents-key",
        has_key,
        "top-level langchain_agents key present"
        if has_key
        else "missing top-level langchain_agents key",
    )

    if not has_key:
        for name in missing_names[2:]:
            add(name, False, "no langchain_agents key")
        return

    agents = parsed["langchain_agents"]
    profile = _find_profile(agents, "Research") if isinstance(agents, dict) else None
    add(
        "profile-research",
        profile is not None,
        "profiles list contains a profile named 'Research'"
        if profile is not None
        else "no profile named 'Research' in langchain_agents.profiles",
    )

    if profile is None:
        for name in missing_names[3:]:
            add(name, False, "no Research profile")
        return

    type_ok = profile.get("type") == "deep"
    add(
        "type-deep",
        type_ok,
        "type: deep" if type_ok else f"type is {profile.get('type')!r}, expected 'deep'",
    )

    llm = profile.get("llm")
    llm_ok = isinstance(llm, str) and "@" in llm and llm.strip() != ""
    add(
        "llm-format",
        llm_ok,
        f"llm '{llm}' uses model_id@provider format" if llm_ok else f"llm {llm!r} is not in model_id@provider format",
    )

    plan_ok = profile.get("enable_planning") is True
    add(
        "enable-planning",
        plan_ok,
        "enable_planning: true" if plan_ok else "enable_planning is not true",
    )

    fs_ok = profile.get("enable_file_system") is True
    add(
        "enable-file-system",
        fs_ok,
        "enable_file_system: true" if fs_ok else "enable_file_system is not true",
    )

    tools = _tools_list(profile)
    web_ok = any(
        isinstance(t, dict) and t.get("spec") == "web_search" for t in tools
    )
    add(
        "web-search-tool",
        web_ok,
        "tools include spec: web_search" if web_ok else "no web_search tool spec found in tools",
    )

    mcp = profile.get("mcp_servers")
    mcp_ok = isinstance(mcp, list) and len(mcp) > 0
    add(
        "mcp-servers-list",
        mcp_ok,
        f"mcp_servers: {mcp}" if mcp_ok else "mcp_servers missing or empty",
    )

    skills = profile.get("skills")
    dirs = skills.get("directories") if isinstance(skills, dict) else None
    dirs_ok = isinstance(dirs, list) and len(dirs) > 0 and any(
        isinstance(d, str) and "${paths.project}" in d and "skills" in d for d in dirs
    )
    add(
        "skills-directories",
        dirs_ok,
        f"skills.directories: {dirs}" if dirs_ok else "skills.directories missing or not pointing at ${paths.project}/skills",
    )

    cp = profile.get("checkpointer")
    cp_ok = isinstance(cp, dict) and cp.get("type") == "memory"
    add(
        "checkpointer-memory",
        cp_ok,
        "checkpointer: {type: memory}" if cp_ok else "checkpointer is not type memory",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checks = []
    _check_yaml(YAML_PATH, checks)
    _check_python(PY_PATH, checks)

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
