#!/usr/bin/env python3
"""Golden-principles linter. Standard library only. Exit 1 on any violation.

GP-1 AGENTS.md is a map (<150 lines)       GP-2 Required docs exist
GP-3 Diagrams are Mermaid only             GP-4 No secrets committed
GP-5 Active plans have frontmatter         GP-6 Relative doc links resolve
GP-7 CI is hardened (permissions, pinned refs)

Behaviourally identical to golden_principles.js; run it from the repository root.
"""

import os
import re
import sys
from pathlib import Path

SKIP = {"node_modules", ".git", "dist", "build", ".venv", "vendor", ".next", "coverage", "__pycache__"}
BOX = re.compile(r"[┌┐└┘├┤┬┴┼│─╭╮╯╰]|\+-{3,}\+")
BINARY = re.compile(r"\.(png|jpe?g|gif|webp|ico|pdf|zip|gz|woff2?|ttf|lock)$", re.I)
SECRETS = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"), "private key block"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"sk-[A-Za-z0-9_-]{32,}"), "API secret key (sk-...)"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "Google API key"),
)
REQUIRED_DOCS = ("ARCHITECTURE", "DESIGN", "PLANS", "QUALITY_SCORE", "RELIABILITY", "SECURITY", "PRODUCT_SENSE")

errors = []


def fail(gp, file, message):
    errors.append(f"{gp} {file}: {message}")


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def walk(directory):
    found = []
    if not os.path.exists(directory):
        return found
    for entry in sorted(os.scandir(directory), key=lambda item: item.name):
        if entry.name in SKIP:
            continue
        path = os.path.normpath(os.path.join(directory, entry.name))
        if entry.is_dir(follow_symlinks=False):
            found.extend(walk(path))
        else:
            found.append(path)
    return found


def posix(path):
    return path.replace(os.sep, "/")


files = [posix(item) for item in walk(".")]
markdown = [item for item in files if item.endswith(".md")]

# GP-1
if not os.path.exists("AGENTS.md"):
    fail("GP-1", "AGENTS.md", "missing")
elif len(read("AGENTS.md").split("\n")) >= 150:
    fail("GP-1", "AGENTS.md", "must stay under 150 lines (map, not manual)")

# GP-2
for name in REQUIRED_DOCS:
    if not os.path.exists(f"docs/{name}.md"):
        fail("GP-2", f"docs/{name}.md", "missing")

# GP-3: no ASCII-art diagrams outside mermaid fences; no empty mermaid; no other diagram formats
for file in markdown:
    fence = None
    source = read(file)
    for number, line in enumerate(source.split("\n"), 1):
        marker = re.match(r"^\s*(```+|~~~+)\s*(\w*)", line)
        if marker:
            fence = None if fence else (marker.group(2) or "plain")
            continue
        if fence in (None, "plain", "text") and BOX.search(line):
            fail("GP-3", f"{file}:{number}", "ASCII-art diagram, use a Mermaid block instead")
    for block in re.finditer(r"^```mermaid\n([\s\S]*?)^```", source, re.M):
        if not block.group(1).strip():
            fail("GP-3", file, "empty mermaid block")
for file in files:
    if re.search(r"\.(puml|plantuml|dot|gv)$", file):
        fail("GP-3", file, "non-Mermaid diagram source, convert to .mmd")

# GP-4: .env ignored + secret-shaped strings
if os.path.exists(".env"):
    gitignore = read(".gitignore") if os.path.exists(".gitignore") else ""
    if not re.search(r"^\.env$", gitignore, re.M):
        fail("GP-4", ".env", "present but not in .gitignore")
for file in files:
    if BINARY.search(file) or file.endswith("golden_principles.py") or file.endswith("golden_principles.js"):
        continue
    try:
        text = read(file)
    except OSError:
        continue
    for pattern, label in SECRETS:
        if pattern.search(text):
            fail("GP-4", file, f"looks like a committed secret ({label})")

# GP-5
for file in walk("docs/exec-plans/active"):
    if file.endswith(".md") and not re.match(r"---\n[\s\S]*?\bstatus:", read(file)):
        fail("GP-5", posix(file), "missing frontmatter with status")

# GP-6
for file in markdown:
    fence = False
    for line in read(file).split("\n"):
        if re.match(r"^\s*(```|~~~)", line):
            fence = not fence
            continue
        if fence:
            continue
        for link in re.finditer(r"\]\((?!https?:|mailto:|#)([^)#\s]+)", line):
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(file), link.group(1)))):
                fail("GP-6", file, f"broken link -> {link.group(1)}")

# GP-7: workflows declare permissions and never use floating branch refs
for file in walk(".github/workflows"):
    if not re.search(r"\.ya?ml$", file):
        continue
    workflow = read(file)
    if not re.search(r"^permissions:", workflow, re.M):
        fail("GP-7", posix(file), "missing top-level `permissions:` (least privilege)")
    for use in re.finditer(r"uses:\s*([^\s#]+)@(main|master|latest)\b", workflow):
        fail("GP-7", posix(file), f"action {use.group(1)} pinned to floating ref @{use.group(2)}")

if errors:
    print("\n".join(errors) + f"\n\n{len(errors)} golden-principle violation(s)", file=sys.stderr)
    sys.exit(1)
print("golden principles: all clear")
