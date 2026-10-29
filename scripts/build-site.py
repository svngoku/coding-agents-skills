#!/usr/bin/env python3
"""Build the data payload behind the skills gallery (site/public/gallery.json).

The gallery UI is a Vite + React + shadcn/ui app in site/. This script is the
content pipeline: it reads skills/ and README.md, renders every Markdown file
to HTML, and writes the JSON the app fetches at runtime. Components never parse
Markdown, so the Python standard library stays the only text-processing tool.

Usage:
    python3 scripts/build-site.py                # writes site/public/gallery.json
    python3 scripts/build-site.py --output dir   # writes dir/gallery.json
    python3 scripts/build-site.py --check        # fail if the payload is stale

Standard library only, on purpose: this repo ships no runtime dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_markdown import render_markdown  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_OUTPUT = REPO_ROOT / "site" / "public"
OUTPUT_NAME = "gallery.json"

# ---------------------------------------------------------------------------
# Skill taxonomy. Kept here (not in the skills) so the gallery can group skills
# without every skill carrying a category field.
# ---------------------------------------------------------------------------
CATEGORY_ORDER = [
    "AI & Agents",
    "Architecture & Design",
    "Data & Storage",
    "Testing & Quality",
    "Security",
    "Performance",
    "DevOps & Cloud",
    "UI & Frontend",
    "Post-Training & ML",
    "Other",
]

SKILL_CATEGORY = {
    "adaption-ai": "AI & Agents",
    "api-design": "Architecture & Design",
    "clairic": "AI & Agents",
    "database-design": "Data & Storage",
    "ddd": "Architecture & Design",
    "genai-tk": "AI & Agents",
    "langchain": "AI & Agents",
    "linkup-extract": "AI & Agents",
    "linkup-fetch": "AI & Agents",
    "linkup-research": "AI & Agents",
    "linkup-search": "AI & Agents",
    "linkup-workflow": "AI & Agents",
    "microservices-patterns": "Architecture & Design",
    "performance-optimization": "Performance",
    "prime-rl": "Post-Training & ML",
    "scalingo": "DevOps & Cloud",
    "security-best-practices": "Security",
    "smolagents": "AI & Agents",
    "testing-patterns": "Testing & Quality",
    "typesafe-ai": "AI & Agents",
    "ui": "UI & Frontend",
    "unsloth-hf-jobs": "Post-Training & ML",
}

ARTIFACT_KINDS = [
    ("references", "Refs"),
    ("scripts", "Scripts"),
    ("instructions", "Tasks"),
    ("graders", "Graders"),
    ("rubrics", "Rubrics"),
    ("solutions", "Solutions"),
    ("fixtures", "Fixtures"),
]


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def parse_frontmatter_block(text: str) -> dict:
    """Frontmatter parser that keeps folded (`>`) blocks intact."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    data: dict[str, str] = {}
    current = None
    folded: list[str] = []
    for raw in block.splitlines():
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", raw)
        if match and not raw.startswith((" ", "\t")):
            if current:
                data[current] = " ".join(x for x in folded if x).strip()
            current = match.group(1)
            value = match.group(2).strip()
            folded = []
            if value in {">", "|", ">-", "|-"}:
                continue
            data[current] = value.strip('"').strip("'")
            current = None
            folded = []
        elif current:
            folded.append(raw.strip())
    if current:
        data[current] = " ".join(x for x in folded if x).strip()
    return data


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def count_lines(path: Path) -> int:
    return len(read_text(path).splitlines())


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def repo_url() -> str:
    origin = git_output("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/]+([^/]+)/([^/\s]+?)(?:\.git)?$", origin)
    if match:
        return f"https://github.com/{match.group(1)}/{match.group(2)}"
    return "https://github.com/svngoku/coding-agents-skills"


def repo_slug() -> str:
    """owner/repo form used in the install commands."""
    return repo_url().removeprefix("https://github.com/").removesuffix(".git")


def repo_branch() -> str:
    """Branch used for the GitHub links (CI checks out a detached HEAD)."""
    ci_ref = os.environ.get("GITHUB_REF_NAME")
    if ci_ref:
        return ci_ref
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch in {"", "HEAD"}:
        return "main"
    return branch


# ---------------------------------------------------------------------------
# README parsing: per-skill title, description, capabilities, use cases
# ---------------------------------------------------------------------------
def readme_sections(readme: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current = None
    buf: list[str] = []
    for line in readme.splitlines():
        match = re.match(r"^###\s+\d+\.\s+(.*)$", line)
        if match:
            if current is not None:
                sections.append((current, "\n".join(buf)))
            current = match.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections.append((current, "\n".join(buf)))
    return sections


def section_fields(body: str) -> dict:
    """Split a README skill section into labelled fields."""
    fields: dict[str, list[str]] = {}
    label = None
    for line in body.splitlines():
        label_match = re.match(r"^\*\*([^*]+)\*\*:?\s*(.*)$", line)
        if label_match:
            label = label_match.group(1).strip().rstrip(":").strip()
            fields[label] = []
            rest = label_match.group(2).strip()
            if rest:
                fields[label].append(rest)
            continue
        if label is None:
            continue
        if line.startswith("---"):
            label = None
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            fields[label].append(stripped[2:].strip())
        elif stripped:
            fields[label].append(stripped)
    return fields


def plain_markdown(text: str) -> str:
    """Strip Markdown syntax so README fields can be shown as short plain labels."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def match_section(title: str, skill_name: str, *, exact: bool = False) -> bool:
    """Match a README section (e.g. "Domain-Driven Design (DDD)") to a skill folder."""
    slug = slugify(title)
    slug_tokens = set(slug.split("-"))
    if re.search(rf"(?:^|-){re.escape(skill_name)}(?:-|$)", slug):
        return True
    first = skill_name.split("-")[0]
    return not exact and first in slug_tokens


# ---------------------------------------------------------------------------
# Skill collection
# ---------------------------------------------------------------------------
def collect_eval(skill_dir: Path) -> dict:
    eval_yaml = skill_dir / "eval.yaml"
    if not eval_yaml.exists():
        return {"present": False, "tasks": [], "threshold": None}
    text = read_text(eval_yaml)
    tasks = re.findall(r"^\s*-\s*name:\s*(\S+)\s*$", text, re.M)
    threshold = re.search(r"^\s*threshold:\s*([0-9.]+)\s*$", text, re.M)
    return {
        "present": True,
        "tasks": tasks,
        "threshold": float(threshold.group(1)) if threshold else None,
    }


def rewrite_links(document_html: str, doc_path: str, skill_name: str, base_url: str, branch: str) -> str:
    """Point relative links in a rendered document at the GitHub source."""

    def replace(match: re.Match) -> str:
        target = match.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(doc_path), target))
        url = f"{base_url}/blob/{branch}/skills/{skill_name}/{resolved}"
        return f'href="{url}"'

    return re.sub(r'href="([^"]+)"', replace, document_html)


def collect_skill(skill_dir: Path, readme_fields: dict, base_url: str, branch: str) -> dict:
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    text = read_text(skill_md)
    front = parse_frontmatter_block(text)
    fields = readme_fields.get(name, {})

    references = []
    for ref in sorted((skill_dir / "references").rglob("*.md")) if (skill_dir / "references").is_dir() else []:
        if "__pycache__" in ref.relative_to(skill_dir).parts:
            continue
        ref_path = ref.relative_to(skill_dir / "references").as_posix()
        references.append(
            {
                "file": ref_path,
                "title": first_heading(read_text(ref)) or slugify(ref.stem).replace("-", " ").title(),
                "lines": count_lines(ref),
                "url": f"{base_url}/blob/{branch}/skills/{name}/references/{ref_path}",
            }
        )

    scripts = []
    if (skill_dir / "scripts").is_dir():
        for script in sorted((skill_dir / "scripts").rglob("*.py")):
            scripts.append(
                {
                    "file": str(script.relative_to(skill_dir)),
                    "lines": count_lines(script),
                    "url": f"{base_url}/blob/{branch}/skills/{name}/{script.relative_to(skill_dir)}",
                }
            )

    eval_info = collect_eval(skill_dir)

    capabilities = [plain_markdown(item) for item in fields.get("Capabilities", [])]
    capabilities = [item for item in capabilities if item]
    use_cases = [plain_markdown(item) for item in fields.get("Use Cases", [])]
    use_cases = [item for item in use_cases if item]
    description = plain_markdown(" ".join(fields.get("Description", [])))
    if not description:
        description = front.get("description", "")
    status = " ".join(fields.get("Status", [])).strip().lstrip("✅").strip() or "Complete"

    title = " ".join(fields.get("__title__", [])).strip() or first_heading(text) or name

    artifacts = {}
    for kind, _ in ARTIFACT_KINDS:
        if kind == "references":
            artifacts[kind] = len(references)
            continue
        path = skill_dir / kind
        artifacts[kind] = sum(
            1 for item in path.iterdir()
            if item.name != "__pycache__" and item.suffix not in {".pyc", ".pyo"}
        ) if path.is_dir() else 0

    documents = [
        {
            "path": "SKILL.md",
            "label": "SKILL.md",
            "kind": "skill",
            "html": rewrite_links(render_markdown(text), "SKILL.md", name, base_url, branch),
            "lines": count_lines(skill_md),
            "url": f"{base_url}/blob/{branch}/skills/{name}/SKILL.md",
        }
    ]
    for ref in references:
        documents.append(
            {
                "path": f"references/{ref['file']}",
                "label": ref["title"],
                "kind": "guide",
                "html": rewrite_links(
                    render_markdown(read_text(skill_dir / "references" / ref["file"])),
                    f"references/{ref['file']}",
                    name,
                    base_url,
                    branch,
                ),
                "lines": ref["lines"],
                "url": ref["url"],
            }
        )
    for script in scripts:
        source = read_text(skill_dir / script["file"])
        escaped = html.escape(source, quote=False)
        documents.append(
            {
                "path": script["file"],
                "label": script["file"],
                "kind": "script",
                "html": f'<pre tabindex="0"><code class="language-python">{escaped}</code></pre>',
                "lines": script["lines"],
                "url": script["url"],
            }
        )

    docs_lines = count_lines(skill_md) + sum(r["lines"] for r in references)
    total_lines = docs_lines + sum(s["lines"] for s in scripts)

    haystack = " ".join(
        [
            name,
            title,
            description,
            " ".join(capabilities),
            " ".join(use_cases),
            " ".join(r["title"] + " " + r["file"] for r in references),
        ]
    ).lower()

    return {
        "name": name,
        "title": title,
        "category": SKILL_CATEGORY.get(name, "Other"),
        "description": description,
        "trigger": front.get("description", ""),
        "capabilities": capabilities,
        "useCases": use_cases,
        "status": status,
        "references": references,
        "scripts": scripts,
        "documents": documents,
        "eval": eval_info,
        "artifacts": artifacts,
        "docsLines": docs_lines,
        "totalLines": total_lines,
        "haystack": haystack,
        "skillUrl": f"{base_url}/blob/{branch}/skills/{name}/SKILL.md",
        "dirUrl": f"{base_url}/tree/{branch}/skills/{name}",
    }


def collect_skills() -> list[dict]:
    readme = read_text(REPO_ROOT / "README.md")
    sections = readme_sections(readme)
    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())

    readme_fields: dict[str, dict] = {}
    used: set[str] = set()
    for title, body in sections:
        fields = section_fields(body)
        fields["__title__"] = [title]
        matches = [path.name for path in skill_dirs if match_section(title, path.name, exact=True)]
        if not matches:
            matches = [path.name for path in skill_dirs if match_section(title, path.name)]
        for name in matches:
            if name in used:
                continue
            readme_fields[name] = fields
            used.add(name)
            break

    base_url = repo_url()
    branch = repo_branch()
    skills = [
        collect_skill(path, readme_fields, base_url, branch)
        for path in skill_dirs
    ]
    return skills


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------
def build_stats(skills: list[dict]) -> dict:
    return {
        "skills": len(skills),
        "references": sum(len(s["references"]) for s in skills),
        "scripts": sum(len(s["scripts"]) for s in skills),
        "evalTasks": sum(len(s["eval"]["tasks"]) for s in skills),
        "docsLines": sum(s["docsLines"] for s in skills),
        "documents": sum(len(s["documents"]) for s in skills),
    }


def build_categories(skills: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    for skill in skills:
        counts[skill["category"]] = counts.get(skill["category"], 0) + 1
    return [
        {"name": category, "count": counts[category]}
        for category in CATEGORY_ORDER
        if counts.get(category)
    ]


def build_payload(skills: list[dict]) -> dict:
    """The single JSON document consumed by the React app."""
    return {
        "schemaVersion": 1,
        "generatedBy": "scripts/build-site.py",
        "repo": {
            "url": repo_url(),
            "slug": repo_slug(),
            "branch": repo_branch(),
        },
        "stats": build_stats(skills),
        "categories": build_categories(skills),
        "skills": skills,
    }


def serialise(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"directory that receives {OUTPUT_NAME} (default: site/public/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=f"exit non-zero if {OUTPUT_NAME} is stale",
    )
    args = parser.parse_args()

    skills = collect_skills()
    if not skills:
        raise SystemExit("no skills found")
    unmapped = [s["name"] for s in skills if s["category"] == "Other"]
    if unmapped:
        print(
            "note: no category mapped for " + ", ".join(unmapped) + " (add them to SKILL_CATEGORY)",
            file=sys.stderr,
        )

    payload = build_payload(skills)
    output = serialise(payload)
    output_dir = Path(args.output).resolve()
    target = output_dir / OUTPUT_NAME

    if args.check:
        current = read_text(target)
        if current != output:
            print(f"stale: {target} does not match the generated payload", file=sys.stderr)
            return 1
        print(f"ok: {target} is up to date")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    stats = payload["stats"]
    print(
        f"wrote {target} "
        f"({stats['skills']} skills, {stats['documents']} documents, "
        f"{len(output) / 1024 / 1024:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
