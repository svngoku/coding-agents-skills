#!/usr/bin/env python3
"""Build the static skills gallery (site/index.html) from the skills/ directory.

The page is a single self-contained HTML file: Tailwind CSS v4 is loaded from
the CDN at runtime, everything else (markup, data, behaviour) is generated here.

Usage:
    python3 scripts/build-site.py                # writes site/index.html
    python3 scripts/build-site.py --output dir   # writes dir/index.html
    python3 scripts/build-site.py --check        # fail if the page is stale

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
TEMPLATE_PATH = Path(__file__).resolve().parent / "site-template.html"
DEFAULT_OUTPUT = REPO_ROOT / "site"

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
    "database-design": "Data & Storage",
    "ddd": "Architecture & Design",
    "genai-tk": "AI & Agents",
    "langchain": "AI & Agents",
    "microservices-patterns": "Architecture & Design",
    "performance-optimization": "Performance",
    "prime-rl": "Post-Training & ML",
    "scalingo": "DevOps & Cloud",
    "security-best-practices": "Security",
    "smolagents": "AI & Agents",
    "testing-patterns": "Testing & Quality",
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


def match_section(title: str, skill_name: str) -> bool:
    """Match a README section (e.g. "Domain-Driven Design (DDD)") to a skill folder."""
    slug = slugify(title)
    slug_tokens = set(slug.split("-"))
    if re.search(rf"(?<![a-z0-9-]){re.escape(skill_name)}(?![a-z0-9-])", slug):
        return True
    first = skill_name.split("-")[0]
    return first in slug_tokens


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
    for ref in sorted((skill_dir / "references").glob("*.md")) if (skill_dir / "references").is_dir() else []:
        references.append(
            {
                "file": ref.name,
                "title": first_heading(read_text(ref)) or slugify(ref.stem).replace("-", " ").title(),
                "lines": count_lines(ref),
                "url": f"{base_url}/blob/{branch}/skills/{name}/references/{ref.name}",
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
        path = skill_dir / kind
        artifacts[kind] = len(list(path.iterdir())) if path.is_dir() else 0

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
                "html": f'<pre><code class="language-python">{escaped}</code></pre>',
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

    readme_fields: dict[str, dict] = {}
    used: set[str] = set()
    for title, body in sections:
        fields = section_fields(body)
        fields["__title__"] = [title]
        for skill_dir in sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir()):
            name = skill_dir.name
            if name in used:
                continue
            if match_section(title, name):
                readme_fields[name] = fields
                used.add(name)
                break

    base_url = repo_url()
    branch = repo_branch()
    skills = [
        collect_skill(path, readme_fields, base_url, branch)
        for path in sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    ]
    return skills


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def esc(value: str) -> str:
    return html.escape(value, quote=True)


CARD_CLASS = (
    "skill-card group flex flex-col rounded-2xl border border-transparent bg-white p-5 shadow-sm "
    "dark:border-neutral-800 dark:bg-neutral-900/60"
)
TRANSITION = (
    "transition-[color,background-color,border-color,scale] duration-150 ease-out "
    "motion-reduce:transition-none motion-safe:active:scale-[0.96]"
)

PILL_CLASS = (
    "inline-flex items-center rounded-md bg-neutral-100 px-2 py-0.5 text-xs font-medium tabular-nums "
    "text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300"
)
LINK_CLASS = (
    "font-medium text-emerald-700 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 "
    "focus-visible:outline-emerald-500 dark:text-emerald-400"
)
SUBHEAD_CLASS = "font-semibold text-neutral-700 dark:text-neutral-200"
CHEVRON = (
    '<svg class="size-4 group-open:rotate-180" viewBox="0 0 20 20" fill="none" stroke="currentColor" '
    'stroke-width="1.5" aria-hidden="true">'
    '<path d="M5 7.5 10 12.5 15 7.5" stroke-linecap="round" stroke-linejoin="round" /></svg>'
)


def render_cards(skills: list[dict]) -> str:
    out = []
    for skill in skills:
        name = esc(skill["name"])
        capabilities = "".join(
            '<li class="min-w-0 max-w-full break-words rounded-md border border-neutral-200 px-2 py-1 '
            f'text-xs text-neutral-600 dark:border-neutral-800 dark:text-neutral-300">{esc(c)}</li>'
            for c in skill["capabilities"]
        )
        references = "".join(
            '<li class="flex items-baseline gap-2">'
            f'<button type="button" data-open="{name}" data-path="references/{esc(r["file"])}" '
            f'title="{esc(r["title"])}" class="truncate text-left {LINK_CLASS}">{esc(r["title"])}</button>'
            '<span class="shrink-0 text-xs tabular-nums text-neutral-400 dark:text-neutral-500">'
            f'{r["lines"]} lines</span></li>'
            for r in skill["references"]
        )
        use_cases = "".join(f"<li>{esc(u)}</li>" for u in skill["useCases"])
        scripts = "".join(
            f'<li class="min-w-0">'
            f'<a href="{esc(s["url"])}" class="break-words font-mono text-xs {LINK_CLASS}">'
            f'{esc(s["file"])}</a></li>'
            for s in skill["scripts"]
        )
        if skill["eval"]["present"]:
            eval_label = f'eval · {len(skill["eval"]["tasks"])} task(s)'
        else:
            eval_label = "no eval harness"

        details = ""
        if references:
            details += (
                f'<div><h3 class="{SUBHEAD_CLASS}">Reference guides</h3>'
                f'<ul class="mt-1 space-y-1">{references}</ul></div>'
            )
        if scripts:
            details += (
                f'<div><h3 class="{SUBHEAD_CLASS}">Scripts</h3>'
                f'<ul class="mt-1 space-y-1">{scripts}</ul></div>'
            )
        if use_cases:
            details += (
                f'<div><h3 class="{SUBHEAD_CLASS}">Use cases</h3>'
                '<ul class="mt-1 list-disc space-y-1 pl-4 text-neutral-600 dark:text-neutral-300">'
                f"{use_cases}</ul></div>"
            )
        details += (
            '<div class="flex flex-wrap gap-3 pt-1">'
            f'<button type="button" data-open="{name}" class="{LINK_CLASS}">Read the skill</button>'
            f'<a href="{esc(skill["skillUrl"])}" class="{LINK_CLASS}">SKILL.md on GitHub</a>'
            f'<a href="{esc(skill["dirUrl"])}" class="{LINK_CLASS}">Skill directory</a>'
            "</div>"
        )

        out.append(
            f"""
      <article
        class="{CARD_CLASS}"
        data-name="{name}"
        data-title="{esc(skill["title"])}"
        data-category="{esc(skill["category"])}"
        data-refs="{len(skill["references"])}"
        data-lines="{skill["docsLines"]}"
        data-search="{esc(skill["haystack"])}"
        id="skill-{name}"
      >
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <h2 class="text-lg font-semibold text-balance text-neutral-900 dark:text-neutral-100">
              <button
                type="button"
                data-open="{name}"
                class="text-left transition-colors hover:text-emerald-700 focus-visible:outline-2
                       focus-visible:outline-offset-2 focus-visible:outline-emerald-500
                       dark:hover:text-emerald-400"
              >
                {esc(skill["title"])}
              </button>
            </h2>
            <p class="mt-0.5 break-words font-mono text-xs text-neutral-500 dark:text-neutral-400">
              skills/{name}/SKILL.md
            </p>
          </div>
          <span class="shrink-0 rounded-full bg-neutral-100 px-2.5 py-0.5 text-xs font-medium
                       text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300">
            {esc(skill["category"])}
          </span>
        </div>

        <p class="mt-3 line-clamp-4 text-pretty text-sm leading-6 text-neutral-600 dark:text-neutral-300">
          {esc(skill["description"])}
        </p>

        {f'<ul class="mt-4 flex flex-wrap gap-1.5">{capabilities}</ul>' if capabilities else ""}

        <div class="mt-4 flex flex-wrap items-center gap-2">
          <span class="{PILL_CLASS}">{len(skill["references"])} guides</span>
          <span class="{PILL_CLASS}">{skill["docsLines"]} lines</span>
          <span class="{PILL_CLASS}">{esc(skill["status"])}</span>
          <span class="{PILL_CLASS}">{esc(eval_label)}</span>
        </div>

        <details class="group mt-4 border-t border-neutral-200 pt-3 dark:border-neutral-800">
          <summary class="flex cursor-pointer list-none items-center gap-1.5 text-sm font-medium
                          text-emerald-700 hover:underline focus-visible:outline-2
                          focus-visible:outline-offset-2 focus-visible:outline-emerald-500
                          dark:text-emerald-400">
            {CHEVRON}
            Details &amp; references
          </summary>
          <div class="mt-3 space-y-3 text-sm">{details}</div>
        </details>
      </article>"""
        )
    return "\n".join(out)


def render_matrix(skills: list[dict]) -> str:
    head = "".join(
        '<th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase '
        f'text-neutral-500 dark:text-neutral-400">{esc(label)}</th>'
        for _, label in ARTIFACT_KINDS
    )
    rows = []
    for skill in skills:
        cells = []
        for kind, _ in ARTIFACT_KINDS:
            count = skill["artifacts"][kind]
            if count:
                cells.append(
                    '<td class="px-3 py-2 text-center text-xs font-medium tabular-nums text-emerald-700 '
                    f'dark:text-emerald-400">{count}</td>'
                )
            else:
                cells.append('<td class="px-3 py-2 text-center text-xs text-neutral-300 dark:text-neutral-600">·</td>')
        link = (
            f'<a href="#skill-{esc(skill["name"])}" class="{LINK_CLASS}">{esc(skill["title"])}</a>'
        )
        rows.append(
            '<tr class="border-t border-neutral-200 dark:border-neutral-800" '
            f'data-category="{esc(skill["category"])}" data-search="{esc(skill["haystack"])}">'
            '<th scope="row" class="px-3 py-2 text-left font-medium">'
            f"{link}"
            f'<span class="ml-2 font-mono text-xs text-neutral-400">{esc(skill["name"])}</span></th>'
            + "".join(cells)
            + '<td class="px-3 py-2 text-right text-xs tabular-nums text-neutral-500 dark:text-neutral-400">'
            f'{skill["docsLines"]}</td></tr>'
        )
    return f"""
      <table class="w-full border-collapse text-sm">
        <thead>
          <tr>
            <th scope="col" class="px-3 py-2 text-left text-xs font-semibold uppercase text-neutral-500
                       dark:text-neutral-400">Skill</th>
            {head}
            <th scope="col" class="px-3 py-2 text-right text-xs font-semibold uppercase text-neutral-500
                       dark:text-neutral-400">Doc lines</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>"""


def render_category_chips(skills: list[dict]) -> str:
    counts: dict[str, int] = {}
    for skill in skills:
        counts[skill["category"]] = counts.get(skill["category"], 0) + 1
    base = (
        'category-chip inline-flex min-h-11 items-center gap-1.5 rounded-full '
        "border border-neutral-300 bg-white px-3 py-1.5 text-sm font-medium text-neutral-700 "
        + TRANSITION + " hover:border-emerald-600 hover:text-emerald-700 "
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-500 "
        "dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200 dark:hover:border-emerald-600"
    )
    chips = [
        f'<button type="button" data-category="all" aria-pressed="true" class="{base}">'
        f'All <span class="text-xs tabular-nums text-neutral-400">{len(skills)}</span></button>'
    ]
    for category in CATEGORY_ORDER:
        if category not in counts:
            continue
        chips.append(
            f'<button type="button" data-category="{esc(category)}" aria-pressed="false" class="{base}">'
            f'{esc(category)} <span class="text-xs tabular-nums text-neutral-400">{counts[category]}</span></button>'
        )
    return "\n        ".join(chips)


def render_stats(skills: list[dict]) -> str:
    refs = sum(len(s["references"]) for s in skills)
    scripts = sum(len(s["scripts"]) for s in skills)
    tasks = sum(len(s["eval"]["tasks"]) for s in skills)
    lines = sum(s["docsLines"] for s in skills)
    stats = [
        ("Skills", len(skills), "loadable agent skills"),
        ("Reference guides", refs, "deep-dive documents"),
        ("Eval tasks", tasks, "skillgrade harnesses"),
        ("Example scripts", scripts, "runnable helpers"),
        ("Documentation", f"{lines:,}", "markdown lines"),
    ]
    cells = []
    for label, value, hint in stats:
        cells.append(
            """
        <div class="rounded-2xl border border-neutral-200 bg-white px-4 py-3 dark:border-neutral-800
                    dark:bg-neutral-900/60">
          <div class="text-2xl font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
            {value}
          </div>
          <div class="text-sm font-medium text-neutral-600 dark:text-neutral-300">{label}</div>
          <div class="text-xs text-neutral-400 dark:text-neutral-500">{hint}</div>
        </div>""".format(
                value=esc(str(value)), label=esc(label), hint=esc(hint)
            )
        )
    return "\n".join(cells)


def build_page(skills: list[dict], font_url: str, font_css: str) -> str:
    template = read_text(TEMPLATE_PATH)
    if not template:
        raise SystemExit(f"template not found: {TEMPLATE_PATH}")

    replacements = {
        "{{SKILL_COUNT}}": str(len(skills)),
        "{{CATEGORY_CHIPS}}": render_category_chips(skills),
        "{{STATS}}": render_stats(skills),
        "{{CARDS}}": render_cards(skills),
        "{{MATRIX}}": render_matrix(skills),
        "{{REPO_URL}}": repo_url(),
        "{{BRANCH}}": repo_branch(),
        "{{FONT_URL}}": font_url,
        "{{FONT_CSS}}": (
            f'<link rel="stylesheet" href="{esc(font_css)}" />' if font_css else ""
        ),
        "{{DOCS_JSON}}": json.dumps(
            {skill["name"]: skill["documents"] for skill in skills}, ensure_ascii=False
        ).replace("</", "<\\/"),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output directory (default: site/)")
    parser.add_argument(
        "--font-url",
        default=os.environ.get("NB_FONT_URL", "fonts"),
        help=(
            "directory that serves the licensed NB International Pro woff2 files "
            "(default: fonts, or the NB_FONT_URL environment variable)"
        ),
    )
    parser.add_argument(
        "--font-css",
        default=os.environ.get("NB_FONT_CSS", ""),
        help="URL of a stylesheet that declares the NB International Pro @font-face rules",
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero if the generated page is stale")
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
    page = build_page(skills, args.font_url.rstrip("/"), args.font_css.strip())

    output_dir = Path(args.output).resolve()
    target = output_dir / "index.html"

    if args.check:
        current = read_text(target)
        if current != page:
            print(f"stale: {target} does not match the generated output", file=sys.stderr)
            return 1
        print(f"ok: {target} is up to date")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    print(f"wrote {target} ({len(skills)} skills, {len(page.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
