"""A small Markdown renderer for the static skill gallery.

Standard library only. It covers the constructs the skills actually use:
headings, paragraphs, fenced and indented code, tables, ordered/unordered
lists (nested), task lists, blockquotes, horizontal rules, inline code,
emphasis, strong, strikethrough, links, images and raw HTML blocks.

Anything it does not understand is escaped and shown as text, never dropped.
"""

from __future__ import annotations

import html
import re

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HR_RE = re.compile(r"^\s*(?:-[ \t]*){3,}$|^\s*(?:\*[ \t]*){3,}$|^\s*(?:_[ \t]*){3,}$")
TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
HTML_BLOCK_RE = re.compile(r"^\s{0,3}</?[a-zA-Z][^>]*>")
AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
PLACEHOLDER = "\x00code{}\x00"


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "section"


# ---------------------------------------------------------------------------
# Inline rendering
# ---------------------------------------------------------------------------
def render_inline(text: str) -> str:
    codes: list[str] = []

    def stash_code(match: re.Match) -> str:
        codes.append(f"<code>{match.group(1)}</code>")
        return PLACEHOLDER.format(len(codes) - 1)

    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", stash_code, escaped)

    escaped = re.sub(
        r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}" loading="lazy" />',
        escaped,
    )
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escaped,
    )
    escaped = AUTOLINK_RE.sub(lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>', escaped)
    escaped = re.sub(r"~~([^~]+)~~", r"<s>\1</s>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"(?<![\w_])_([^_\n]+)_(?!\w)", r"<em>\1</em>", escaped)

    for index, code in enumerate(codes):
        escaped = escaped.replace(PLACEHOLDER.format(index), code)
    return escaped


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------
def render_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    html_out, _ = render_blocks(lines)
    return "\n".join(html_out)


def render_blocks(lines: list[str]) -> tuple[list[str], int]:
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(2)
            language = fence.group(3)
            body: list[str] = []
            index += 1
            while index < len(lines):
                closing = FENCE_RE.match(lines[index])
                if closing and closing.group(2)[0] == marker[0] and len(closing.group(2)) >= len(marker):
                    index += 1
                    break
                body.append(lines[index])
                index += 1
            klass = f' class="language-{html.escape(language)}"' if language else ""
            code = html.escape("\n".join(body), quote=False)
            # tabindex makes horizontally scrollable blocks reachable by keyboard.
            out.append(f'<pre tabindex="0"><code{klass}>{code}</code></pre>')
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            content = render_inline(heading.group(2))
            ident = slugify(re.sub(r"[^\w\s-]", "", heading.group(2)))
            out.append(f'<h{level} id="{ident}">{content}</h{level}>')
            index += 1
            continue

        if HR_RE.match(line):
            out.append("<hr />")
            index += 1
            continue

        if "|" in line and index + 1 < len(lines) and TABLE_DELIM_RE.match(lines[index + 1]):
            table, index = render_table(lines, index)
            out.append(table)
            continue

        if LIST_RE.match(line):
            block, index = render_list(lines, index)
            out.append(block)
            continue

        if BLOCKQUOTE_RE.match(line):
            block, index = render_blockquote(lines, index)
            out.append(block)
            continue

        if HTML_BLOCK_RE.match(line):
            block, index = render_html_block(lines, index)
            out.append(block)
            continue

        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip():
            candidate = lines[index]
            if (
                FENCE_RE.match(candidate)
                or HEADING_RE.match(candidate)
                or HR_RE.match(candidate)
                or LIST_RE.match(candidate)
                or BLOCKQUOTE_RE.match(candidate)
                or HTML_BLOCK_RE.match(candidate)
            ):
                break
            paragraph.append(candidate.strip())
            index += 1
        if paragraph:
            out.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
    return out, index


def render_table(lines: list[str], index: int) -> tuple[str, int]:
    def split_row(row: str) -> list[str]:
        cells = row.strip().strip("|").split("|")
        return [cell.strip() for cell in cells]

    header = split_row(lines[index])
    alignments: list[str] = []
    for spec in split_row(lines[index + 1]):
        left = spec.startswith(":")
        right = spec.endswith(":")
        if left and right:
            alignments.append("center")
        elif right:
            alignments.append("right")
        else:
            alignments.append("left")
    index += 2
    body: list[list[str]] = []
    while index < len(lines) and lines[index].strip() and "|" in lines[index]:
        body.append(split_row(lines[index]))
        index += 1

    def cell(tag: str, value: str, position: int) -> str:
        align = alignments[position] if position < len(alignments) else "left"
        return f'<{tag} class="text-{align}">{render_inline(value)}</{tag}>'

    parts = ["<table>", "<thead>", "<tr>"]
    for position, value in enumerate(header):
        parts.append(cell("th", value, position))
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row in body:
        parts.append("<tr>")
        for position, value in enumerate(row):
            parts.append(cell("td", value, position))
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return '<div class="table-scroll">' + "".join(parts) + "</div>", index


def render_list(lines: list[str], index: int) -> tuple[str, int]:
    first = LIST_RE.match(lines[index])
    assert first is not None
    base_indent = len(first.group(1))
    ordered = bool(re.match(r"\d", first.group(2)))
    tag = "ol" if ordered else "ul"
    parts = [f"<{tag}>"]
    while index < len(lines):
        match = LIST_RE.match(lines[index])
        if not match:
            break
        indent = len(match.group(1))
        if indent < base_indent:
            break
        item_lines = [match.group(3)]
        index += 1
        while index < len(lines):
            nxt = lines[index]
            if not nxt.strip():
                lookahead = index + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead < len(lines) and LIST_RE.match(lines[lookahead]) and len(
                    LIST_RE.match(lines[lookahead]).group(1)
                ) >= base_indent:
                    index = lookahead
                    continue
                break
            if LIST_RE.match(nxt) and len(LIST_RE.match(nxt).group(1)) <= indent:
                break
            item_lines.append(nxt.strip())
            index += 1
        parts.append(render_list_item(item_lines))
    parts.append(f"</{tag}>")
    return "".join(parts), index


def render_list_item(item_lines: list[str]) -> str:
    task = re.match(r"^\[([ xX])\]\s+(.*)$", item_lines[0]) if item_lines else None
    classes = ""
    if task:
        checked = "checked" if task.group(1).lower() == "x" else "unchecked"
        classes = f' class="task-{checked}"'
    blocks, _ = render_blocks(item_lines)
    if not blocks:
        return "<li></li>"
    if task:
        blocks[0] = f'<p class="task-line"><span class="task-box">{"☑" if classes.endswith("checked") else "☐"}</span> {blocks[0][3:] if blocks[0].startswith("<p>") else blocks[0]}</p>'
    if len(blocks) == 1 and blocks[0].startswith("<p>"):
        inner = blocks[0][3:-4]
    else:
        inner = "".join(blocks)
    return f"<li{classes}>{inner}</li>"


def render_blockquote(lines: list[str], index: int) -> tuple[str, int]:
    quoted: list[str] = []
    while index < len(lines):
        match = BLOCKQUOTE_RE.match(lines[index])
        if not match:
            break
        quoted.append(match.group(1))
        index += 1
    inner, _ = render_blocks(quoted)
    return f"<blockquote>{''.join(inner)}</blockquote>", index


def render_html_block(lines: list[str], index: int) -> tuple[str, int]:
    block: list[str] = []
    while index < len(lines) and lines[index].strip():
        block.append(lines[index])
        index += 1
    return "\n".join(block), index
