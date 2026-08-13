#!/usr/bin/env python3
"""Deterministic grader for the ui "checkout-card" eval task.

Runs from the agent's workspace (cwd) and checks the single required artifact:

  CheckoutCard.tsx  a React + Tailwind order-summary component that follows
                    the ui skill's rules: full-height layout with h-dvh
                    (never h-screen), aria-label on icon-only buttons, a
                    confirmation dialog (AlertDialog / confirm) for the
                    destructive action, a cn() utility (clsx + tailwind-merge)
                    for class logic, text-balance heading, tabular-nums on
                    monetary values, no gradients, no purple accents, no
                    tracking-* letter-spacing overrides, and state-driven
                    interactions.

The TSX file is inspected as text (Python stdlib only, no network, no TS
parser). Checks are lenient about quoting/formatting/naming variations but
strict about the skill's core API surface and anti-patterns.

Prints JSON to stdout:
  {"score": 0.91, "details": "10/11 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import json
import os
import re
import sys

TSX_PATH = "CheckoutCard.tsx"
PASS_BAR = 0.8

FULL_HEIGHT_TOKENS = (
    "h-screen", "h-dvh", "h-svh",
    "min-h-screen", "min-h-dvh", "min-h-svh",
    "100vh", "100dvh",
)
GRADIENT_TOKENS = (
    "bg-gradient-to-", "bg-linear-to-", "bg-radial", "bg-conic", "linear-gradient(",
)
PURPLE_TOKENS = ("purple-", "violet-", "fuchsia-")

BUTTON_RE = re.compile(r"<(?:button|Button)\b([^>]*)>", re.I)
ICON_TAG_RE = re.compile(r"<[A-Z][A-Za-z0-9]*\b")
WORD_RE = re.compile(r"[A-Za-z0-9]")


def _opening_tag_end(text, start):
    """Index just past the closing '>' of the tag at <start>.

    Handles '>' inside quoted attributes and inside {...} JSX expressions
    (e.g. onClick={() => removeItem(id)}), so arrow functions and template
    literals in attributes do not truncate the opening tag.
    """
    i = start
    depth = 0  # inside {...}
    quote = None
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == ">" and depth == 0:
            return i + 1
        i += 1
    return -1


def _buttons_with_inner(text):
    """Yield (opening_tag, inner_html) for each button element found."""
    out = []
    for m in BUTTON_RE.finditer(text):
        tag_end = _opening_tag_end(text, m.start())
        if tag_end == -1:
            continue
        open_tag = text[m.start():tag_end]
        if open_tag.rstrip().endswith("/>"):
            out.append((open_tag, ""))  # self-closing
            continue
        rest = text[tag_end:]
        close_lower = rest.find("</button>")
        close_upper = rest.find("</Button>")
        close = -1
        for cand in (close_lower, close_upper):
            if cand != -1 and (close == -1 or cand < close):
                close = cand
        if close == -1 or close > 4000:
            out.append((open_tag, ""))
        else:
            out.append((open_tag, rest[:close]))
    return out


def _visible_text(inner):
    """Text inside a button after removing SVG blocks and tags."""
    inner = re.sub(r"<svg\b.*?</svg>", "", inner, flags=re.S | re.I)
    inner = re.sub(r"<[^>]+>", "", inner)
    for entity, ch in (("&nbsp;", " "), ("&#215;", "x"), ("&times;", "x")):
        inner = inner.replace(entity, ch)
    return inner.strip()


def _is_icon_only(inner):
    """True when a button contains an icon element and no visible word text."""
    lower = inner.lower()
    has_icon = (
        "<svg" in lower
        or bool(ICON_TAG_RE.search(inner))
        or "aria-hidden" in lower
    )
    has_word = bool(WORD_RE.search(_visible_text(inner)))
    return has_icon and not has_word


def _check_tsx(checks):
    def add(name, passed, msg):
        checks.append({"name": name, "passed": bool(passed), "message": msg})

    check_names = (
        "component-defined", "full-height-dvh", "icon-only-button-aria-label",
        "destructive-confirmation", "cn-utility", "no-gradients",
        "text-balance", "tabular-nums", "no-tracking", "no-purple",
        "state-driven",
    )

    if not os.path.isfile(TSX_PATH):
        for name in check_names:
            add(name, False, "CheckoutCard.tsx not found in the workspace")
        return

    try:
        with open(TSX_PATH, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        for name in check_names:
            add(name, False, f"cannot read CheckoutCard.tsx: {e}")
        return

    # 1. A CheckoutCard component is actually defined and returns JSX.
    defined = "CheckoutCard" in text and bool(re.search(r"\breturn\s*\(", text))
    add("component-defined", defined,
        "CheckoutCard component defined and returns JSX" if defined
        else "no CheckoutCard component with a JSX return found")

    # 2. Full-height layout: h-dvh, never h-screen (or no full-height at all).
    has_screen = "h-screen" in text
    has_dvh = bool(re.search(r"\bh-dvh\b|\bmin-h-dvh\b|100dvh", text))
    has_any_full = any(tok in text for tok in FULL_HEIGHT_TOKENS)
    if has_screen:
        tok = "min-h-screen" if "min-h-screen" in text else "h-screen"
        add("full-height-dvh", False, f"{tok} used — use h-dvh instead")
    elif has_dvh:
        add("full-height-dvh", True, "h-dvh used for the full-height layout")
    elif not has_any_full:
        add("full-height-dvh", True,
            "no full-height utility used (h-dvh is preferred)")
    else:
        add("full-height-dvh", False,
            "full-height utility used but not h-dvh")

    # 3. Every icon-only button carries an aria-label.
    buttons = _buttons_with_inner(text)
    icon_buttons = [b for b in buttons if _is_icon_only(b[1])]
    labeled = [b for b in icon_buttons if re.search(r"aria-label\s*=", b[0])]
    if not icon_buttons:
        add("icon-only-button-aria-label", False,
            "no icon-only button found (remove buttons must be icon-only with aria-label)")
    elif len(labeled) == len(icon_buttons):
        add("icon-only-button-aria-label", True,
            f"all {len(icon_buttons)} icon-only button(s) have aria-label")
    else:
        add("icon-only-button-aria-label", False,
            f"{len(icon_buttons) - len(labeled)}/{len(icon_buttons)} icon-only button(s) missing aria-label")

    # 4. Destructive action guarded by a confirmation dialog.
    has_dialog = "AlertDialog" in text
    has_confirm = bool(re.search(r"\b(?:window\.)?confirm\s*\(", text))
    if has_dialog or has_confirm:
        add("destructive-confirmation", True,
            "AlertDialog or confirm() present"
            + (" (AlertDialog)" if has_dialog else " (confirm())"))
    else:
        add("destructive-confirmation", False,
            "no AlertDialog / confirm() found for the destructive action")

    # 5. cn() utility — or clsx + tailwind-merge.
    has_cn = bool(re.search(r"\bcn\s*\(", text))
    has_clsx = bool(re.search(r"\bclsx\b", text))
    has_tw = "tailwind-merge" in text or bool(re.search(r"\btwMerge\s*\(", text))
    if has_cn or (has_clsx and has_tw):
        add("cn-utility", True, "cn() (clsx + tailwind-merge) used for class logic")
    else:
        add("cn-utility", False,
            "no cn() (clsx + tailwind-merge) for conditional classes")

    # 6. No gradients.
    hits = [t for t in GRADIENT_TOKENS if t in text.lower()]
    add("no-gradients", not hits,
        "no gradient utilities" if not hits else f"gradient found: {hits[0]}")

    # 7. text-balance on the heading.
    add("text-balance", "text-balance" in text,
        "text-balance present" if "text-balance" in text
        else "missing text-balance on the heading")

    # 8. tabular-nums on monetary values.
    add("tabular-nums", "tabular-nums" in text,
        "tabular-nums present on prices" if "tabular-nums" in text
        else "missing tabular-nums on monetary values")

    # 9. No letter-spacing overrides.
    tracking = re.search(r"tracking-[A-Za-z0-9_]", text)
    add("no-tracking", tracking is None,
        "no tracking-* overrides" if tracking is None
        else f"letter-spacing override: {tracking.group(0)}")

    # 10. No purple / multicolor accents.
    purple = [t for t in PURPLE_TOKENS if t in text.lower()]
    add("no-purple", not purple,
        "no purple-family colors" if not purple else f"purple-family color: {purple[0]}")

    # 11. Interactions are state-driven (useState / useReducer).
    stateful = bool(re.search(r"\buse(?:State|Reducer)\s*(?:<[^>]*>)?\s*\(", text))
    add("state-driven", stateful,
        "state hook used for interactions" if stateful
        else "no useState/useReducer — interactions are not state-driven")


def main():
    checks = []
    _check_tsx(checks)

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
