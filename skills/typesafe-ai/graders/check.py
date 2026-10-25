"""Hermetic structural checks for the support-triage evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MESSAGE = "I was charged twice yesterday. Please refund the duplicate charge."


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def grade(root: Path) -> dict[str, Any]:
    """Check the task's explicit profile, not the entire public API schema."""
    try:
        request = json.loads(
            (root / "request.json").read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
        design = (root / "design.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {"score": 0.0, "details": "Missing or malformed request.json/design.md"}

    if not isinstance(request, dict):
        return {"score": 0.0, "details": "request.json must contain an object"}
    questions = request.get("questions")
    questions = questions if isinstance(questions, dict) else {}
    state = request.get("state")
    team = questions.get("team")
    team = team if isinstance(team, dict) else {}
    billing = questions.get("billing")
    billing = billing if isinstance(billing, dict) else {}
    urgency = questions.get("urgency")
    urgency = urgency if isinstance(urgency, dict) else {}
    choices = team.get("criteria")
    levels = urgency.get("criteria")
    noul_criteria = billing.get("criteria")
    checks = [
        {
            "name": "HTTP envelope and design artifact",
            "passed": set(request) == {"model", "state", "questions"}
            and request.get("model") == "jev-1.14.0" and text(design),
        },
        {
            "name": "Synthetic evidence preserved",
            "passed": isinstance(state, dict) and state.get("message") == MESSAGE,
        },
        {
            "name": "Three instructed text primitives",
            "passed": set(questions) == {"team", "billing", "urgency"}
            and all(
                isinstance(q, dict) and q.get("type") == kind
                and text(q.get("instructions"))
                and set(q) <= {"type", "instructions", "criteria"}
                for q, kind in ((team, "choice"), (billing, "noul"), (urgency, "score"))
            ),
        },
        {
            "name": "Choice includes fallback with described labels",
            "passed": isinstance(choices, dict)
            and set(choices) == {"billing", "technical", "other"}
            and all(text(v) for v in choices.values()),
        },
        {
            "name": "Portable Score and Noul criteria",
            "passed": isinstance(levels, list) and 2 <= len(levels) <= 10
            and all(text(v) for v in levels) and len(set(levels)) == len(levels)
            and (
                "criteria" not in billing
                or isinstance(noul_criteria, dict) and bool(noul_criteria)
                and set(noul_criteria) <= {"true", "false"}
                and all(text(v) for v in noul_criteria.values())
            ),
        },
    ]
    passed = sum(check["passed"] for check in checks)
    # Every group is an essential contract requirement. A malformed primitive
    # must not pass merely because four of five groups meet a 0.8 threshold.
    return {
        "score": 1.0 if passed == len(checks) else 0.0,
        "details": f"{passed}/{len(checks)} required structural groups passed; semantics use the rubric",
        "checks": checks,
    }


if __name__ == "__main__":
    result = grade(Path.cwd())
    print(json.dumps(result))
    raise SystemExit(0 if result["score"] >= 0.8 else 1)
