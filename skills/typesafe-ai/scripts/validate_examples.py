"""Offline checks for this skill's synthetic text-example profile, not the API."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


SKILL_DIR = Path(__file__).resolve().parents[1]
TEXT_TYPES = {"choice", "noul", "score"}


class ExampleError(ValueError):
    """An asset violates this skill's deliberately conservative example profile."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExampleError(message)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ExampleError("Non-finite JSON constant")


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_pairs,
        parse_constant=_invalid_constant,
    )


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _instructions(value: Any) -> bool:
    return _text(value) or isinstance(value, (dict, list)) and bool(value)


def _bounded_number(value: Any, lower: float, upper: float, name: str) -> None:
    require(
        type(value) in (int, float) and lower <= value <= upper,
        f"{name} must be a finite number in [{lower}, {upper}]",
    )


def validate_request(payload: Any, surface: str) -> None:
    """Validate our complete-instruction, non-image HTTP/MCP fixture profile."""
    require(surface in {"http", "mcp"}, "Unknown example surface")
    require(isinstance(payload, dict), "Request must be an object")
    require(
        set(payload) <= {"model", "state", "questions"},
        "Request fields outside the text-example profile",
    )
    if surface == "http" or "model" in payload:
        require(_text(payload.get("model")), "A nonempty model is required here")
    require(
        isinstance(payload.get("state"), (str, dict, list)),
        "State must be a string, object, or array",
    )
    questions = payload.get("questions")
    require(isinstance(questions, dict) and bool(questions), "Questions are required")
    for key, question in questions.items():
        require(_text(key), "Question IDs must be nonempty strings")
        require(isinstance(question, dict), "Question must be an object")
        require(
            set(question) <= {"type", "instructions", "criteria"},
            "Question fields outside the example profile",
        )
        kind = question.get("type")
        require(isinstance(kind, str) and kind in TEXT_TYPES, "Unsupported text primitive")
        require(_instructions(question.get("instructions")), "Complete instructions required")
        criteria = question.get("criteria")
        if kind == "choice":
            require(isinstance(criteria, dict), "Choice needs a criteria map")
            require(1 <= len(criteria) <= 255, "Choice needs 1–255 options")
            require(
                all(_text(k) and (v is None or _text(v)) for k, v in criteria.items()),
                "Example Choice labels/descriptions must be text (or null descriptions)",
            )
        elif kind == "score":
            require(isinstance(criteria, list), "Score needs ordered criteria")
            require(2 <= len(criteria) <= 10, "Portable Score needs 2–10 levels")
            require(all(_text(v) for v in criteria), "Portable Score levels need text")
        elif "criteria" in question:
            require(isinstance(criteria, dict) and bool(criteria), "Noul criteria need a map")
            require(set(criteria) <= {"true", "false"}, "Unknown Noul criterion")
            require(all(_text(v) for v in criteria.values()), "Noul meanings need text")


def _distribution(value: Any, keys: set[str]) -> None:
    require(isinstance(value, dict) and set(value) == keys, "Distribution labels mismatch")
    for probability in value.values():
        _bounded_number(probability, 0, 1, "Probability")
    require(
        math.isclose(math.fsum(value.values()), 1, abs_tol=1e-6),
        "Example distribution must sum to one",
    )


def validate_response(payload: Any, request: Any, surface: str = "http") -> None:
    """Check a canonical HTTP-shaped mock result after any transport unwrapping."""
    validate_request(request, surface)
    require(isinstance(payload, dict), "Response must be a parsed object")
    require(_text(payload.get("model")), "Returned model is required")
    answers = payload.get("answers")
    require(
        isinstance(answers, dict) and set(answers) == set(request["questions"]),
        "Answer IDs mismatch",
    )
    usage = payload.get("usage")
    require(isinstance(usage, dict), "HTTP-shaped usage is required")
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        require(type(value) is int and value >= 0, "HTTP usage must be nonnegative integers")
    for key, question in request["questions"].items():
        answer = answers[key]
        kind = question["type"]
        require(isinstance(answer, dict) and answer.get("type") == kind, "Answer type mismatch")
        if kind == "noul":
            _bounded_number(answer.get("noul"), 0, 1, "Noul")
            continue
        _bounded_number(answer.get("confidence"), 0, 1, "Confidence")
        if kind == "choice":
            criteria = question["criteria"]
            _distribution(answer.get("probabilities"), set(criteria))
            winner = answer.get("choice")
            require(isinstance(winner, str) and winner in criteria, "Unknown Choice winner")
            probabilities = answer["probabilities"]
            require(
                probabilities[winner] >= max(probabilities.values()) - 1e-6,
                "Choice winner is not a maximum-probability option",
            )
        else:
            criteria = question["criteria"]
            legend = {str(index): text for index, text in enumerate(criteria)}
            require(answer.get("legend") == legend, "Score legend mismatch")
            _distribution(answer.get("probabilities"), set(legend))
            _bounded_number(answer.get("score"), 0, len(criteria) - 1, "Score")
            expected = math.fsum(
                int(level) * probability
                for level, probability in answer["probabilities"].items()
            )
            require(
                math.isclose(answer["score"], expected, abs_tol=1e-6),
                "Mock Score must match its weighted distribution",
            )


def validate_entry(text: str) -> None:
    require(text.startswith("---\n"), "Missing opening frontmatter")
    parts = text[4:].split("\n---\n", 1)
    require(len(parts) == 2, "Missing closing frontmatter")
    header = parts[0]
    require(bool(re.search(r"^name: typesafe-ai$", header, re.M)), "Skill name changed")
    require(bool(re.search(r"^license: MIT$", header, re.M)), "MIT metadata missing")
    lines = header.splitlines()
    require("description: >" in lines, "Expected folded description")
    index = lines.index("description: >") + 1
    description: list[str] = []
    for line in lines[index:]:
        if not line.startswith((" ", "\t")):
            break
        description.append(line.strip())
    require(0 < len(" ".join(description)) <= 500, "Description must contain 1–500 characters")
    require(len(text.splitlines()) <= 300, "Keep the entry point at most 300 lines")


def validate_links(root: Path) -> int:
    root = root.resolve()
    count = 0
    for path in root.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text("utf-8")):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            resolved = (path.parent / unquote(parts.path)).resolve()
            require(resolved.is_relative_to(root), "Local link escapes the skill directory")
            require(resolved.exists(), f"Missing local link from {path.name}: {parts.path}")
            count += 1
    return count


def validate_cases(payload: Any) -> int:
    require(isinstance(payload, dict) and _text(payload.get("note")), "Cases need a note")
    cases = payload.get("cases")
    require(isinstance(cases, list) and bool(cases), "Cases must be a nonempty list")
    names: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "Case must be an object")
        for field in ("name", "situation", "expected_action", "reason"):
            require(_text(case.get(field)), "Case fields must contain explanatory text")
        require(case["name"] not in names, "Duplicate case name")
        names.add(case["name"])
    return len(cases)


def check_skill(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validate_entry((root / "SKILL.md").read_text("utf-8"))
    links = validate_links(root)
    examples = root / "examples"
    http = load_json(examples / "request.http.json")
    mcp = load_json(examples / "request.mcp.json")
    response = load_json(examples / "response.mock.json")
    for surface, request in (("http", http), ("mcp", mcp)):
        validate_response(response, request, surface)
    cases = validate_cases(load_json(examples / "cases.json"))
    ledger = (root / "references" / "sources.md").read_text("utf-8")
    require("## Indexed URLs\n" in ledger, "Indexed source ledger missing")
    urls = re.findall(r"^- <(https://[^>]+)>$", ledger.split("## Indexed URLs\n", 1)[1], re.M)
    require(len(urls) == len(set(urls)) == 131, "Expected 131 unique indexed source URLs")
    return {
        "status": "passed",
        "offline": True,
        "markdown_files": len(list(root.rglob("*.md"))),
        "local_links": links,
        "request_examples": 2,
        "manual_cases": cases,
        "indexed_sources": len(urls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=Path, default=SKILL_DIR)
    args = parser.parse_args()
    try:
        result = check_skill(args.skill_dir)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
