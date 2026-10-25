"""Regression tests for the hermetic evaluation grader, not model accuracy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("triage_grader", ROOT / "graders/check.py")
assert SPEC is not None and SPEC.loader is not None
GRADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADER)


class GraderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(
            ["/bin/sh", str(ROOT / "solutions/reference-support-triage")],
            cwd=self.root, check=True,
        )
        self.request = json.loads((self.root / "request.json").read_text())

    def score(self) -> float:
        (self.root / "request.json").write_text(json.dumps(self.request), encoding="utf-8")
        return GRADER.grade(self.root)["score"]

    def test_reference_passes(self) -> None:
        self.assertEqual(self.score(), 1.0)

    def test_missing_or_empty_design_fails(self) -> None:
        (self.root / "design.md").unlink()
        self.assertEqual(self.score(), 0.0)
        (self.root / "design.md").write_text(" ", encoding="utf-8")
        self.assertEqual(self.score(), 0.0)

    def test_bad_json_fails(self) -> None:
        for raw in ('{"model":1,"model":2}', '{"state": NaN}', '[]', '{'):
            with self.subTest(raw=raw):
                (self.root / "request.json").write_text(raw, encoding="utf-8")
                self.assertEqual(GRADER.grade(self.root)["score"], 0.0)

    def test_bad_primitives_and_shapes_fail_without_crashing(self) -> None:
        original = json.dumps(self.request)
        for key, value in (
            ("team", {"type": "box", "instructions": "Locate the issue"}),
            ("billing", None),
            ("urgency", {"type": "score", "instructions": "Rate", "criteria": [{}]}),
        ):
            with self.subTest(key=key):
                self.request = json.loads(original)
                self.request["questions"][key] = value
                self.assertEqual(self.score(), 0.0)
        for value in (None, [], "bad"):
            self.request = json.loads(original)
            self.request["questions"] = value
            self.assertEqual(self.score(), 0.0)

    def test_missing_fallback_fails(self) -> None:
        del self.request["questions"]["team"]["criteria"]["other"]
        self.assertEqual(self.score(), 0.0)

    def test_invalid_score_and_noul_criteria_fail(self) -> None:
        original = json.dumps(self.request)
        for levels in ([None, "Severe"], ["Routine"], ["Same", "Same"], ["x"] * 11):
            self.request = json.loads(original)
            self.request["questions"]["urgency"]["criteria"] = levels
            self.assertEqual(self.score(), 0.0)
        self.request = json.loads(original)
        self.request["questions"]["billing"]["criteria"] = {"maybe": "Unclear"}
        self.assertEqual(self.score(), 0.0)

    def test_wrong_model_or_message_fails(self) -> None:
        self.request["model"] = "jev-latest"
        self.assertEqual(self.score(), 0.0)
        self.request["model"] = "jev-1.14.0"
        self.request["state"]["message"] = "Unrelated"
        self.assertEqual(self.score(), 0.0)

    def test_missing_instructions_fails(self) -> None:
        del self.request["questions"]["billing"]["instructions"]
        self.assertEqual(self.score(), 0.0)


if __name__ == "__main__":
    unittest.main()
