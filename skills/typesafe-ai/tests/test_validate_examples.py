"""Deterministic regression tests; no networking, SDK, key, or model required."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "skill_examples", ROOT / "scripts" / "validate_examples.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = CHECK.load_json(ROOT / "examples" / "request.http.json")
        self.response = CHECK.load_json(ROOT / "examples" / "response.mock.json")

    def test_complete_skill(self) -> None:
        result = CHECK.check_skill(ROOT)
        self.assertTrue(result["offline"])
        self.assertEqual(result["indexed_sources"], 131)
        self.assertEqual(result["manual_cases"], 12)

    def test_mcp_may_omit_model_but_http_cannot(self) -> None:
        del self.request["model"]
        CHECK.validate_request(self.request, "mcp")
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "http")

    def test_empty_model_rejected(self) -> None:
        self.request["model"] = ""
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "mcp")

    def test_state_types(self) -> None:
        for state in ("text", {}, []):
            self.request["state"] = state
            CHECK.validate_request(self.request, "http")
        for state in (None, False, 12):
            with self.subTest(state=state), self.assertRaises(CHECK.ExampleError):
                self.request["state"] = state
                CHECK.validate_request(self.request, "http")

    def test_empty_questions_rejected(self) -> None:
        self.request["questions"] = {}
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "http")

    def test_structured_instructions(self) -> None:
        for instructions in ({"judgment": "Check billing"}, ["Check billing"]):
            self.request["questions"]["billing"]["instructions"] = instructions
            CHECK.validate_request(self.request, "http")

    def test_incomplete_instructions_rejected_by_profile(self) -> None:
        for instructions in ("", {}, [], None, False):
            with self.subTest(value=instructions), self.assertRaises(CHECK.ExampleError):
                self.request["questions"]["billing"]["instructions"] = instructions
                CHECK.validate_request(self.request, "http")

    def test_experimental_primitive_not_in_text_profile(self) -> None:
        self.request["questions"]["billing"]["type"] = "box"
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "mcp")

    def test_extra_transport_fields_outside_profile(self) -> None:
        self.request["assets"] = {}
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "http")

    def test_choice_nullable_description(self) -> None:
        self.request["questions"]["team"]["criteria"]["other"] = None
        CHECK.validate_request(self.request, "http")

    def test_choice_needs_candidates(self) -> None:
        self.request["questions"]["team"]["criteria"] = {}
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "http")

    def test_portable_score_levels(self) -> None:
        for criteria in (["only one"], ["level"] * 11, ["ok", None], ["ok", ""]):
            with self.subTest(criteria=criteria), self.assertRaises(CHECK.ExampleError):
                self.request["questions"]["urgency"]["criteria"] = criteria
                CHECK.validate_request(self.request, "http")

    def test_unknown_noul_criterion(self) -> None:
        self.request["questions"]["billing"]["criteria"] = {"maybe": "unclear"}
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_request(self.request, "http")

    def test_missing_answer(self) -> None:
        del self.response["answers"]["billing"]
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_unexpected_answer(self) -> None:
        self.response["answers"]["extra"] = {"type": "noul", "noul": 0.9}
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_answer_type_mismatch(self) -> None:
        self.response["answers"]["billing"]["type"] = "choice"
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_unknown_choice_winner(self) -> None:
        self.response["answers"]["team"]["choice"] = "missing"
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_choice_winner_matches_distribution(self) -> None:
        self.response["answers"]["team"]["choice"] = "technical"
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_tied_choice_winners_are_valid(self) -> None:
        answer = self.response["answers"]["team"]
        answer["probabilities"] = {"billing": 0.5, "technical": 0.5, "other": 0}
        for winner in ("billing", "technical"):
            answer["choice"] = winner
            CHECK.validate_response(self.response, self.request)

    def test_probability_range_and_numeric_types(self) -> None:
        for value in (-0.1, 1.1, True, None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(CHECK.ExampleError):
                self.response["answers"]["billing"]["noul"] = value
                CHECK.validate_response(self.response, self.request)

    def test_valid_uncertainty_is_not_a_contract_error(self) -> None:
        self.response["answers"]["billing"]["noul"] = 0.5
        self.response["answers"]["urgency"]["confidence"] = 0.49
        CHECK.validate_response(self.response, self.request)

    def test_distribution_labels_and_sum(self) -> None:
        invalid = (
            {"billing": 1},
            {"billing": 0.5, "technical": 0.1, "other": 0.1},
            {"billing": 1.1, "technical": -0.1, "other": 0},
        )
        for probabilities in invalid:
            with self.subTest(value=probabilities), self.assertRaises(CHECK.ExampleError):
                self.response["answers"]["team"]["probabilities"] = probabilities
                CHECK.validate_response(self.response, self.request)

    def test_score_legend_must_preserve_mapping(self) -> None:
        self.response["answers"]["urgency"]["legend"]["0"] = "Opposite meaning"
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_score_weighted_expectation(self) -> None:
        self.response["answers"]["urgency"]["score"] = 1.9
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response(self.response, self.request)

    def test_http_usage_is_not_sdk_nullable_usage(self) -> None:
        for value in (None, True, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(CHECK.ExampleError):
                self.response["usage"]["input_tokens"] = value
                CHECK.validate_response(self.response, self.request)

    def test_error_text_is_not_a_parsed_result(self) -> None:
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_response("Permission denied", self.request)

    def test_strict_json_rejects_duplicate_keys_and_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for text in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text, encoding="utf-8")
                    CHECK.load_json(path)

    def test_local_link_missing_and_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "skill"
            root.mkdir()
            entry = root / "SKILL.md"
            (root.parent / "outside.md").write_text("Outside the skill", encoding="utf-8")
            for target, error in (
                ("missing.md", "Missing local link"),
                ("../outside.md", "escapes the skill directory"),
            ):
                with self.subTest(target=target), self.assertRaisesRegex(CHECK.ExampleError, error):
                    entry.write_text(f"[target]({target})", encoding="utf-8")
                    CHECK.validate_links(root)

    def test_valid_relative_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SKILL.md").write_text("[reference](ref.md)", encoding="utf-8")
            (root / "ref.md").write_text("# Reference", encoding="utf-8")
            self.assertEqual(CHECK.validate_links(root), 1)

    def test_frontmatter_required_and_bounded(self) -> None:
        entry = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        for invalid in (entry.replace("name: typesafe-ai", "name: other"), entry + "\n" * 301):
            with self.subTest(length=len(invalid)), self.assertRaises(CHECK.ExampleError):
                CHECK.validate_entry(invalid)

    def test_case_completeness_and_duplicate_names(self) -> None:
        cases = CHECK.load_json(ROOT / "examples" / "cases.json")
        duplicate = copy.deepcopy(cases)
        duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_cases(duplicate)
        del cases["cases"][0]["reason"]
        with self.assertRaises(CHECK.ExampleError):
            CHECK.validate_cases(cases)


if __name__ == "__main__":
    unittest.main()
