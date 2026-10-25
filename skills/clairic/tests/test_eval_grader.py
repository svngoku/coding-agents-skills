"""Regression tests at the grader's public artifact/JSON boundary; no live SDK calls."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("clairic_grader", ROOT / "graders/check.py")
assert SPEC is not None and SPEC.loader is not None
GRADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADER)


class GraderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(
            ["/bin/sh", str(ROOT / "solutions/reference-invoice")],
            cwd=self.root, check=True, timeout=10,
        )
        self.source = (self.root / "invoice.py").read_text(encoding="utf-8")

    def test_reference_passes(self) -> None:
        result = GRADER.grade(self.root)
        self.assertEqual(result["score"], 1.0, result)
        self.assertEqual(len(result["checks"]), 4)

    def test_missing_artifact_fails(self) -> None:
        for filename in ("invoice.py", "requirements.txt", "design.md"):
            with self.subTest(filename=filename):
                path = self.root / filename
                contents = path.read_bytes()
                path.unlink()
                self.assertEqual(GRADER.grade(self.root)["score"], 0.0)
                path.write_bytes(contents)

    def test_live_context_cannot_shadow_text_or_model(self) -> None:
        for target in ("text", "InvoiceExtraction"):
            with self.subTest(target=target):
                candidate = self.source.replace("as client:", f"as {target}:")
                candidate = candidate.replace("return client.extract(", f"return {target}.extract(")
                (self.root / "invoice.py").write_text(candidate, encoding="utf-8")
                self.assertEqual(GRADER.grade(self.root)["score"], 0.0)

    def test_misplaced_future_import_is_not_valid_python(self) -> None:
        candidate = self.source.replace("from __future__ import annotations\n", "")
        candidate += "\nfrom __future__ import annotations\n"
        (self.root / "invoice.py").write_text(candidate, encoding="utf-8")
        self.assertEqual(GRADER.grade(self.root)["score"], 0.0)

    def test_parenthesized_field_does_not_define_pydantic_annotation(self) -> None:
        candidate = self.source.replace("invoice_number: TextAnswer", "(invoice_number): TextAnswer")
        (self.root / "invoice.py").write_text(candidate, encoding="utf-8")
        self.assertEqual(GRADER.grade(self.root)["score"], 0.0)

    def grade_source(self, candidate: str) -> dict:
        (self.root / "invoice.py").write_text(candidate, encoding="utf-8")
        return GRADER.grade(self.root)

    def assert_rejected_replacement(self, old: str, new: str) -> None:
        self.assertIn(old, self.source)
        result = self.grade_source(self.source.replace(old, new))
        self.assertEqual(result["score"], 0.0, result)

    def with_helper(self, body: str) -> str:
        prefix = self.source.split("\ndef accept_total(", 1)[0]
        return (
            prefix + "\ndef accept_total(result: Result[InvoiceExtraction]) -> float | None:\n"
            + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n"
        )

    def test_reference_is_executable_and_uses_no_live_calls_when_generated(self) -> None:
        self.assertTrue((ROOT / "solutions/reference-invoice").stat().st_mode & 0o111)
        self.assertIn("320 synthetic", GRADER.grade(self.root)["checks"][-1]["details"])

    def test_cli_emits_json_and_failure_exit_code_without_sdk_or_credentials(self) -> None:
        for expected_code, expected_score in ((0, 1.0), (1, 0.0)):
            if expected_code:
                (self.root / "invoice.py").write_text("not valid Python!", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(ROOT / "graders/check.py")],
                cwd=self.root, capture_output=True, text=True,
                env={"PYTHONDONTWRITEBYTECODE": "1"}, timeout=10,
            )
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertEqual(json.loads(result.stdout)["score"], expected_score)
            self.assertEqual(result.stderr, "")

    def test_empty_design_fails_but_semantics_are_not_keyword_scored(self) -> None:
        path = self.root / "design.md"
        path.write_text(" \n", encoding="utf-8")
        self.assertEqual(GRADER.grade(self.root)["score"], 0.0)
        path.write_text("Semantics are deliberately reserved for the LLM rubric.", encoding="utf-8")
        self.assertEqual(GRADER.grade(self.root)["score"], 1.0)

    def test_version_pin_and_extra_requirement_rejected(self) -> None:
        for content in ("clairic", "clairic>=0.1.0", "clairic==9.0", "clairic==0.1.0\nother==1"):
            with self.subTest(content=content):
                (self.root / "requirements.txt").write_text(content, encoding="utf-8")
                self.assertEqual(GRADER.grade(self.root)["score"], 0.0)
        (self.root / "requirements.txt").write_text(
            "# task dependency\nclairic==0.1.0 # pinned\n", encoding="utf-8"
        )
        self.assertEqual(GRADER.grade(self.root)["score"], 1.0)

    def test_malformed_oversized_or_symlink_artifacts_fail(self) -> None:
        for data in (b"\xff", b"\x00", b"x" * 65_537, b"def broken(:"):
            with self.subTest(data_length=len(data)):
                (self.root / "invoice.py").write_bytes(data)
                self.assertEqual(GRADER.grade(self.root)["score"], 0.0)
        (self.root / "invoice.py").unlink()
        (self.root / "invoice.py").symlink_to(ROOT / "graders/check.py")
        self.assertEqual(GRADER.grade(self.root)["score"], 0.0)

    def test_schema_must_use_exact_supported_answer_types(self) -> None:
        for old, new in (
            ("invoice_number: TextAnswer", "invoice_number: str"),
            ("NumberAnswer[float]", "NumberAnswer"),
            ("NumberAnswer[float]", "NumberAnswer[int]"),
            ("due_date: DateAnswer", "due_date: str"),
            ("class InvoiceExtraction(BaseModel):", "class InvoiceExtraction:"),
            ("class InvoiceExtraction(BaseModel):", "@decorator\nclass InvoiceExtraction(BaseModel):"),
            ("    invoice_number: TextAnswer", "    invoice_number: TextAnswer\n    extra: TextAnswer"),
            ("    invoice_number: TextAnswer", "    invoice_number: TextAnswer\n    invoice_number: TextAnswer"),
            ("    invoice_number: TextAnswer", "    invoice_number: TextAnswer = None"),
        ):
            with self.subTest(old=old, new=new):
                self.assert_rejected_replacement(old, new)

    def test_number_spec_requires_keyword_unit_and_decimals(self) -> None:
        original = 'Number(unit="USD", decimals=2)'
        for replacement in (
            'Number("USD", 2)', 'Number(unit="EUR", decimals=2)',
            'Number(unit="USD", decimals=True)', 'Number(unit="USD", decimals=7)',
            'Number(unit="USD")', 'Number(unit="USD", decimals=2, options={})',
        ):
            with self.subTest(replacement=replacement):
                self.assert_rejected_replacement(original, replacement)

    def test_supported_formatting_order_and_context_variable_variations_pass(self) -> None:
        candidate = self.source.replace(
            'Number(unit="USD", decimals=2)', "Number( decimals = 2, unit = 'USD' )"
        )
        candidate = candidate.replace("as client:", "as session:")
        candidate = candidate.replace("client.extract(", "session.extract(")
        candidate = candidate.replace(
            "    invoice_number: TextAnswer", '    """Invoice fields."""\n    invoice_number: TextAnswer'
        )
        self.assertEqual(self.grade_source(candidate)["score"], 1.0)
        candidate = self.source.replace("from __future__ import annotations\n", "")
        self.assertEqual(self.grade_source(candidate)["score"], 1.0)

    def test_imports_are_public_canonical_and_bound(self) -> None:
        for old, new in (
            ("from clairic import", "from clairic._client import"),
            ("from clairic import", "from other import"),
            ("from math import isfinite", ""),
            ("from math import isfinite", "from math import isfinite as predicate"),
            ("from datetime import date", "from datetime import *"),
            ("from pydantic import BaseModel, Field", "from pydantic import BaseModel"),
        ):
            with self.subTest(new=new):
                self.assert_rejected_replacement(old, new)

    def test_constructor_credentials_endpoints_and_unsupported_kwargs_fail(self) -> None:
        for replacement in (
            "Clairic()", "Clairic(timeout=1.0)", "Clairic(30.0)",
            'Clairic(timeout=30.0, account_id="hardcoded")',
            'Clairic(timeout=30.0, api_token="hardcoded")',
            'Clairic(timeout=30.0, base_url="https://example.invalid")',
            "Clairic(timeout=30.0, http_client=transport)",
        ):
            with self.subTest(replacement=replacement):
                self.assert_rejected_replacement("Clairic(timeout=30.0)", replacement)

    def test_extract_model_date_and_only_supported_parameters(self) -> None:
        original = "client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 26))"
        for replacement in (
            'client.extract(text, "clairic", reference_date=date(2026, 10, 26))',
            "client.extract(text, InvoiceExtraction)",
            "client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 25))",
            'client.extract(text, InvoiceExtraction, reference_date="2026-10-26")',
            "client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 26), images=[])",
            "client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 26), options={})",
            "client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 26), stream=True)",
            "client.run(text, InvoiceExtraction, reference_date=date(2026, 10, 26))",
            "None",
        ):
            with self.subTest(replacement=replacement):
                self.assert_rejected_replacement(original, replacement)

    def test_import_time_work_and_extra_functions_fail_without_execution(self) -> None:
        marker = self.root / "should-not-exist"
        suffixes = (
            "\nClairic(timeout=30.0)\n",
            "\nextract_invoice('live')\n",
            "\nif __name__ == '__main__':\n    extract_invoice('live')\n",
            "\ndef hidden():\n    return Clairic()\n",
            f"\nimport pathlib\npathlib.Path({str(marker)!r}).touch()\n",
            "\nraise RuntimeError('candidate module was imported')\n",
        )
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                self.assertEqual(self.grade_source(self.source + suffix)["score"], 0.0)
                self.assertFalse(marker.exists())

    def test_malformed_essential_group_cannot_pass_with_partial_credit(self) -> None:
        result = self.grade_source(self.source.replace("decimals=2", "decimals=3"))
        self.assertEqual(sum(c["passed"] for c in result["checks"]), 3)
        self.assertEqual(result["score"], 0.0)
        self.assertLess(0.7 * result["score"] + 0.3, 0.8)

    def test_each_acceptance_guard_and_boundary_is_behaviorally_tested(self) -> None:
        for old, new in (
            ("if result.truncated:", "if False:"),
            ("if value is None:", "if False:"),
            ("total.p_null >= 0.5", "total.p_null > 0.5"),
            ("total.p_null >= 0.5", "False"),
            ("total.confidence < 0.9", "total.confidence <= 0.9"),
            ("total.confidence < 0.9", "False"),
            ("not isfinite(value)", "False"),
            ("value < 0", "False"),
            ("value < 0", "value <= 0"),
            ("return value", "return 0.0"),
        ):
            with self.subTest(old=old, new=new):
                result = self.grade_source(self.source.replace(old, new))
                self.assertEqual(result["score"], 0.0, result)
                self.assertFalse(result["checks"][-1]["passed"])

    def test_constant_null_and_guess_fallback_cannot_pass(self) -> None:
        for body in (
            "return None",
            "return 999.0",
            "return result.data.total.guess",
            "if result.data.total.value is None:\n    return 0.0\nreturn result.data.total.value",
        ):
            with self.subTest(body=body):
                result = self.grade_source(self.with_helper(body))
                self.assertEqual(result["score"], 0.0)
                self.assertFalse(result["checks"][-1]["passed"])

    def test_alternate_branching_and_math_isfinite_pass(self) -> None:
        body = """
            amount = result.data.total
            if not result.truncated:
                if amount.value is not None:
                    if amount.p_null < 0.5 and amount.confidence >= 0.9:
                        if math.isfinite(amount.value) and amount.value >= 0:
                            return amount.value
            return None
        """
        candidate = self.with_helper(body).replace("from math import isfinite", "import math")
        self.assertEqual(self.grade_source(candidate)["score"], 1.0)

    def test_no_general_sandbox_escape_syntax_is_executed(self) -> None:
        marker = self.root / "escape-marker"
        attacks = (
            f"return __import__('pathlib').Path({str(marker)!r}).touch()",
            "return result.__class__",
            "return result.data.total.__dict__",
            "return isfinite.__globals__",
            "return math.isfinite.__self__",
            "return ().__class__.__base__.__subclasses__()",
            "return __builtins__",
            "predicate = isfinite\nreturn predicate(result.data.total.value)",
            "isfinite = result.data.total.value\nreturn isfinite",
            "math = result\nreturn math",
            "result = result.data\nreturn None",
            "result.data.total.value = 0.0\nreturn None",
            "import os\nreturn os.system('false')",
            "while True:\n    pass",
            "for item in result:\n    pass",
            "return [item for item in result]",
            "return (lambda: None)()",
            "def nested():\n    return None\nreturn nested()",
            "return result['data']",
            "return 2 ** 10000000",
            "return (value := 1)",
            "return f'{result}'",
            "try:\n    return None\nexcept:\n    return None",
            "global isfinite\nreturn None",
            "return isfinite(*result)",
            "return isfinite(value=result.data.total.value)",
        )
        for attack in attacks:
            with self.subTest(attack=attack):
                result = self.grade_source(self.with_helper(attack))
                self.assertEqual(result["score"], 0.0, result)
                if "checks" in result:
                    self.assertFalse(result["checks"][-1]["passed"])
                self.assertFalse(marker.exists())

    def test_annotations_defaults_and_decorators_are_never_executed(self) -> None:
        marker = self.root / "signature-marker"
        payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"
        for candidate in (
            self.source.replace(
                "def accept_total(result: Result[InvoiceExtraction])",
                f"def accept_total(result: {payload})",
            ),
            self.source.replace(
                "def accept_total(result: Result[InvoiceExtraction])",
                f"def accept_total(result: Result[InvoiceExtraction] = {payload})",
            ),
            self.source.replace("def accept_total(", f"@{payload}\ndef accept_total("),
        ):
            with self.subTest(candidate_tail=candidate[candidate.index("def extract_invoice"):]):
                self.assertEqual(self.grade_source(candidate)["score"], 0.0)
                self.assertFalse(marker.exists())
                # The isolated helper still passes: these expressions are stripped, not evaluated.
                outcome, details = GRADER.helper_outcomes(ast.parse(candidate))
                self.assertTrue(outcome, details)
                self.assertFalse(marker.exists())

    def test_duplicate_functions_and_oversized_helper_fail(self) -> None:
        extra = "\ndef accept_total(result):\n    return None\n"
        self.assertEqual(self.grade_source(self.source + extra)["score"], 0.0)
        oversized = "value = result.data.total.value\n" * 40 + "return value"
        result = self.grade_source(self.with_helper(oversized))
        self.assertEqual(result["score"], 0.0)
        self.assertIn("bounded AST", result["checks"][-1]["details"])

    def test_imports_must_bind_names_before_definitions(self) -> None:
        lines = self.source.splitlines(keepends=True)
        import_line = "from pydantic import BaseModel, Field\n"
        self.assertIn(import_line, lines)
        candidate = self.source.replace(import_line, "") + "\n" + import_line
        self.assertEqual(self.grade_source(candidate)["score"], 0.0)

    def test_runtime_invalid_generic_annotation_fails(self) -> None:
        candidate = self.source.replace("from __future__ import annotations\n", "")
        candidate = candidate.replace(
            "text: str", "text: float[InvoiceExtraction]"
        )
        self.assertEqual(self.grade_source(candidate)["score"], 0.0)

if __name__ == "__main__":
    unittest.main()
