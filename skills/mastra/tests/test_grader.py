import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
GRADER = SKILL_ROOT / "graders" / "check.py"
REFERENCE = SKILL_ROOT / "solutions" / "reference-support-agent"
NAMESPACE = runpy.run_path(str(GRADER))
grade = NAMESPACE["grade"]


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_path = self.root / "src" / "mastra" / "index.ts"
        self.code_path.parent.mkdir(parents=True)
        self.reference_code = (REFERENCE / "src" / "mastra" / "index.ts").read_text(encoding="utf-8")
        self.reference_design = (REFERENCE / "design.md").read_text(encoding="utf-8")
        self.write(self.reference_code, self.reference_design)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, code, design):
        self.code_path.write_text(code, encoding="utf-8")
        (self.root / "design.md").write_text(design, encoding="utf-8")

    def assert_group_failed(self, result, group):
        checks = {check["name"]: check["passed"] for check in result["checks"]}
        self.assertIn(group, checks)
        self.assertFalse(checks[group], result)
        self.assertLess(result["score"], 1.0)
        self.assertLessEqual(result["score"], 0.6)

    def test_reference_scores_one(self):
        result = grade(self.root)
        self.assertEqual(1.0, result["score"], result)
        self.assertEqual(7, len(result["checks"]))
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_key_corruptions_fail_their_essential_group(self):
        mutations = (
            (
                "provider-model-format",
                lambda code, design: (code.replace('model: "openai/gpt-6.1-sol"', 'model: "openai"'), design),
                "current-agent-api",
            ),
            (
                "missing-tool-output-schema",
                lambda code, design: (code.replace("    outputSchema: orderSchema,\n", "", 1), design),
                "validated-tools",
            ),
            (
                "approval-disabled",
                lambda code, design: (code.replace("requireApproval: true", "requireApproval: false"), design),
                "refund-approval-authorization-idempotency",
            ),
            (
                "client-supplied-authorization-identity",
                lambda code, design: (
                    code.replace('context.requestContext.get("actorId")', "inputData.actorId")
                        .replace('context.requestContext.get("tenantId")', "inputData.tenantId"),
                    design,
                ),
                "refund-approval-authorization-idempotency",
            ),
            (
                "workflow-not-committed",
                lambda code, design: (code.replace(".commit()", ".commit"), design),
                "typed-workflow-and-status-lifecycle",
            ),
            (
                "unbounded-agent",
                lambda code, design: (code.replace("maxSteps: 4", "maxSteps: 50"), design),
                "bounded-agent-execution",
            ),
            (
                "volatile-storage",
                lambda code, design: (code.replace("file:./support-memory.db", ":memory:"), design),
                "persistent-memory-and-trusted-identity",
            ),
        )
        for name, mutate, expected_group in mutations:
            with self.subTest(name=name):
                code, design = mutate(self.reference_code, self.reference_design)
                self.write(code, design)
                self.assert_group_failed(grade(self.root), expected_group)

    def test_design_requires_explanation_not_keywords(self):
        start = self.reference_design.index("## Approval binding")
        end = self.reference_design.index("## Tenant authorization")
        design = self.reference_design[:start] + self.reference_design[end:]
        self.write(self.reference_code, design)
        self.assert_group_failed(grade(self.root), "substantive-design-reasoning")

    def test_commented_out_reference_receives_no_code_credit(self):
        self.write("/*\n" + self.reference_code + "\n*/", self.reference_design)
        result = grade(self.root)
        self.assertLessEqual(result["score"], 0.2, result)
        self.assertEqual(1, sum(check["passed"] for check in result["checks"]))
        self.assertTrue(result["checks"][-1]["passed"])

    def test_missing_malformed_and_oversized_artifacts_return_zero(self):
        self.code_path.unlink()
        result = grade(self.root)
        self.assertEqual(0.0, result["score"])
        self.assertEqual([], result["checks"])

        self.code_path.write_bytes(b"\xff\xfe\x00")
        self.assertEqual(0.0, grade(self.root)["score"])

        self.code_path.write_text("x" * (NAMESPACE["CODE_LIMIT"] + 1), encoding="utf-8")
        self.assertEqual(0.0, grade(self.root)["score"])

    def test_cli_emits_json_without_traceback_for_missing_files(self):
        self.code_path.unlink()
        process = subprocess.run(
            [sys.executable, str(GRADER)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(1, process.returncode)
        self.assertEqual("", process.stderr)
        self.assertEqual(0.0, json.loads(process.stdout)["score"])


if __name__ == "__main__":
    unittest.main()
