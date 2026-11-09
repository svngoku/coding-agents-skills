import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
GRADER = SKILL_ROOT / "graders" / "check.py"
SOLUTION = SKILL_ROOT / "solutions" / "reference-ledger-api"
NAMESPACE = runpy.run_path(str(GRADER))
grade = NAMESPACE["grade"]


class GraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.reference_dir.cleanup)
        subprocess.run(
            ["bash", str(SOLUTION)],
            cwd=cls.reference_dir.name,
            env={**os.environ, "HARNESS_REPO_SKILL_DIR": str(SKILL_ROOT)},
            check=True,
            capture_output=True,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        shutil.copytree(Path(self.reference_dir.name) / "ledger-api", self.workspace / "ledger-api")
        self.project = self.workspace / "ledger-api"

    def edit(self, relative, transform):
        path = self.project / relative
        path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")

    def assert_group_failed(self, group):
        result = grade(self.workspace)
        checks = {check["name"]: check["passed"] for check in result["checks"]}
        self.assertIn(group, checks)
        self.assertFalse(checks[group], result)
        self.assertLessEqual(result["score"], 0.6)

    def test_reference_scores_one(self):
        result = grade(self.workspace)
        self.assertEqual(1.0, result["score"], result)
        self.assertEqual(7, len(result["checks"]))
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_oversized_agents_map_fails(self):
        self.edit("AGENTS.md", lambda text: text + "\nfiller\n" * 120)
        self.assert_group_failed("tailored-agents-map")

    def test_claude_must_stay_a_pointer(self):
        self.edit("CLAUDE.md", lambda text: text + "Extra duplicated rules.\n")
        self.assert_group_failed("tailored-agents-map")

    def test_money_boundary_is_required(self):
        self.edit(
            "AGENTS.md",
            lambda text: text.replace("minor units", "units").replace("float", "number"),
        )
        self.assert_group_failed("tailored-agents-map")

    def test_unrendered_placeholder_fails(self):
        self.edit("AGENTS.md", lambda text: text + "\nName: {{PROJECT_NAME}}\n")
        self.assert_group_failed("tailored-agents-map")

    def test_missing_active_plan_link_fails(self):
        self.edit("docs/PLANS.md", lambda text: text.replace("exec-plans/active/", "exec-plans/old/"))
        self.assert_group_failed("knowledge-base-and-plan")

    def test_broken_relative_link_fails(self):
        self.edit("docs/DESIGN.md", lambda text: text + "\nSee [gone](missing-file.md).\n")
        self.assert_group_failed("knowledge-base-and-plan")

    def test_committed_env_and_leftovers_fail(self):
        (self.project / ".env").write_text("APP_ENV=development\n", encoding="utf-8")
        self.assert_group_failed("security-baseline-intact")
        (self.project / ".env").unlink()
        (self.project / "AGENTS.md.harness-new").write_text("x\n", encoding="utf-8")
        self.assert_group_failed("security-baseline-intact")

    def test_secret_shaped_string_fails(self):
        self.edit(".env.example", lambda text: text + "TOKEN=AKIAABCDEFGHIJKLMNOP\n")
        self.assert_group_failed("security-baseline-intact")

    def test_ci_without_permissions_or_with_floating_ref_fails(self):
        path = ".github/workflows/ci.yml"
        original = (self.project / path).read_text(encoding="utf-8")
        self.edit(path, lambda text: text.replace("permissions:\n  contents: read\n", ""))
        self.assert_group_failed("hardened-ci")
        (self.project / path).write_text(original, encoding="utf-8")
        self.edit(path, lambda text: text.replace("actions/checkout@v7", "actions/checkout@main", 1))
        self.assert_group_failed("hardened-ci")

    def test_ascii_diagram_fails(self):
        self.edit("docs/ARCHITECTURE.md", lambda text: text + "\n+----+\n| A  |\n+----+\n")
        self.assert_group_failed("mermaid-only-diagrams")

    def test_non_mermaid_source_fails(self):
        (self.project / "docs" / "flow.puml").write_text("@startuml\n@enduml\n", encoding="utf-8")
        self.assert_group_failed("mermaid-only-diagrams")

    def test_package_name_mismatch_fails(self):
        self.edit("pyproject.toml", lambda text: text.replace('name = "ledger-api"', 'name = "other"'))
        self.assert_group_failed("python-tooling-consistent")

    def test_missing_smoke_test_fails(self):
        (self.project / "tests" / "test_smoke.py").unlink()
        self.assert_group_failed("python-tooling-consistent")

    def test_handoff_claiming_admin_settings_are_on_fails(self):
        self.edit(
            "handoff.md",
            lambda text: text + "\nBranch protection is enabled and secret scanning is configured.\n",
        )
        self.assert_group_failed("honest-admin-handoff")

    def test_handoff_missing_an_admin_item_fails(self):
        self.edit("handoff.md", lambda text: text.replace("private vulnerability reporting", "reports"))
        self.assert_group_failed("honest-admin-handoff")

    def test_missing_project_returns_zero(self):
        shutil.rmtree(self.project)
        result = grade(self.workspace)
        self.assertEqual(0.0, result["score"])
        self.assertEqual([], result["checks"])

    def test_symlinked_project_returns_zero(self):
        target = self.workspace / "elsewhere"
        self.project.rename(target)
        os.symlink(target, self.project)
        self.assertEqual(0.0, grade(self.workspace)["score"])

    def test_cli_emits_json_without_traceback_for_missing_project(self):
        shutil.rmtree(self.project)
        process = subprocess.run(
            [sys.executable, str(GRADER)],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(1, process.returncode)
        self.assertEqual("", process.stderr)
        self.assertEqual(0.0, json.loads(process.stdout)["score"])


if __name__ == "__main__":
    unittest.main()
