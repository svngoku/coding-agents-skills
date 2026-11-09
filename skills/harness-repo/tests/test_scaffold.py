import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = SKILL_ROOT / "scripts" / "scaffold.sh"
JS_LINTER = SKILL_ROOT / "assets" / "repo" / ".opencode" / "tools" / "golden_principles.js"
PY_LINTER = SKILL_ROOT / "assets" / "tools" / "golden_principles.py"
HAS_NODE = shutil.which("node") is not None


def scaffold(parent, stack, *extra):
    return subprocess.run(
        ["bash", str(SCAFFOLD), "--name", "demo", "--stack", stack, "--desc", "Demo project.",
         "--path", str(parent), *extra],
        check=True, capture_output=True, text=True,
    ).stdout


def lint(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    lines = sorted(line for line in result.stderr.splitlines() if line and not line.endswith("violation(s)"))
    return result.returncode, lines


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)

    def test_python_stack_uses_the_python_linter_everywhere(self):
        scaffold(self.parent, "python")
        project = self.parent / "demo"
        self.assertTrue((project / ".opencode/tools/golden_principles.py").is_file())
        self.assertFalse((project / ".opencode/tools/golden_principles.js").exists())
        self.assertTrue(os.access(project / ".opencode/tools/golden_principles.py", os.X_OK))
        for relative in ("AGENTS.md", "README.md", ".github/workflows/ci.yml", ".pre-commit-config.yaml",
                         ".github/pull_request_template.md", "docs/ARCHITECTURE.md"):
            text = (project / relative).read_text(encoding="utf-8")
            self.assertIn("golden_principles.py", text, relative)
            self.assertNotIn("golden_principles.js", text, relative)
            self.assertNotIn("{{", text.replace("${{", ""), relative)
        ci = (project / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("setup-node", ci)
        self.assertEqual(0, lint(["python3", ".opencode/tools/golden_principles.py"], project)[0])

    def test_other_stacks_keep_the_node_linter(self):
        for stack in ("node", "go", "ruby", "generic"):
            with self.subTest(stack=stack):
                scaffold(self.parent / stack, stack)
                project = self.parent / stack / "demo"
                self.assertTrue((project / ".opencode/tools/golden_principles.js").is_file())
                self.assertFalse((project / ".opencode/tools/golden_principles.py").exists())
                self.assertIn("node .opencode/tools/golden_principles.js", (project / "AGENTS.md").read_text())

    def test_rerun_is_idempotent_and_plan_folders_survive_git(self):
        scaffold(self.parent, "python")
        self.assertIn("created=0", scaffold(self.parent, "python").splitlines()[-1])
        project = self.parent / "demo"
        for folder in ("active", "completed"):
            self.assertTrue((project / "docs/exec-plans" / folder / ".gitkeep").is_file(), folder)

    def test_existing_initial_setup_plan_is_not_duplicated(self):
        completed = self.parent / "demo" / "docs" / "exec-plans" / "completed"
        completed.mkdir(parents=True)
        (completed / "2026-05-30-initial-setup.md").write_text("# Done\n", encoding="utf-8")
        scaffold(self.parent, "python")
        active = self.parent / "demo" / "docs" / "exec-plans" / "active"
        self.assertEqual([], sorted(path.name for path in active.glob("*initial-setup*.md")))

    def test_plans_index_has_numbered_active_and_completed_tables(self):
        scaffold(self.parent, "python")
        plans = (self.parent / "demo/docs/PLANS.md").read_text(encoding="utf-8")
        self.assertIn("## Active Plans", plans)
        self.assertIn("## Completed Plans", plans)
        self.assertIn("Depends on", plans)
        self.assertIn("[001: Initial setup]", plans)
        template = (self.parent / "demo/docs/exec-plans/_template.md").read_text(encoding="utf-8")
        for heading in ("Executor instructions", "Drift check", "## Commands You Will Need",
                        "## Scope", "## STOP Conditions"):
            self.assertIn(heading, template)


@unittest.skipUnless(HAS_NODE, "node is required to compare the two linters")
class LinterParityTests(unittest.TestCase):
    """The Python linter must agree with the JavaScript one on every golden principle."""

    @classmethod
    def setUpClass(cls):
        cls.pristine = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.pristine.cleanup)
        scaffold(Path(cls.pristine.name), "python")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "demo"
        shutil.copytree(Path(self.pristine.name) / "demo", self.project)

    def both(self):
        js = lint(["node", str(JS_LINTER)], self.project)
        py = lint(["python3", str(PY_LINTER)], self.project)
        return js, py

    def write(self, relative, text, mode="w"):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(text)

    def assert_same(self, expected_prefix=None):
        js, py = self.both()
        self.assertEqual(js, py)
        if expected_prefix is None:
            self.assertEqual(0, js[0])
        else:
            self.assertEqual(1, js[0])
            self.assertTrue(any(line.startswith(expected_prefix) for line in js[1]), js)

    def test_clean_tree_agrees(self):
        self.assert_same()

    def test_each_principle_agrees_when_violated(self):
        cases = (
            ("GP-1", lambda: self.write("AGENTS.md", "filler\n" * 160, "a")),
            ("GP-2", lambda: (self.project / "docs/RELIABILITY.md").unlink()),
            ("GP-3", lambda: self.write("docs/DESIGN.md", "\n+----+\n| A  |\n+----+\n", "a")),
            ("GP-3 ", lambda: self.write("docs/flow.puml", "@startuml\n@enduml\n")),
            ("GP-4", lambda: self.write(".env.example", "TOKEN=AKIAABCDEFGHIJKLMNOP\n", "a")),
            ("GP-5", lambda: self.write("docs/exec-plans/active/bad.md", "# no frontmatter\n")),
            ("GP-6", lambda: self.write("docs/DESIGN.md", "\nSee [x](nope.md).\n", "a")),
            ("GP-7", lambda: self.write(
                ".github/workflows/x.yml", "name: x\njobs:\n  a:\n    steps:\n      - uses: actions/checkout@main\n")),
        )
        for prefix, mutate in cases:
            with self.subTest(principle=prefix):
                shutil.rmtree(self.project)
                shutil.copytree(Path(self.pristine.name) / "demo", self.project)
                mutate()
                self.assert_same(prefix)

    def test_env_file_must_be_ignored(self):
        self.write(".env", "APP_ENV=development\n")
        self.assert_same()
        (self.project / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
        self.assert_same("GP-4")


if __name__ == "__main__":
    unittest.main()
