"""Regression tests for reproducible gallery metadata."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "build_site", Path(__file__).with_name("build-site.py")
)
assert SPEC is not None and SPEC.loader is not None
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)


class ArtifactCountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.skill = Path(self.temp.name) / "example"
        self.skill.mkdir()
        (self.skill / "SKILL.md").write_text(
            "---\nname: example\ndescription: Example\n---\n# Example\n",
            encoding="utf-8",
        )

    def collect(self):
        return SITE.collect_skill(self.skill, {}, "https://example.com/repo", "main")

    def test_absent_artifact_directories_are_zero(self):
        self.assertTrue(all(count == 0 for count in self.collect()["artifacts"].values()))

    def test_python_caches_do_not_change_the_payload(self):
        scripts = self.skill / "scripts"
        scripts.mkdir()
        (scripts / "example.py").write_text("pass\n", encoding="utf-8")
        clean = self.collect()
        cache = scripts / "__pycache__"
        cache.mkdir()
        (cache / "example.cpython-313.pyc").write_bytes(b"cache")
        (scripts / "legacy.pyc").write_bytes(b"cache")
        (scripts / "legacy.pyo").write_bytes(b"cache")
        self.assertEqual(clean, self.collect())

    def test_caches_are_ignored_in_every_artifact_kind(self):
        for kind, _ in SITE.ARTIFACT_KINDS:
            with self.subTest(kind=kind):
                directory = self.skill / kind
                directory.mkdir()
                (directory / "__pycache__").mkdir()
                self.assertEqual(self.collect()["artifacts"][kind], 0)

    def test_real_files_and_subdirectories_still_count(self):
        graders = self.skill / "graders"
        graders.mkdir()
        (graders / "check.py").write_text("pass\n", encoding="utf-8")
        (graders / "helpers").mkdir()
        self.assertEqual(self.collect()["artifacts"]["graders"], 2)


if __name__ == "__main__":
    unittest.main()
