"""Regression tests for reproducible gallery metadata."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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

    def test_nested_reference_guides_are_counted_and_rendered(self):
        references = self.skill / "references"
        nested = references / "workflows" / "sales"
        nested.mkdir(parents=True)
        (references / "overview.md").write_text("# Overview\n", encoding="utf-8")
        (nested / "example.md").write_text("# Example\n\nA nested recipe.\n", encoding="utf-8")
        cache = references / "__pycache__"
        cache.mkdir()
        (cache / "ignored.md").write_text("# Not a guide\n", encoding="utf-8")
        skill = self.collect()
        self.assertEqual(skill["artifacts"]["references"], 2)
        self.assertEqual(
            [ref["file"] for ref in skill["references"]],
            ["overview.md", "workflows/sales/example.md"],
        )
        document = next(doc for doc in skill["documents"] if doc["path"].endswith("example.md"))
        self.assertEqual(document["url"], "https://example.com/repo/blob/main/skills/example/references/workflows/sales/example.md")
        self.assertIn("A nested recipe.", document["html"])


class ReadmeMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skills = self.root / "skills"
        self.skills.mkdir()

    def collect(self, names, sections):
        for name in names:
            directory = self.skills / name
            directory.mkdir()
            (directory / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Fallback {name}\n---\n# {name}\n",
                encoding="utf-8",
            )
        (self.root / "README.md").write_text(sections, encoding="utf-8")
        with patch.object(SITE, "REPO_ROOT", self.root), patch.object(
            SITE, "SKILLS_DIR", self.skills
        ), patch.object(SITE, "repo_url", return_value="https://example.com/repo"), patch.object(
            SITE, "repo_branch", return_value="main"
        ):
            return {skill["name"]: skill for skill in SITE.collect_skills()}

    def test_shared_prefix_skills_keep_their_own_metadata(self):
        # README order deliberately differs from alphabetical directory order.
        names = ["linkup-search", "linkup-fetch", "linkup-research", "linkup-extract", "linkup-workflow"]
        sections = "\n".join(
            f"### {index}. {name.replace('-', ' ').title()}\n"
            f"**Description:** Description for {name}\n"
            f"**Capabilities:**\n- Capability for {name}\n"
            f"**Use Cases:**\n- Use case for {name}\n---\n"
            for index, name in enumerate(names, 1)
        )
        result = self.collect(names, sections)
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(result[name]["description"], f"Description for {name}")
                self.assertEqual(result[name]["capabilities"], [f"Capability for {name}"])
                self.assertEqual(result[name]["useCases"], [f"Use case for {name}"])

    def test_existing_provider_name_fallback_is_preserved(self):
        result = self.collect(
            ["typesafe-ai"],
            "### 1. Build with TypeSafe and Jev\n**Description:** Typed judgments\n",
        )
        self.assertEqual(result["typesafe-ai"]["description"], "Typed judgments")

    def test_repeated_exact_title_does_not_steal_another_skill(self):
        result = self.collect(
            ["linkup-fetch", "linkup-search"],
            "### 1. Linkup Search\n**Description:** Search\n"
            "### 2. Linkup Search\n**Description:** Duplicate\n",
        )
        self.assertEqual(result["linkup-search"]["description"], "Search")
        self.assertEqual(result["linkup-fetch"]["description"], "Fallback linkup-fetch")


if __name__ == "__main__":
    unittest.main()
