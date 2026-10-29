"""End-to-end content-pipeline regressions for the five Linkup skills."""

import importlib.util
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
NAMES = ("search", "fetch", "research", "extract", "workflow")
SPEC = importlib.util.spec_from_file_location("linkup_site", ROOT / "scripts/build-site.py")
assert SPEC is not None and SPEC.loader is not None
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)


class LinkupMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(SITE, "repo_branch", return_value="main"):
            cls.skills = {skill["name"]: skill for skill in SITE.collect_skills()}

    def test_each_skill_has_its_own_title_category_and_description(self):
        descriptions = set()
        for suffix in NAMES:
            name = f"linkup-{suffix}"
            with self.subTest(skill=name):
                skill = self.skills[name]
                self.assertEqual(skill["title"], f"Linkup {suffix.title()}")
                self.assertEqual(skill["category"], "AI & Agents")
                self.assertTrue(skill["capabilities"])
                self.assertTrue(skill["useCases"])
                descriptions.add(skill["description"])
        self.assertEqual(len(descriptions), len(NAMES))

    def test_skill_structure_and_loader_triggers(self):
        for suffix in NAMES:
            name = f"linkup-{suffix}"
            text = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
            with self.subTest(skill=name):
                front = SITE.parse_frontmatter_block(text)
                self.assertEqual(front["name"], name)
                self.assertIn("description: >\n", text)
                self.assertIn("Also trigger for", front["description"])
                self.assertLess(len(front["description"]), 500)
                self.assertLess(len(text.splitlines()), 300)
                for heading in ("Quick Reference", "Core Workflow", "Anti-Patterns to Avoid", "When to Use / Not Use"):
                    self.assertIn(f"## {heading}", text)

    def test_all_skill_links_resolve_locally(self):
        for suffix in NAMES:
            directory = ROOT / "skills" / f"linkup-{suffix}"
            text = (directory / "SKILL.md").read_text(encoding="utf-8")
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
            with self.subTest(skill=directory.name):
                self.assertTrue(links)
                for link in links:
                    self.assertNotRegex(link, r"^(?:https?:|/)")
                    self.assertTrue((directory / link.split("#")[0]).is_file(), link)

    def test_every_bundled_guide_is_readable_in_the_gallery(self):
        for suffix in NAMES:
            name = f"linkup-{suffix}"
            directory = ROOT / "skills" / name
            expected = {
                path.relative_to(directory).as_posix()
                for path in (directory / "references").rglob("*.md")
            }
            documents = self.skills[name]["documents"]
            with self.subTest(skill=name):
                self.assertTrue(expected)
                self.assertEqual(expected, {doc["path"] for doc in documents if doc["kind"] == "guide"})
                self.assertTrue(all(doc["html"].strip() for doc in documents))
                for path in expected:
                    self.assertRegex(Path(path).name, r"^[a-z0-9]+(?:-[a-z0-9]+)*\.md$")

    def test_provenance_is_pinned_and_self_contained(self):
        for suffix in NAMES:
            directory = ROOT / "skills" / f"linkup-{suffix}"
            with self.subTest(skill=directory.name):
                source = (directory / "references/sources-and-compatibility.md").read_text(encoding="utf-8")
                self.assertIn("LinkupPlatform/skills", source)
                self.assertRegex(source, r"\b[0-9a-f]{40}\b")
                self.assertIn("2026-10-29", source)
                self.assertTrue((directory / "LICENSE").is_file())

    def test_all_five_harnesses_are_discoverable(self):
        for suffix in NAMES:
            name = f"linkup-{suffix}"
            with self.subTest(skill=name):
                info = self.skills[name]["eval"]
                self.assertTrue(info["present"])
                self.assertTrue(info["tasks"])
                self.assertEqual(info["threshold"], 0.8)
                directory = ROOT / "skills" / name
                for kind in ("instructions", "graders", "rubrics", "solutions"):
                    self.assertTrue(any((directory / kind).iterdir()))

    def test_source_links_target_main_not_the_work_branch(self):
        for suffix in NAMES:
            skill = self.skills[f"linkup-{suffix}"]
            with self.subTest(skill=skill["name"]):
                self.assertIn("/tree/main/skills/", skill["dirUrl"])
                self.assertTrue(all("/blob/main/skills/" in doc["url"] for doc in skill["documents"]))

    def test_complete_rest_search_examples_use_current_wire_shape(self):
        for suffix in ("search", "fetch"):
            path = ROOT / "skills" / f"linkup-{suffix}" / "references/linkup-api-reference.md"
            examples = [
                json.loads(body)
                for body in re.findall(r"```json\n(.*?)\n```", path.read_text(), re.S)
            ]
            requests = [body for body in examples if isinstance(body, dict) and ("q" in body or "query" in body)]
            with self.subTest(skill=suffix):
                self.assertGreaterEqual(len(requests), 2)
                for body in requests:
                    self.assertIn("q", body)
                    self.assertNotIn("query", body)
                    self.assertIn(body["depth"], ("flash", "fast", "standard", "deep"))
                    self.assertIn("outputType", body)
                    if body["outputType"] == "structured":
                        self.assertIsInstance(body["structuredOutputSchema"], str)
                        self.assertEqual(json.loads(body["structuredOutputSchema"])["type"], "object")

    def test_concrete_research_examples_serialize_wire_schema(self):
        for suffix in ("research", "extract"):
            path = ROOT / "skills" / f"linkup-{suffix}" / "references/linkup-specialized-endpoints.md"
            bodies = re.findall(r"POST /v1/research\n(\{\n.*?\n\})", path.read_text(), re.S)
            complete = [json.loads(body) for body in bodies if "..." not in body]
            with self.subTest(skill=suffix):
                self.assertGreaterEqual(len(complete), 3)
                structured = [body for body in complete if body["outputType"] == "structured"]
                self.assertTrue(structured)
                for body in structured:
                    self.assertIsInstance(body["structuredOutputSchema"], str)
                    self.assertEqual(json.loads(body["structuredOutputSchema"])["type"], "object")

    def test_recipe_queries_are_quoted_yaml_strings(self):
        directory = ROOT / "skills/linkup-workflow/references/workflows"
        for path in directory.glob("*/*.md"):
            with self.subTest(recipe=path.name):
                queries = [
                    value
                    for fence in re.findall(r"```yaml\n(.*?)\n```", path.read_text(), re.S)
                    for value in re.findall(r"^\s+q: (.+)$", fence, re.M)
                ]
                self.assertTrue(queries)
                for value in queries:
                    self.assertIsInstance(json.loads(value), str)

    def test_multi_page_scrape_recipes_use_deep(self):
        directory = ROOT / "skills/linkup-workflow/references/workflows"
        targets = {
            "marketing/competitor-messaging-scan.md": "Scrape competitor homepages",
            "research/competitor-tracker.md": "Scrape known competitor pages",
        }
        for file, step in targets.items():
            with self.subTest(recipe=file):
                block = re.search(rf"name: {re.escape(step)}\n.*?```", (directory / file).read_text(), re.S)
                self.assertIsNotNone(block)
                self.assertIn("depth: deep", block.group())
                self.assertNotIn("depth: standard", block.group())

    def test_workflow_routes_to_other_skills_not_missing_files(self):
        directory = ROOT / "skills/linkup-workflow/references"
        for file in ("linkup-workflow-guide.md", "linkup-prompt-optimizer-knowledge.md"):
            text = (directory / file).read_text()
            with self.subTest(guide=file):
                self.assertIn("separately available `linkup-search` skill", text)
                self.assertNotIn("`linkup-agent-query-mental-model.md`", text)
                self.assertNotIn("`linkup-specialized-endpoints.md`", text)


if __name__ == "__main__":
    unittest.main()
