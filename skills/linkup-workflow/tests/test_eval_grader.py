"""Public-boundary regressions. Only trusted solution/grader programs execute."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "solutions/reference-plan"
GRADER = ROOT / "graders/check.py"


class GraderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        subprocess.run([str(REFERENCE)],
                       cwd=self.workspace, check=True, capture_output=True, timeout=10)

    def grade(self):
        result = subprocess.run([sys.executable, "-I", "-S", str(GRADER)],
                                cwd=self.workspace, check=True, capture_output=True,
                                text=True, timeout=10)
        return json.loads(result.stdout)

    def plan(self):
        return json.loads((self.workspace / "request-plan.json").read_text())

    def write(self, plan):
        (self.workspace / "request-plan.json").write_text(json.dumps(plan))

    def assert_rejected(self):
        result = self.grade()
        self.assertLessEqual(result["score"], 0.6, result)

    def test_reference(self):
        result = self.grade()
        self.assertEqual(result["score"], 1.0, result)
        self.assertEqual(len(result["checks"]), 5)

    def test_reordered_properties_and_policy_lists(self):
        def reverse(value):
            if isinstance(value, dict):
                return {key: reverse(item) for key, item in reversed(list(value.items()))}
            if isinstance(value, list):
                return [reverse(item) for item in value]
            return value
        plan = reverse(self.plan())
        plan["policy"]["writeTargets"].reverse()
        self.write(plan)
        self.assertEqual(self.grade()["score"], 1.0)

    def test_case_insensitive_http_header_names(self):
        plan = self.plan()
        plan["transport"]["headers"] = {k.lower(): v for k, v in plan["transport"]["headers"].items()}
        self.write(plan)
        self.assertEqual(self.grade()["score"], 1.0)

    def test_missing_json(self):
        (self.workspace / "request-plan.json").unlink()
        self.assertEqual(self.grade()["score"], 0.0)

    def test_missing_design(self):
        (self.workspace / "design.md").unlink()
        self.assertEqual(self.grade()["score"], 0.0)

    def test_empty_design(self):
        (self.workspace / "design.md").write_text("")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_malformed_json(self):
        (self.workspace / "request-plan.json").write_text("{ broken")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_invalid_utf8(self):
        (self.workspace / "request-plan.json").write_bytes(b"\xff")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_non_object_json(self):
        (self.workspace / "request-plan.json").write_text("[]")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_duplicate_json_keys(self):
        source = (self.workspace / "request-plan.json").read_text()
        (self.workspace / "request-plan.json").write_text(source.replace('"version": 1', '"version": 1, "version": 1'))
        self.assertEqual(self.grade()["score"], 0.0)

    def test_non_finite_json(self):
        source = (self.workspace / "request-plan.json").read_text()
        (self.workspace / "request-plan.json").write_text(source.replace('"version": 1', '"version": NaN'))
        self.assertEqual(self.grade()["score"], 0.0)

    def test_oversized_json(self):
        (self.workspace / "request-plan.json").write_text(" " * 65537 + "{}")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_oversized_design(self):
        (self.workspace / "design.md").write_text("x" * 16385)
        self.assertEqual(self.grade()["score"], 0.0)

    def test_symlink_json(self):
        path = self.workspace / "request-plan.json"
        target = self.workspace / "target.json"
        path.rename(target)
        path.symlink_to(target)
        self.assertEqual(self.grade()["score"], 0.0)

    def test_symlink_design(self):
        path = self.workspace / "design.md"
        target = self.workspace / "target.md"
        path.rename(target)
        path.symlink_to(target)
        self.assertEqual(self.grade()["score"], 0.0)

    def test_directory_artifact(self):
        path = self.workspace / "request-plan.json"
        path.unlink()
        path.mkdir()
        self.assertEqual(self.grade()["score"], 0.0)

    def test_candidate_python_never_executes(self):
        (self.workspace / "candidate.py").write_text("raise RuntimeError('candidate code executed')")
        (self.workspace / "json.py").write_text("raise RuntimeError('candidate module imported')")
        (self.workspace / "sitecustomize.py").write_text("raise RuntimeError('candidate startup executed')")
        self.assertEqual(self.grade()["score"], 1.0)


    def test_exponent_overflow_rejected(self):
        source = (self.workspace / "request-plan.json").read_text()
        path = self.workspace / "request-plan.json"
        path.write_text(source.replace('"timeoutSeconds": 30', '"timeoutSeconds": 1e999', 1))
        self.assertEqual(self.grade()["score"], 0.0)

    def test_enormous_integer_does_not_crash(self):
        plan = self.plan()
        plan["transport"]["timeoutSeconds"] = 10 ** 1000
        self.write(plan)
        self.assert_rejected()

    def test_invalid_design_utf8(self):
        (self.workspace / "design.md").write_bytes(b"\xff")
        self.assertEqual(self.grade()["score"], 0.0)

    def test_reordered_schema_requirements(self):
        plan = self.plan()
        request = plan.get("request", plan.get("submit", {}))
        body = request.get("body", {})
        serialized = "structuredOutputSchema" in body
        schema = json.loads(body["structuredOutputSchema"]) if serialized else body.get("schema")
        if schema is None:
            return
        def reorder(value):
            if isinstance(value, dict):
                result = {key: reorder(item) for key, item in reversed(list(value.items()))}
                if "required" in result:
                    result["required"].reverse()
                if isinstance(result.get("type"), list):
                    result["type"].reverse()
                return result
            if isinstance(value, list):
                return [reorder(item) for item in value]
            return value
        reordered = reorder(schema)
        if serialized:
            body["structuredOutputSchema"] = json.dumps(reordered)
        else:
            body["schema"] = reordered
        self.write(plan)
        self.assertEqual(self.grade()["score"], 1.0)


def mutation_test(path, value):
    def test(self):
        plan = self.plan()
        parent = plan
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        self.write(plan)
        self.assert_rejected()
    return test


def deletion_test(path):
    def test(self):
        plan = self.plan()
        parent = plan
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
        self.write(plan)
        self.assert_rejected()
    return test

MUTATIONS = {'ambiguous_url_first_hit': (['bindings', 'officialUrl', 'onAmbiguous'], 'first-hit'),
 'binding_in_api_body': (['steps', 1, 'request', 'body', 'officialUrl'], '${officialUrl}'),
 'boolean_version': (['version'], True),
 'budget_exceeded': (['policy', 'maxRequests'], 3),
 'crm_write': (['policy', 'externalWrites'], ['crm']),
 'discarded_sources': (['policy', 'sourcePolicy'], 'discard'),
 'dropped_discovery_sources': (['result', 'discoverySourcesFrom'], ''),
 'dropped_sources': (['result', 'preserveSources'], False),
 'extra_output_write': (['policy', 'writeTargets'],
                        ['request-plan.json', 'design.md', '../credentials']),
 'guessed_url': (['steps', 1, 'request', 'body', 'url'], 'https://northstar.example/careers'),
 'invented_mcp_tool': (['steps', 0, 'tool'], 'linkup-workflow'),
 'invented_workflow_endpoint': (['steps', 0, 'request', 'path'], '/v1/workflow'),
 'missing_auth_placeholder': (['transport', 'headers'], {'Content-Type': 'application/json'}),
 'missing_url_guess': (['bindings', 'officialUrl', 'onMissing'], 'guess-url'),
 'network_during_planning': (['policy', 'networkDuringPlanning'], True),
 'no_fetch_dependency': (['steps', 1, 'dependsOn'], []),
 'no_official_evidence': (['bindings', 'officialUrl', 'requireOfficialDomainEvidence'], False),
 'no_rendering': (['steps', 1, 'request', 'body', 'renderJs'], False),
 'parallel_tasks_wrapper': (['steps', 0, 'request', 'path'], '/v1/tasks'),
 'real_credential': (['transport', 'headers', 'Authorization'], 'Bearer not-a-real-secret'),
 'rest_query_instead_of_q': (['steps', 0, 'request', 'body', 'query'], 'Wrong REST field'),
 'sdk_fetch_casing': (['steps', 1, 'request', 'body', 'render_js'], True),
 'unbounded_request_timeout': (['transport', 'timeoutSeconds'], 0),
 'unbounded_retries': (['policy', 'maxRetries'], 100),
 'unknown_planner_endpoint': (['endpoint'], '/v1/workflow'),
 'unsafe_url_scheme': (['bindings', 'officialUrl', 'requireHttps'], False),
 'write_handoff': (['result', 'handoff'], 'write-crm'),
 'wrong_base_url': (['transport', 'baseUrl'], 'http://api.linkup.so'),
 'wrong_binding_step': (['bindings', 'officialUrl', 'fromStep'], 'fetch'),
 'wrong_discovery_depth': (['steps', 0, 'request', 'body', 'depth'], 'flash'),
 'wrong_fetch_endpoint': (['steps', 1, 'request', 'path'], '/v1/workflow'),
 'wrong_page_markdown': (['result', 'pageMarkdownFrom'], 'fetch.content'),
 'wrong_q_casing': (['steps', 0, 'request', 'body', 'Q'], 'Wrong REST casing'),
 'wrong_results_object_shape': (['bindings', 'officialUrl', 'fromPath'], '$'),
 'wrong_search_output': (['steps', 0, 'request', 'body', 'outputType'], 'sourcedAnswer')}
DELETIONS = {'required_query_or_url': ['steps', 0, 'request', 'body', 'q'],
 'required_result': ['result'],
 'required_transport': ['transport']}

MUTATIONS.update({'boolean_timeout': (['transport', 'timeoutSeconds'], True),
 'ignore_errors': (['policy', 'onError'], 'ignore'),
 'inline_credential_source': (['policy', 'credentialSource'], 'inline'),
 'invented_missing_values': (['policy', 'missingValuePolicy'], 'invent'),
 'wrong_content_type': (['transport', 'headers', 'Content-Type'], 'text/plain'),
 'wrong_page_url_mapping': (['result', 'pageUrlFrom'], 'fetch.url'),
 'wrong_url_selection': (['bindings', 'officialUrl', 'select'], 'first-result')})
DELETIONS["required_policy"] = ["policy"]

for name, (path, value) in MUTATIONS.items():
    setattr(GraderTests, "test_reject_" + name, mutation_test(path, value))
for name, path in DELETIONS.items():
    setattr(GraderTests, "test_missing_" + name, deletion_test(path))

if __name__ == "__main__":
    unittest.main()
