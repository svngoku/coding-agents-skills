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

    def test_lower_positive_polling_rate_is_safe(self):
        plan = self.plan()
        plan["poll"]["maxRequestsPerSecond"] = 0.001
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

MUTATIONS = {'ambiguous_submit_retries': (['poll', 'onAmbiguousSubmit'], 'retry-post'),
 'array_schema': (['submit', 'body', 'schema', 'type'], 'array'),
 'boolean_attempt_cap': (['poll', 'maxAttempts'], True),
 'boolean_version': (['version'], True),
 'cancel_discards_id': (['poll', 'onCancel'], 'discard-task-id'),
 'completed_keeps_polling': (['poll', 'states', 'completed'], 'wait'),
 'credential_leak_to_download': (['result', 'download', 'headers'],
                                 {'Authorization': 'Bearer ${LINKUP_API_KEY}'}),
 'discarded_sources': (['policy', 'sourcePolicy'], 'discard'),
 'download_follows_redirects': (['result', 'download', 'followRedirects'], True),
 'duplicate_submissions': (['policy', 'maxSubmissions'], 2),
 'expired_url_resubmits': (['result', 'download', 'onExpiredUrl'], 'submit-again'),
 'extra_output_write': (['policy', 'writeTargets'],
                        ['request-plan.json', 'design.md', '../credentials']),
 'failed_task_reads_output': (['poll', 'states', 'failed'], 'read-output'),
 'forbidden_retries': (['policy', 'onForbidden'], 'retry'),
 'get_has_body': (['poll', 'body'], {'id': '${taskId}'}),
 'inline_rows_fiction': (['result', 'resultUrlPath'], 'output.rows'),
 'invented_output_type': (['submit', 'body', 'outputType'], 'structured'),
 'json_array_not_ndjson': (['result', 'download', 'format'], 'json'),
 'list_not_poll': (['poll', 'path'], '/v1/extract'),
 'missing_auth_placeholder': (['transport', 'headers'], {'Content-Type': 'application/json'}),
 'multiple_downloads': (['result', 'download', 'maxDownloads'], 100),
 'network_during_planning': (['policy', 'networkDuringPlanning'], True),
 'no_attempt_cap': (['poll', 'maxAttempts'], 1000000),
 'no_deadline': (['poll', 'deadlineSeconds'], 0),
 'no_persistence': (['result', 'download', 'persistBeforeExpiry'], False),
 'no_saved_id': (['poll', 'persistTaskIdBeforePolling'], False),
 'non_https_download': (['result', 'download', 'requireHttps'], False),
 'non_object_lines': (['result', 'download', 'recordType'], 'array'),
 'pending_reads_output': (['poll', 'states', 'pending'], 'read-output'),
 'permanent_result_url': (['result', 'download', 'urlLifetimeHours'], 8760),
 'poll_post': (['poll', 'method'], 'POST'),
 'poll_too_fast': (['poll', 'maxRequestsPerSecond'], 2),
 'poll_wrong_id_source': (['poll', 'taskIdFrom'], 'new-request.id'),
 'processing_resubmits': (['poll', 'states', 'processing'], 'submit-again'),
 'real_credential': (['transport', 'headers', 'Authorization'], 'Bearer not-a-real-secret'),
 'rest_query_instead_of_q': (['submit', 'body', 'query'], 'Wrong REST field'),
 'resubmit_active_task': (['poll', 'resubmitWhileActive'], True),
 'serialized_schema': (['submit', 'body', 'schema'], '{"type":"object"}'),
 'timeout_resubmits': (['poll', 'onTimeout'], 'submit-again'),
 'too_fast_poll_schedule': (['poll', 'initialDelaySeconds'], 0.1),
 'unbounded_download': (['result', 'download', 'maxBytes'], 0),
 'unbounded_poll_delay': (['poll', 'maxDelaySeconds'], 100000),
 'unbounded_request_timeout': (['transport', 'timeoutSeconds'], 0),
 'unbounded_retries': (['policy', 'maxRetries'], 100),
 'unconfirmed_beta': (['policy', 'betaAccess'], 'assume-open-access'),
 'unknown_body_field': (['submit', 'body', 'includeSources'], True),
 'unknown_planner_endpoint': (['endpoint'], '/v1/workflow'),
 'unknown_status_loops': (['poll', 'onUnknownStatus'], 'wait'),
 'unsafe_output_path': (['result', 'download', 'saveAs'], '../events.ndjson'),
 'unseeded_homepage': (['submit', 'body', 'url'], 'https://example.com'),
 'unverified_urls': (['submit', 'body', 'verifyUrls'], False),
 'wrong_backoff': (['poll', 'backoffMultiplier'], 2),
 'wrong_base_url': (['transport', 'baseUrl'], 'http://api.linkup.so'),
 'wrong_credits': (['result', 'creditsUsedPath'], 'input.price'),
 'wrong_endpoint': (['submit', 'path'], '/v1/search'),
 'wrong_q_casing': (['submit', 'body', 'Q'], 'Wrong REST casing'),
 'wrong_rows_count': (['result', 'rowsReturnedPath'], 'output.rows.length')}
DELETIONS = {'required_poll': ['poll'],
 'required_query_or_url': ['submit', 'body', 'q'],
 'required_result': ['result'],
 'required_schema': ['submit', 'body', 'schema'],
 'required_transport': ['transport']}

MUTATIONS.update({'boolean_timeout': (['transport', 'timeoutSeconds'], True),
 'ignore_errors': (['policy', 'onError'], 'ignore'),
 'inline_credential_source': (['policy', 'credentialSource'], 'inline'),
 'invented_missing_values': (['policy', 'missingValuePolicy'], 'invent'),
 'unbounded_download_timeout': (['result', 'download', 'timeoutSeconds'], 0),
 'wrong_content_type': (['transport', 'headers', 'Content-Type'], 'text/plain'),
 'wrong_download_method': (['result', 'download', 'method'], 'POST'),
 'wrong_download_source': (['result', 'download', 'urlFrom'], 'input.url')})
DELETIONS["required_policy"] = ["policy"]

for name, (path, value) in MUTATIONS.items():
    setattr(GraderTests, "test_reject_" + name, mutation_test(path, value))
for name, path in DELETIONS.items():
    setattr(GraderTests, "test_missing_" + name, deletion_test(path))

if __name__ == "__main__":
    unittest.main()
