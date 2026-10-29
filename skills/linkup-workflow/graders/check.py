#!/usr/bin/env python3
"""Offline JSON-plan grading only: never import or execute candidate code."""

import json
import math
from pathlib import Path
import stat


def keys(value, expected):
    return isinstance(value, dict) and set(value) == set(expected)


def number(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def text_value(value):
    return isinstance(value, str) and 30 <= len(value.strip()) <= 4000


def names(value, expected):
    return isinstance(value, list) and all(isinstance(x, str) for x in value) and (
        len(value) == len(expected) and set(value) == set(expected)
    )


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("non-finite JSON number")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def parse_json(value):
    return json.loads(value, object_pairs_hook=no_duplicates, parse_constant=invalid_constant,
                      parse_float=finite_float)


def artifact(root, name, limit):
    path = root / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("artifact must be a bounded regular file, not a symlink")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("oversized artifact")
    return raw.decode("utf-8")


def transport(value):
    if not keys(value, ("baseUrl", "headers", "timeoutSeconds")):
        return False
    headers = value["headers"]
    if not isinstance(headers, dict) or not all(isinstance(k, str) for k in headers):
        return False
    folded = {k.lower(): v for k, v in headers.items()}
    return (
        value["baseUrl"] == "https://api.linkup.so"
        and len(folded) == len(headers)
        and folded == {
            "authorization": "Bearer ${LINKUP_API_KEY}",
            "content-type": "application/json",
        }
        and number(value["timeoutSeconds"], 1, 60)
    )


def request(value, path, body_keys):
    return (
        keys(value, ("method", "path", "body"))
        and value["method"] == "POST" and value["path"] == path
        and keys(value["body"], body_keys)
    )


def policy(value, extra):
    common = {
        "networkDuringPlanning": False,
        "credentialSource": "environment",
        "maxRetries": 0,
        "sourcePolicy": "preserve",
        "onError": "stop-and-report",
    }
    expected = dict(common, **extra)
    if not keys(value, tuple(expected) + ("writeTargets",)):
        return False
    return (
        all(type(value[k]) is type(v) and value[k] == v for k, v in expected.items())
        and names(value["writeTargets"], ("request-plan.json", "design.md"))
    )


def grade(root):
    try:
        plan = parse_json(artifact(root, "request-plan.json", 65536))
        design = artifact(root, "design.md", 16384)
        if not isinstance(plan, dict) or len(design.strip()) < 32:
            raise ValueError("expected a JSON object and a nonempty design note")
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        return {"score": 0.0, "details": str(error), "checks": []}

    checks = []
    def check(name, callback):
        try:
            passed = bool(callback())
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, RecursionError):
            passed = False
        checks.append({"name": name, "passed": passed})

    check("artifact-boundary", lambda: keys(plan, ROOT_KEYS) and type(plan["version"]) is int
          and plan["version"] == 1)
    check("transport-and-auth", lambda: transport(plan["transport"]))
    check("request-contract", lambda: request_contract(plan))
    check("result-and-lifecycle", lambda: result_contract(plan))
    check("bounded-source-preserving-policy", lambda: policy_contract(plan))
    count = sum(item["passed"] for item in checks)
    # Every group is essential: even a perfect 30% rubric cannot erase a failed group.
    score = 1.0 if count == len(checks) else min(0.6, count / len(checks))
    return {"score": score, "details": f"{count}/{len(checks)} essential groups passed",
            "checks": checks}

ROOT_KEYS = ("version", "transport", "steps", "bindings", "result", "policy")


def request_contract(plan):
    steps = plan["steps"]
    if not isinstance(steps, list) or len(steps) != 2:
        return False
    discover, fetch = steps
    if not (
        keys(discover, ("id", "request")) and discover["id"] == "discover"
        and keys(fetch, ("id", "dependsOn", "request")) and fetch["id"] == "fetch"
        and names(fetch["dependsOn"], ("discover",))
        and request(discover["request"], "/v1/search", ("q", "depth", "outputType"))
        and request(fetch["request"], "/v1/fetch", ("url", "renderJs"))
    ):
        return False
    search_body, fetch_body = discover["request"]["body"], fetch["request"]["body"]
    return (
        text_value(search_body["q"]) and search_body["depth"] == "standard"
        and search_body["outputType"] == "searchResults"
        and fetch_body["url"] == "${officialUrl}" and fetch_body["renderJs"] is True
    )


def result_contract(plan):
    if not keys(plan["bindings"], ("officialUrl",)):
        return False
    binding = plan["bindings"]["officialUrl"]
    return (
        keys(binding, (
            "fromStep", "fromPath", "select", "requireOfficialDomainEvidence",
            "requireHttps", "onMissing", "onAmbiguous",
        ))
        and binding["fromStep"] == "discover" and binding["fromPath"] == "results"
        and binding["select"] == "single-official-careers-url"
        and binding["requireOfficialDomainEvidence"] is True
        and binding["requireHttps"] is True
        and binding["onMissing"] == "stop-no-fetch" and binding["onAmbiguous"] == "stop-no-fetch"
        and keys(plan["result"], (
            "pageMarkdownFrom", "pageUrlFrom", "discoverySourcesFrom", "preserveSources", "handoff"
        ))
        and plan["result"]["pageMarkdownFrom"] == "fetch.markdown"
        and plan["result"]["pageUrlFrom"] == "bindings.officialUrl"
        and plan["result"]["discoverySourcesFrom"] == "discover.results"
        and plan["result"]["preserveSources"] is True
        and plan["result"]["handoff"] == "draft-only"
    )


def policy_contract(plan):
    return policy(plan["policy"], {
        "maxRequests": 2, "missingValuePolicy": "report-unknown", "externalWrites": [],
    })


if __name__ == "__main__":
    print(json.dumps(grade(Path.cwd()), allow_nan=False))
