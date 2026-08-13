#!/usr/bin/env python3
"""Deterministic grader for the adaption-ai augmentation pipeline task.

Runs in the workspace after the agent finishes. Prints JSON to stdout:
{"score": 0.0-1.0, "details": str, "checks": [{"name", "passed", "message"}]}
"""
import glob
import json
import os
import sys


def load_source():
    if os.path.exists("pipeline.py"):
        return open("pipeline.py", encoding="utf-8").read()
    for f in glob.glob("*.py"):
        return open(f, encoding="utf-8").read()
    return ""


def main():
    src = load_source()
    # Normalize quote style so 'prompt' and "prompt" both match.
    normalized = src.replace('"', "'")
    checks = [
        ("client from env key", "ADAPTION_API_KEY" in src or "api_key" in src),
        ("upload_file", "upload_file" in src),
        ("run with column_mapping", "column_mapping" in src),
        ("prompt mapping key", "'prompt'" in normalized),
        ("hallucination mitigation", "hallucination_mitigation" in src),
        ("recipe specification", "recipe_specification" in src),
        ("max_rows cap", "max_rows" in src),
        ("idempotency key", "idempotency_key" in src),
        ("estimate before run", "estimate" in src),
        ("wait for completion", "wait_for_completion" in src),
        ("quality evaluation", "get_evaluation" in src),
        ("download result", "download" in src),
    ]
    passed = [name for name, ok in checks if ok]
    score = len(passed) / len(checks)
    result = {
        "score": round(score, 3),
        "details": f"{len(passed)}/{len(checks)} checks passed",
        "checks": [
            {"name": name, "passed": ok,
             "message": "found" if ok else "missing"}
            for name, ok in checks
        ],
    }
    print(json.dumps(result))
    sys.exit(0 if score >= 0.8 else 1)


if __name__ == "__main__":
    main()
