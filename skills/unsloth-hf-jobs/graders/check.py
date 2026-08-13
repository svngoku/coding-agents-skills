#!/usr/bin/env python3
"""Deterministic grader for the unsloth-hf-jobs train.sh task.

Runs in the workspace after the agent finishes. Prints JSON to stdout:
{"score": 0.0-1.0, "details": str, "checks": [{"name", "passed", "message"}]}

Checks the API surface the skill teaches (remote script URL, job-level options,
the "--" separator, script-level training options, smoke-test variant), not
cosmetic formatting.
"""
import glob
import json
import os
import re
import sys


def load_script():
    """Find the agent's train.sh. Prefer train.sh, then any reference-*
    solution staged by skillgrade --validate, then any other .sh file."""
    if os.path.exists("train.sh"):
        return open("train.sh", encoding="utf-8").read()
    for pattern in ("reference-*", "solutions/reference-*", "*.sh"):
        for f in glob.glob(pattern):
            return open(f, encoding="utf-8").read()
    return ""


def flag_value(text, flag, value):
    """True if `flag` is followed by exactly `value` (also matches --flag=value)."""
    return re.search(
        r"{}(?:[\s=]+){}(?=\s|$)".format(re.escape(flag), re.escape(value)),
        text,
    ) is not None


def has_value(text, flag):
    """True if `flag` is present and followed by a non-empty, non-flag value."""
    m = re.search(r"{}(?:\s+)(\S+)".format(re.escape(flag)), text)
    return bool(m) and not m.group(1).startswith("--")


def main():
    src = load_script()
    if not src:
        print(json.dumps({
            "score": 0.0,
            "details": "train.sh not found in the workspace",
            "checks": [
                {"name": "train.sh present", "passed": False,
                 "message": "no train.sh (or reference solution) found"}
            ],
        }))
        sys.exit(1)

    # Normalize: join backslash line continuations, collapse whitespace, and
    # accept --flag=value syntax so checks are robust but still exact on values.
    text = src.replace("\\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(--[a-z0-9-]+)=", r"\1 ", text)

    url = "https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/"
    trackio_ok = any(
        m.group(1).endswith("/trackio")
        for m in re.finditer(r"--trackio-space\s+(\S+)", text)
    )
    lr_ok = re.search(r"--learning-rate\s+(?:2e-4|0\.0002)(?=\s|$)", text) is not None
    ev_ok = re.search(r"--eval-split\s+(?:0\.1|\.1)(?=\s|$)", text) is not None

    checks = [
        ("remote script URL", url in text),
        ("sft-qwen3-vl.py script", "sft-qwen3-vl.py" in text),
        ("two hf jobs uv run commands", text.count("hf jobs uv run") >= 2),
        ("-- separator before script flags", " -- --" in text),
        ("--flavor a100-large", flag_value(text, "--flavor", "a100-large")),
        ("--secrets HF_TOKEN", flag_value(text, "--secrets", "HF_TOKEN")),
        ("--timeout 4h", flag_value(text, "--timeout", "4h")),
        ("--dataset with value", has_value(text, "--dataset")),
        ("--output-repo with value", has_value(text, "--output-repo")),
        ("--lora-r 16", flag_value(text, "--lora-r", "16")),
        ("--learning-rate 2e-4", lr_ok),
        ("--batch-size 2", flag_value(text, "--batch-size", "2")),
        ("--gradient-accumulation 4", flag_value(text, "--gradient-accumulation", "4")),
        ("--eval-split 0.1", ev_ok),
        ("--merge-model", "--merge-model" in text),
        ("--trackio-space <user>/trackio", trackio_ok),
        ("smoke-test --max-steps 10", flag_value(text, "--max-steps", "10")),
    ]

    passed = [name for name, ok in checks if ok]
    score = len(passed) / len(checks)
    result = {
        "score": round(score, 3),
        "details": "{}/{} checks passed".format(len(passed), len(checks)),
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
