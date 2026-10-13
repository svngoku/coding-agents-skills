#!/usr/bin/env python3
"""Deterministic grader for the prime-rl "reverse-text-grpo-run" eval task.

Runs from the agent's workspace (cwd) and checks two artifacts:

  rl.toml   a `uv run rl` config: reverse-text train/eval sources written as
            dotted keys, group_size > 1, IcePop loss, checkpoint retention,
            W&B project, explicit renderer, managed inference, no inline tables
  run.sh    dry run, named launch, absolute-step resume, DCP -> HF export,
            all through `uv run` (no torchrun / vllm serve / raw python)

Python 3.11+ standard library only (tomllib). Prints JSON to stdout:
  {"score": 0.95, "details": "20/21 checks passed", "checks": [...]}
Exit code 1 if the score is below the pass bar (0.8).
"""

import glob
import json
import os
import re
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

PASS_BAR = 0.8
TOML_NAME = "rl.toml"
SH_NAME = "run.sh"
MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
TASKSET = "reverse-text"
RUN_NAME = "rt-grpo"


def find(name):
    """Prefer the workspace file, then a staged reference solution."""
    if os.path.exists(name):
        return name
    for pattern in ("reference-*/" + name, "solutions/reference-*/" + name):
        for path in sorted(glob.glob(pattern)):
            return path
    return None


def read(path):
    if not path:
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def get(d, dotted, default=None):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def group_or_source(group, sources, key):
    """A source's own value wins; otherwise inherit the group default."""
    values = []
    for src in sources:
        v = get(src, key)
        values.append(v if v is not None else get(group, key))
    return values


def is_reverse_text_source(src):
    return (
        get(src, "env.taskset.id") == TASKSET
        and get(src, "env.agent.harness.id") == "null"
        and get(src, "env.agent.runtime.type") == "subprocess"
    )


def num_eq(value, target):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value - target) < 1e-12


def toml_checks(raw):
    checks = []
    data = None
    err = ""
    if tomllib is None:
        err = "python3 lacks tomllib (needs 3.11+)"
    elif raw:
        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            err = str(exc)
    else:
        err = "rl.toml not found"

    checks.append(("rl.toml parses as TOML", data is not None, err or "ok"))
    if data is None:
        return checks, False

    no_inline = re.search(r"=\s*\{", re.sub(r"#.*", "", raw)) is None
    checks.append(("no inline tables", no_inline, "found `= {`" if not no_inline else "ok"))

    checks.append(("max_steps = 40", num_eq(data.get("max_steps"), 40), repr(data.get("max_steps"))))
    checks.append(("seq_len = 2048", num_eq(data.get("seq_len"), 2048), repr(data.get("seq_len"))))
    model = get(data, "model.name") or get(data, "trainer.model.name")
    checks.append(("policy model name", model == MODEL, repr(model)))

    orch = data.get("orchestrator", {})
    train = orch.get("train", {}) if isinstance(orch, dict) else {}
    evals = orch.get("eval", {}) if isinstance(orch, dict) else {}
    train_sources = train.get("source", []) if isinstance(train, dict) else []
    eval_sources = evals.get("source", []) if isinstance(evals, dict) else []
    train_sources = train_sources if isinstance(train_sources, list) else []
    eval_sources = eval_sources if isinstance(eval_sources, list) else []

    checks.append(("orchestrator.batch_size = 128", num_eq(get(orch, "batch_size"), 128),
                   repr(get(orch, "batch_size"))))

    rt_train = [s for s in train_sources if is_reverse_text_source(s)]
    checks.append(("train source: reverse-text taskset, null harness, subprocess runtime",
                   bool(rt_train), "{} train source(s)".format(len(train_sources))))
    checks.append(("train source named reverse-text",
                   any(s.get("name") == "reverse-text" for s in rt_train), "ok" if rt_train else "missing"))

    gs = group_or_source(train, rt_train, "group_size") if rt_train else [get(train, "group_size")]
    checks.append(("group_size = 16 (not the default 1)", bool(gs) and all(num_eq(v, 16) for v in gs), repr(gs)))

    mct = group_or_source(train, rt_train, "sampling.max_completion_tokens") if rt_train else []
    checks.append(("train max_completion_tokens = 128",
                   bool(mct) and all(num_eq(v, 128) for v in mct), repr(mct)))

    algo = get(train, "algo.type")
    src_algos = [get(s, "algo.type") for s in rt_train]
    grpo_ok = algo in (None, "grpo") and all(a in (None, "grpo") for a in src_algos)
    checks.append(("algorithm is GRPO (default or explicit)", grpo_ok, repr(algo)))

    rt_eval = [s for s in eval_sources if is_reverse_text_source(s)]
    named = any(s.get("name") == "reverse-text-eval" for s in rt_eval)
    checks.append(("eval source reverse-text-eval on reverse-text", bool(rt_eval) and named,
                   "{} eval source(s)".format(len(eval_sources))))
    interval = group_or_source(evals, rt_eval, "interval") if rt_eval else [get(evals, "interval")]
    checks.append(("online eval every 10 steps", bool(interval) and all(num_eq(v, 10) for v in interval),
                   repr(interval)))
    limit = group_or_source(evals, rt_eval, "select.limit") if rt_eval else []
    checks.append(("eval limited to 128 tasks", bool(limit) and all(num_eq(v, 128) for v in limit),
                   repr(limit)))

    lr = get(data, "trainer.optim.lr")
    checks.append(("trainer.optim.lr = 3e-6", num_eq(lr, 3e-6), repr(lr)))

    loss = get(data, "trainer.loss.type")
    checks.append(("trainer.loss.type = icepop", loss == "icepop", repr(loss)))

    ckpt = data.get("ckpt")
    ckpt_ok = isinstance(ckpt, dict) and num_eq(ckpt.get("interval"), 10) and num_eq(ckpt.get("keep_last"), 3)
    checks.append(("[ckpt] interval = 10, keep_last = 3", ckpt_ok, repr(ckpt)))

    project = get(data, "monitors.wandb.project")
    checks.append(("monitors.wandb.project = reverse-text", project == "reverse-text", repr(project)))

    renderer = get(data, "orchestrator.renderer.name")
    checks.append(("orchestrator.renderer.name = prime-qwen3", renderer == "prime-qwen3", repr(renderer)))

    checks.append(("[inference] present (rl launches the server)", isinstance(data.get("inference"), dict),
                   "ok" if isinstance(data.get("inference"), dict) else "missing"))
    return checks, True


def sh_checks(raw):
    text = raw.replace("\\\n", " ")
    lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    lines = [re.sub(r"(--[a-z0-9.-]+)=", r"\1 ", ln) for ln in lines if ln]
    rl_lines = [ln for ln in lines if re.search(r"\buv run rl\b", ln) and re.search(r"@\s*\S*rl\.toml", ln)]

    def has_flag(ln, flag, value=None):
        if value is None:
            return re.search(r"(?:^|\s){}(?=\s|$)".format(re.escape(flag)), ln) is not None
        return re.search(r"(?:^|\s){}\s+{}(?=\s|$)".format(re.escape(flag), re.escape(value)), ln) is not None

    dry = any(has_flag(ln, "--dry-run") for ln in rl_lines)
    launch = any(has_flag(ln, "--run.name", RUN_NAME) and not has_flag(ln, "--dry-run")
                 and not has_flag(ln, "--resume") for ln in rl_lines)
    resume = any(has_flag(ln, "--run.name", RUN_NAME) and has_flag(ln, "--resume")
                 and has_flag(ln, "--max-steps", "80") for ln in rl_lines)
    export = any(re.search(r"\buv run\b.*tools/convert_dcp_to_bf16\.py\s+\S*{}/checkpoints/step_80\b".format(
        re.escape(RUN_NAME)), ln) for ln in lines)
    banned = [ln for ln in lines if re.search(r"\bvllm serve\b|^torchrun\b|^python3?\b|\s-m\s+prime_rl", ln)]

    return [
        ("run.sh: dry run via uv run rl @ rl.toml --dry-run", dry, "ok" if dry else "missing"),
        ("run.sh: launch with --run.name rt-grpo", launch, "ok" if launch else "missing"),
        ("run.sh: resume rt-grpo with --resume --max-steps 80", resume, "ok" if resume else "missing"),
        ("run.sh: export step_80 with tools/convert_dcp_to_bf16.py", export, "ok" if export else "missing"),
        ("run.sh: no vllm serve / bare torchrun / raw python", bool(raw) and not banned,
         "; ".join(banned) if banned else ("ok" if raw else "run.sh not found")),
    ]


def main():
    toml_raw = read(find(TOML_NAME))
    sh_raw = read(find(SH_NAME))

    checks, _ = toml_checks(toml_raw)
    checks += sh_checks(sh_raw)

    passed = sum(1 for _, ok, _ in checks if ok)
    score = passed / len(checks)
    print(json.dumps({
        "score": round(score, 3),
        "details": "{}/{} checks passed".format(passed, len(checks)),
        "checks": [{"name": n, "passed": bool(ok), "message": m} for n, ok, m in checks],
    }))
    sys.exit(0 if score >= PASS_BAR else 1)


if __name__ == "__main__":
    main()
