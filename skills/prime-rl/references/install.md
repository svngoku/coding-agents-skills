# Installing prime-rl

prime-rl is a uv monorepo with git submodules (verifiers, renderers, prime-envs, pydantic-config).
It needs Linux with at least one NVIDIA GPU and Python 3.12 (managed by uv). Upstream tests on
RTX 3090/4090/5090, A100, H100, H200 and B200.

## Quick Setup

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

The script clones the repo, initializes submodules recursively, installs uv, and runs
`uv sync --all-extras`.

## Manual Setup

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --all-extras
```

For an existing clone that already tracks every submodule, `git submodule update --init --recursive`
also works (it additionally pulls `deps/prime-kernels`).

## Validate the Install

```bash
uv run python -V                                  # Python 3.12.x
uv run python -c "import flash_attn"              # attention kernels present
uv run sft @ configs/debug/fake/sft.toml          # trainer works (1 GPU)
uv run inference --vllm.model Qwen/Qwen3-0.6B     # inference works (1 GPU)
uv run rl @ examples/basic/reverse-text/rl.toml   # full RL stack (2 GPUs)
```

Optional logins:

```bash
uv run wandb login          # or export WANDB_API_KEY=...
uv run hf auth login        # or export HF_TOKEN=... for gated models and datasets
```

## uv sync Variants

| Command | Installs |
|---------|----------|
| `uv sync` | Slim core |
| `uv sync --group dev` | Core + pytest, ruff, pre-commit (run `uv run pre-commit install` once) |
| `uv sync --all-extras` | Core + every extra (`gpu`, `dashboard`, `dynamo`, `flash-attn`, `flash-attn-cute`, `disagg`, `quack`, `sonic-moe`, `flash-mla`, `kernels`) |
| `uv sync --all-extras --all-packages` | Above + every environment package |
| `uv sync --all-extras --package prime-rl --package gsm8k` | Core + extras + one environment |

## Environment Packages

Environments are uv **workspace members** and are opt-in.

- Packages under `deps/prime-envs/environments/*/*` are auto-discovered. Adding one there needs no
  `pyproject.toml` edit.
- verifiers' example envs are listed explicitly in `[tool.uv.workspace].members`. Add a path there to
  use another one.
- A plain `uv sync` or `uv sync --all-extras` does not install members, and **removes** them if they
  were installed. Re-run with `--all-packages`, a `--package` list, or `--inexact`.
- Keep `--all-extras` on training machines so a targeted `--package` sync does not prune GPU deps.
- If two envs pin conflicting transitive versions, add one of them to `[tool.uv.workspace].exclude`.

Check that an env imports before launching a run that uses it:

```bash
uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym'))"
```

The reverse-text env ships inside the verifiers submodule, so the smoke run needs no extra package.

## Optional Extras

| Extra | Command | Provides |
|-------|---------|----------|
| `kernels` | `uv sync --extra kernels` | Prebuilt prime-kernels CUDA wheels |
| `gpu` | `uv sync --extra gpu` | Mamba2 SSD kernels for NemotronH (no `nvcc` needed) |
| `disagg` | `uv sync --extra disagg` | Prebuilt DeepEP wheel for expert parallelism |
| `dashboard` | `uv sync --extra dashboard` | Dashboard without the GPU stack (head nodes) |

Verify DeepEP with `uv run python -c 'import deep_ep; print(deep_ep.__file__)'`.

Source builds are manual and need a CUDA toolkit matching torch:

```bash
git submodule update --init deps/prime-kernels
uv pip install --no-build-isolation -e deps/prime-kernels                         # kernels
bash deps/prime-kernels/scripts/install_ep_kernels.sh --wheel-dir /tmp/prime-kernels-wheels
uv pip install --reinstall --no-deps /tmp/prime-kernels-wheels/deep_ep-*.whl      # DeepEP
```

Set `TORCH_CUDA_ARCH_LIST` when the build host has no GPU or must target several architectures.
A source build overrides the wheel until the next `uv sync`.

### llm-d Router

Multi-node or disaggregated deployments can use the llm-d Endpoint Picker instead of `vllm-router`
(`[inference.router] type = "llm-d"`). Install its binaries once:

```bash
bash scripts/install_llmd.sh   # writes third_party/llmd/bin/{epp,envoy,pd-sidecar}
```

## Dependency Rules

- Add dependencies to `pyproject.toml`, then run `uv sync --all-extras` to lock them.
- Pin git dependencies with a 7-character commit hash in `rev`.
- Never edit `.venv/`; changes are overwritten by the next sync.
- The workspace uses `exclude-newer = "7 days"`. To bump a package past that window, add it to
  `[tool.uv.exclude-newer-package]` before refreshing `uv.lock`.
