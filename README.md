# mindgames

This folder mirrors the **project layout style** of `spiral-rl/spiral`, but vendors a **minimal subset** of TextArena:

- Environments: `Hanabi`, `TruthAndDeception`
- Minimal core/state/wrappers needed to run them
- Minimal agent implementations (HF local, OpenAI/OpenRouter/Ollama, etc.)
- Simple offline rollout + eval tooling

## Quick start (offline rollouts)
```bash
python mindgames/tools/run_rollouts.py --help
```

## Read JSONL logs (human-friendly)
`tools/run_rollouts.py` writes JSONL (one record per step). For a readable terminal view:
```bash
cd mindgames
python tools/view_jsonl.py data/rollouts.jsonl | less -R
```

To convert JSONL into a single JSON array file:
```bash
cd mindgames
python tools/jsonl_to_json.py data/rollouts.jsonl --out data/rollouts.json
```

## API keys without `export` (dotenv)
If `mindgames/.env` exists, importing `mindgames` will automatically load it (without overriding already-set env vars).

Example: `mindgames/.env`:
```bash
OPENAI_API_KEY="..."
OPENAI_BASE_URL="https://api.uniapi.io/v1"
```

Disable auto-loading by setting `MINDGAMES_LOAD_DOTENV=0`.

## Python env (uv)
This subproject is intended to be used with a local `.venv` managed by `uv`:
```bash
cd mindgames
uv sync                 # creates ./mindgames/.venv and installs dev tools (default)
uv sync --extra agents  # optional: install API + HF agents dependencies
```

## Local vLLM
Run a local OpenAI-compatible server with vLLM, then use `openai:<model>` agents by pointing `OPENAI_BASE_URL` to it:
```bash
cd mindgames
# Large CUDA wheels may need a longer timeout (uv will suggest `UV_HTTP_TIMEOUT` on failure).
export UV_HTTP_TIMEOUT=600
# Retries for flaky connections (uv default is 3).
export UV_HTTP_RETRIES=10
# Optional: keep vLLM tracked by the project metadata (recommended for reproducibility).
# Note: for CUDA builds, set `UV_TORCH_BACKEND` so `torch`/`vllm` resolve to a compatible wheel.
export UV_TORCH_BACKEND=cu121
uv sync --extra serve
# `--torch-backend=auto` selects a CUDA build based on your environment.
# If it picks a too-new CUDA version for your driver, pin it explicitly (check `nvidia-smi`).
# Example: if `nvidia-smi` shows "CUDA Version: 12.2", `cu121` is a good default.
uv --native-tls pip install vllm --torch-backend=cu121

# 1) start server (in one terminal)
uv run vllm serve Qwen/Qwen3-VL-4B-Thinking --host 127.0.0.1 --port 8000 --trust-remote-code
# If you see "Address already in use", pick another port (e.g. 8001).
# To run on a specific GPU, set CUDA_VISIBLE_DEVICES (e.g. GPU 1):
# CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen3-VL-4B-Thinking --host 127.0.0.1 --port 8000 --trust-remote-code

# 2) run rollouts / probes (in another terminal)
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1  # match the port above
export OPENAI_API_KEY=dummy
python tools/probe_fact_leakage.py --agent openai:Qwen/Qwen3-VL-4B-Thinking --out-jsonl data/fact_probe.jsonl
```

## Experiment registry (Phase 1)
Use `mindgames/experiments/` as the source of truth for experiment configs (YAML), and `mindgames/tools/expctl.py` to:
- validate required fields
- compute a stable `run_id`
- render a reproducible `cmd.sh`
- optionally execute it locally via `expctl run` (captures `run.log` + `env_fingerprint.json`)

```bash
cd mindgames
python tools/expctl.py --help
python tools/expctl.py init --template rollout_eval --name hanabi_baseline
python tools/expctl.py prepare experiments/hanabi_baseline/experiment.yaml
python tools/expctl.py run experiments/hanabi_baseline/experiment.yaml
```

## Fact leakage probe (TruthAndDeception)
TruthAndDeception can be solved from **world knowledge** if the fact bank contains common trivia.
Use `mindgames/tools/probe_fact_leakage.py` to measure a *knowledge-only baseline* (no gameplay conversation):
```bash
cd mindgames
uv sync --extra agents
python tools/probe_fact_leakage.py \
  --facts mindgames/envs/TruthAndDeception/facts.json \
  --agent openrouter:moonshotai/kimi-k2:free \
  --out-jsonl data/fact_probe.jsonl

# Resume a partially-written JSONL (skip already-done items):
# python tools/probe_fact_leakage.py ... --out-jsonl data/fact_probe.jsonl --resume
```

## Licensing
- This folder includes code copied from the `textarena` repo (MIT). See `LICENSE_TEXTARENA`.
