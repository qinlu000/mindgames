# mindgames

This folder mirrors the **project layout style** of `spiral-rl/spiral`, but vendors a **minimal subset** of TextArena:

- Environments: `Hanabi`, `TruthAndDeception`
- Minimal core/state/wrappers needed to run them
- Minimal agent implementations (HF local, OpenAI/OpenRouter/Ollama, etc.)
- Simple offline rollout + eval tooling

## Quick start (offline rollouts)
```bash
python mindgames/tools/rollout/run_rollouts.py --help
```

## Parallel Hanabi rollouts (OpenAI-compatible, e.g. OpenRouter)
```bash
# Set your OpenAI-compatible key (OpenRouter uses OPENAI_API_KEY + OPENAI_BASE_URL).
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Run 100 episodes across multiple workers.
EPISODES=100 WORKERS=8 \
bash tools/rollout/run_hanabi_qwen3_235b_thinking_parallel.sh
```

## Read JSONL logs (human-friendly)
`tools/rollout/run_rollouts.py` writes JSONL (one record per step). For a readable terminal view:
```bash
cd mindgames
python tools/data/view_jsonl.py data/rollouts.jsonl | less -R
```

To convert JSONL into a single JSON array file:
```bash
cd mindgames
python tools/data/jsonl_to_json.py data/rollouts.jsonl --out data/rollouts.json
```

## Fine-tune from rollouts (SFT)
1) Convert rollout JSONL → chat-style SFT JSONL:
```bash
cd mindgames
python tools/data/rollouts_to_sft_jsonl.py --in data/rollouts.jsonl --out data/hanabi.sft.jsonl --env-id Hanabi-v0-train --min-score 10
```

2) Train (LoRA) with TRL:
```bash
pip install -U trl transformers accelerate peft datasets
cd mindgames
python tools/train/train_sft_trl.py --model Qwen/Qwen2.5-7B-Instruct --data data/hanabi.sft.jsonl --output-dir runs/sft_out
```

3) Train (LoRA/QLoRA) with ms-swift (CLI):
```bash
cd mindgames
uv add "ms-swift[all]"
bash tools/train/train_sft_msswift.sh

# Override defaults:
# MODEL=Qwen/Qwen3-8B-Instruct DATASET=Hi-ToM/Hi-ToM_Dataset \
# TRAIN_TYPE=qlora OUTPUT_DIR=output/qwen3-8b-hitom \
# CUDA_VISIBLE_DEVICES=1 bash tools/train/train_sft_msswift.sh
```
Note: the script sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` by default for RTX 4000-series stability. Override to `0` on systems with supported P2P/IB.

## GRPO Hanabi (gym env, 2 players)
This uses ms-swift + a vLLM rollout server. The reward comes from the Hanabi env, so keep `REWARD_FUNCS` empty.

Prereqs (one-time):
```bash
cd mindgames
uv sync --extra serve
# or: uv add "ms-swift[all]"
```

Ensure `data/hanabi.grpo.jsonl` exists. If you want a custom max episode length, add
`"max_steps": <int>` to `env_config` (default is 300 from `mindgames/envs/Hanabi/env.py`).
To control training length, set `NUM_TRAIN_EPOCHS` or `MAX_STEPS` in the train command.
If both are set, `MAX_STEPS` wins. Defaults to `MAX_STEPS=500`. With the default Hanabi dataset (1 row), each epoch is ~1 optimizer step.

4+4 GPU split example (8x H800, group size 16):
```bash
# Terminal 1: rollout server (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TENSOR_PARALLEL_SIZE=4 \
VLLM_MAX_NUM_SEQS=16 \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
bash tools/rollout/rollout_hanabi_gym.sh

# Terminal 2: GRPO training (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
DATASET=data/hanabi.grpo.jsonl VLLM_MODE=server \
VLLM_SERVER_HOST=127.0.0.1 VLLM_SERVER_PORT=8000 \
NUM_GENERATIONS=16 GENERATION_BATCH_SIZE=16 \
REWARD_FUNCS= EXTERNAL_PLUGINS= \
bash tools/train/train_grpo_msswift.sh
```
Optional W&B wrapper: `bash tools/train/train_grpo_hanabi_server_wandb.sh`
More single-node multi-GPU notes are in `docs/hanabi_grpo.md`.

## GRPO (Hi-ToM default)
```bash
cd mindgames
uv add "ms-swift[all]"
bash tools/train/train_grpo_msswift.sh

# Reward config (required for Hi-ToM GRPO):
# REWARD_FUNCS=hitom_accuracy \
# EXTERNAL_PLUGINS=tools/swift_plugins/hitom_dataset.py,tools/swift_plugins/hitom_reward.py \
# bash tools/train/train_grpo_msswift.sh

# Multi-GPU example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
#   bash tools/train/train_grpo_msswift.sh
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
python tools/analysis/probe_fact_leakage.py --agent openai:Qwen/Qwen3-VL-4B-Thinking --out-jsonl data/fact_probe.jsonl
```

## Experiment registry (Phase 1)
Use `mindgames/experiments/` as the source of truth for experiment configs (YAML), and `mindgames/tools/exp/expctl.py` to:
- validate required fields
- compute a stable `run_id`
- render a reproducible `cmd.sh`
- optionally execute it locally via `expctl run` (captures `run.log` + `env_fingerprint.json`)

```bash
cd mindgames
python tools/exp/expctl.py --help
python tools/exp/expctl.py init --template rollout_eval --name hanabi_baseline
python tools/exp/expctl.py prepare experiments/hanabi_baseline/experiment.yaml
python tools/exp/expctl.py run experiments/hanabi_baseline/experiment.yaml
```

## Fact leakage probe (TruthAndDeception)
TruthAndDeception can be solved from **world knowledge** if the fact bank contains common trivia.
Use `mindgames/tools/analysis/probe_fact_leakage.py` to measure a *knowledge-only baseline* (no gameplay conversation):
```bash
cd mindgames
uv sync --extra agents
python tools/analysis/probe_fact_leakage.py \
  --facts mindgames/envs/TruthAndDeception/facts.json \
  --agent openrouter:moonshotai/kimi-k2:free \
  --out-jsonl data/fact_probe.jsonl

# Resume a partially-written JSONL (skip already-done items):
# python tools/analysis/probe_fact_leakage.py ... --out-jsonl data/fact_probe.jsonl --resume
```

## Project structure
- `mindgames/`: core package (envs, agents, wrappers).
- `tools/`: scripts grouped by area (rollout, train, serve, data, analysis, exp).
- `data/`: datasets and JSONL rollouts.
- `experiments/`: experiment registry (YAML + rendered commands).
- `output/`: training outputs and checkpoints.

## Tools overview
- `tools/rollout/rollout_hanabi_gym.sh`: vLLM rollout server for Hanabi gym GRPO.
- `tools/rollout/run_hanabi_qwen3_235b_thinking_parallel.sh`: parallel Hanabi rollouts with Qwen3-235B Thinking (OpenAI-compatible endpoint).
- `tools/rollout/hanabi_gym_plugin.py`: gym env plugin used by the rollout server.
- `tools/train/train_grpo_msswift.sh`: GRPO training entrypoint (ms-swift).
- `tools/rollout/run_rollouts.py`: offline rollout runner (non-GRPO).
- `tools/data/rollouts_to_sft_jsonl.py`: convert rollouts to SFT JSONL.
- `tools/train/train_sft_msswift.sh`: SFT training entrypoint (ms-swift).
- `tools/train/train_sft_trl.py`: SFT training (TRL, python).
- `tools/exp/expctl.py`: experiment registry CLI (render/prepare/run).
Full list: `docs/tools.md`.

## Licensing
- This folder includes code copied from the `textarena` repo (MIT). See `LICENSE_TEXTARENA`.
