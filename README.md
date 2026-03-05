# mindgames

## Install environment
```bash
cd mindgames
uv sync --frozen --extra train --extra serve --extra agents
```

## GRPO Hanabi (gym env, 2 players)
This uses ms-swift + a vLLM rollout server. The reward comes from the Hanabi env, so keep `REWARD_FUNCS` empty.

Dataset note:
- `data/hanabi.grpo.template.jsonl` is a template, not for training.
- `data/hanabi.grpo.jsonl` must contain non-empty `messages` rows (already prepared in this repo).
- Quick check:
```bash
wc -l data/hanabi.grpo.jsonl
head -n 1 data/hanabi.grpo.jsonl
```

### Quick start (recommended for handoff, 8xH800)
Use the simple scripts (no machine-specific workaround logic).

Terminal 1 (rollout on GPU 0-3):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
VLLM_DATA_PARALLEL_SIZE=2 \
VLLM_MAX_MODEL_LEN=18000 \
VLLM_MAX_NUM_SEQS=16 \
NCCL_P2P_DISABLE=0 \
NCCL_IB_DISABLE=0 \
bash tools/rollout/rollout_hanabi_gym_simple.sh
```

Terminal 2 (train on GPU 4-7):
```bash
until curl -sf http://127.0.0.1:8000/health/ >/dev/null; do sleep 2; done

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
DATASET=data/hanabi.grpo.jsonl \
VLLM_SERVER_HOST=127.0.0.1 \
VLLM_SERVER_PORT=8000 \
NUM_GENERATIONS=8 \
GENERATION_BATCH_SIZE=32 \
MAX_LENGTH=18000 \
MAX_COMPLETION_LENGTH=18000 \
MAX_STEPS=1000 \
NCCL_P2P_DISABLE=0 \
NCCL_IB_DISABLE=0 \
bash tools/train/train_grpo_hanabi_server_simple.sh
```

`GENERATION_BATCH_SIZE` must be divisible by `NPROC_PER_NODE`.

### W&B
Online logging:
```bash
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=<your_entity_or_team>
export WANDB_PROJECT=mindgames
export WANDB_MODE=online
export REPORT_TO=wandb
```

Offline run + later sync:
```bash
export WANDB_MODE=offline
# run training...
wandb sync wandb/offline-run-*
```

### Robust wrappers (optional)
If the target machine has networking/NCCL quirks, use:
- `tools/rollout/rollout_hanabi_gym.sh`
- `tools/train/train_grpo_hanabi_server_wandb.sh`

Notes for the wrapper:
- It uses `VLLM_MODE=server` (external rollout server), not colocated vLLM.
- It logs training metrics to W&B and uploads `OUTPUT_DIR` as a W&B `model` artifact.
- If `PUSH_TO_HUB=true`, it pushes model outputs to Hugging Face Hub at the end (`hub_strategy=end`).
- If `/workspace/models/Qwen3-8B` exists, scripts use that local model path by default.
- For new repos, HF usually auto-creates `HF_REPO_ID` on first successful push if token has write permission.
- To change account/key/repo, edit defaults in `tools/train/train_grpo_hanabi_server_wandb.sh`.

To change rollout-side GPU/TP settings, edit defaults in `tools/rollout/rollout_hanabi_gym.sh` or `tools/rollout/rollout_hanabi_gym_simple.sh`.

More single-node multi-GPU notes are in `docs/hanabi_grpo.md`.

## Hanabi API batch self-play (OpenAI-compatible, e.g. OpenRouter)
```bash
export OPENAI_BASE_URL="https://openrouter.ai/api/v1" OPENAI_API_KEY="..." && MODELS_FILE="data/hanabi_api_models_15.txt" MODEL_GEN_FILE="data/hanabi_api_model_gen_overrides.json" PARALLEL_JOBS=15 bash tools/rollout/run_hanabi_api_batch_eval.sh
```
`MIN_MODELS` is optional now; if omitted, it defaults to the actual non-comment model count in `MODELS_FILE`.
`EPISODES` defaults to `10` per model.
`TIMEOUT` defaults to `300` seconds per request.
By default the batch script does **not** send `temperature/top_p`; provider-side defaults apply unless you set them.
Set `STREAM=1` to enable streaming for OpenAI-compatible (`openai`/`qwen`) calls, e.g.:
```bash
OPENAI_BASE_URL="https://openrouter.ai/api/v1" OPENAI_API_KEY="..." \
MODELS_FILE="data/hanabi_api_models_15.txt" \
MODEL_GEN_FILE="data/hanabi_api_model_gen_overrides.json" \
PARALLEL_JOBS=15 STREAM=1 \
bash tools/rollout/run_hanabi_api_batch_eval.sh
```

## Hanabi: human + LLM mixed play
```bash
# Example: you play seat 0, seat 1 is a scripted baseline (replace with openai:/qwen: model spec if needed).
uv run python tools/run_rollouts.py \
  --env-id Hanabi-v0-train \
  --num-players 2 \
  --episodes 1 \
  --human-players 0 \
  --llm-agent scripted:hanabi_discard0 \
  --out data/hanabi_human_llm.jsonl
```

## Hanabi: human + LLM web GUI (format-safe actions)
```bash
# Starts a local web app at http://127.0.0.1:8765 .
# Human seat(s) use button/select controls; non-human seats use --llm-agent.
uv run python tools/hanabi_human_ai_gui.py \
  --env-id Hanabi-v0-train \
  --num-players 2 \
  --human-players 0 \
  --llm-agent scripted:hanabi_discard0 \
  --host 127.0.0.1 \
  --port 8765
```

Examples:
- Replace `--llm-agent scripted:hanabi_discard0` with `openai:gpt-4.1-mini` (or `qwen:<model>`) as needed.
- For OpenAI-compatible backends, pass `--openai-base-url` / `--openai-api-key` (or set `OPENAI_BASE_URL` / `OPENAI_API_KEY`).
