# mindgames

## Install environment
```bash
cd mindgames
uv sync --frozen --extra train --extra serve --extra agents
```

The `train` extra now includes `deepspeed`, which is required for runs that pass `--deepspeed` (for example the ZeRO-3 config in `tools/train/deepspeed_zero3_bf16.json`).

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
Use the H800 wrapper (tmux + health checks + server-mode GRPO):

```bash
bash tools/tmux/launch_hanabi_h800_8gpu_tmux.sh
```

Default behavior of this wrapper:
- rollout: `GPU 0,1,2,3` (`PORTS=8100,8101,8102,8103`)
- train: `GPU 4,5,6,7` (`NPROC_PER_NODE=4`)
- GRPO group: `NUM_GENERATIONS=10`, `GENERATION_BATCH_SIZE=40`
- token limits: `MAX_LENGTH=16384`, `MAX_COMPLETION_LENGTH=13000`
- trainer: `vllm_mode=server` (default without deepspeed)

Enable ZeRO-3 only when needed:
```bash
USE_DEEPSPEED=true \
bash tools/tmux/launch_hanabi_h800_8gpu_tmux.sh
```

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
- It sets W&B defaults for Hanabi GRPO (`REPORT_TO=wandb`, `WANDB_*` env pass-through).
- If `/workspace/models/Qwen3-8B` exists, scripts use that local model path by default.
- To change account/project/mode, edit env vars before running `tools/train/train_grpo_hanabi_server_wandb.sh`.

To change rollout-side GPU/TP settings, edit defaults in `tools/rollout/rollout_hanabi_gym.sh` or `tools/rollout/rollout_hanabi_gym_simple.sh`.

More single-node multi-GPU notes are in `docs/hanabi_grpo.md`.

## Hanabi MARSHAL-style training
To use MARSHAL's core ideas (turn-level reward signal + agent-specific normalization) in this repo:
```bash
# start rollout server(s) first
bash tools/rollout/rollout_hanabi_gym.sh

# then launch MARSHAL-style training wrapper
bash tools/train/train_grpo_hanabi_marshal.sh
```
Details and all knobs are in `docs/hanabi_marshal.md`.

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
