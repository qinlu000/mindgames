# mindgames

## Install environment
```bash
cd mindgames
uv sync --extra agents --extra agent-lightning
```

This branch has been simplified to a clean Agent Lightning stack. The older `ms-swift` / GRPO / DeepSpeed training path has been removed from this branch.

## Fresh isolated environment
To create a brand new virtualenv for this branch without reusing the previous `.venv-grpo` setup:
```bash
bash tools/envs/create_agent_lightning_env.sh
```

Default target env:
- `.venv-agent-lightning`

Override example:
```bash
ENV_DIR=/workspace/mindgames/.venv-agent-lightning-qwen \
bash tools/envs/create_agent_lightning_env.sh
```

## Agent Lightning Hanabi training
This entrypoint uses Agent Lightning APO to optimize the Hanabi system prompt from episode reward using the existing OpenAI-compatible agent interface.

Typical local vLLM launch:
```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
MODEL=/workspace/models/Qwen3-8B \
bash tools/train/train_agent_lightning_hanabi.sh
```

Useful knobs:
- `AGENT_KIND=qwen|openai`
- `TRAIN_EPISODES=...` / `VAL_EPISODES=...`
- `TRAIN_TASK_FILE=...jsonl` / `VAL_TASK_FILE=...jsonl`
- `PROMPT_TEMPLATE_FILE=...txt`
- `ENV_KWARGS='{"marshal_dense_reward": true}'`
- `MAX_TRIALS=...`

The task JSONL format is one object per line, for example:
```json
{"id":"train-000001","seed":1,"env_id":"Hanabi-v0-train","num_players":2}
```

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
