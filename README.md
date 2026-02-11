# mindgames

## Install environment
```bash
cd mindgames
uv sync
# OpenAI-compatible rollouts (OpenRouter/OpenAI/etc.).
uv sync --extra agents
# GRPO Hanabi (vLLM server + ms-swift).
uv sync --extra serve
```
If you prefer, you can replace the last line with:
```bash
uv add "ms-swift[all]"
```

## GRPO Hanabi (gym env, 2 players)
This uses ms-swift + a vLLM rollout server. The reward comes from the Hanabi env, so keep `REWARD_FUNCS` empty.

Ensure `data/hanabi.grpo.jsonl` exists. If you want a custom max episode length, add
`"max_steps": <int>` to `env_config` (default is 300 from `mindgames/envs/Hanabi/env.py`).
To control training length, set `NUM_TRAIN_EPOCHS` or `MAX_STEPS` in the train command.
If both are set, `MAX_STEPS` wins. Defaults to `MAX_STEPS=500`. With the default Hanabi dataset (1 row), each epoch is ~1 optimizer step.

4+4 GPU split example (8x H800, group size 16, recommended):
```bash
# Terminal 1: rollout server (script uses built-in 0-3 GPU defaults)
bash tools/rollout/rollout_hanabi_gym.sh

# Terminal 2: GRPO training (script uses built-in 4-7 GPU defaults)
bash tools/train/train_grpo_hanabi_server_wandb.sh
```

Advanced (manual control, optional): use `tools/train/train_grpo_msswift.sh` with explicit env vars.

Notes for the wrapper:
- It uses `VLLM_MODE=server` (external rollout server), not colocated vLLM.
- It logs training metrics to W&B and uploads `OUTPUT_DIR` as a W&B `model` artifact.
- It pushes model outputs to Hugging Face Hub at the end (`hub_strategy=end`).
- For new repos, HF usually auto-creates `HF_REPO_ID` on first successful push if token has write permission.
- To change account/key/repo, edit defaults in `tools/train/train_grpo_hanabi_server_wandb.sh`.

To change rollout-side GPU/TP settings, edit defaults in `tools/rollout/rollout_hanabi_gym.sh`.

More single-node multi-GPU notes are in `docs/hanabi_grpo.md`.

## Parallel Hanabi rollouts (OpenAI-compatible, e.g. OpenRouter)
```bash
# Set your OpenAI-compatible key (OpenRouter uses OPENAI_API_KEY + OPENAI_BASE_URL).
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Run 100 episodes across multiple workers.
EPISODES=100 WORKERS=50 \
bash tools/rollout/run_hanabi_qwen3_235b_thinking_parallel.sh
```
