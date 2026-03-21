# Tools Overview

This branch documents only the code that still belongs to the three-game Agent Lightning mainline.

## Environment
- `tools/envs/create_agent_lightning_verl_env.sh`: sync the current worktree `.venv` with `uv sync --extra agents --extra train`.
- `tools/envs/README.agent-lightning-verl-env.md`: uv environment notes and override examples.

## Training
- `tools/train/agent_lightning_games.py`: shared rollout logic for MiniHanabi, Colonel Blotto, and Negotiation.
- `tools/train/train_agent_lightning_games_verl.py`: shared `dev` / `train` CLI for Agent Lightning + VERL.
- `tools/train/train_agent_lightning_games_verl.sh`: launcher that runs the shared training CLI through `uv run`.

## Rollouts
- `tools/run_rollouts.py`: top-level wrapper for the generic rollout runner.
- `tools/rollout/run_rollouts.py`: generic offline rollout runner for the three supported game families.
- `tools/rollout/rollout_utils.py`: shared rollout serialization helpers.
- `tools/rollout/summarize_rollouts.py`: summarize rollouts JSONL into metrics.
- `tools/rollout/split_rollouts_jsonl.py`: split rollouts JSONL into per-episode files.
- `tools/summarize_rollouts.py`: top-level wrapper for rollout summarization.

## Serving
- `tools/serve/serve_qwen3_8b.sh`: launch a local vLLM server for Qwen3-8B.

## Data
- `tools/data/jsonl_to_json.py`: convert JSONL to a JSON array file.
- `tools/data/rollouts_to_sft_jsonl.py`: convert rollouts JSONL to SFT JSONL.
- `tools/data/view_jsonl.py`: pretty-print JSONL for terminal viewing.
