# Tools Overview

This branch documents only the code that still belongs to the three-game pure VERL mainline.

## Environment
- `tools/envs/create_verl_env.sh`: sync the current worktree `.venv` with `uv sync --extra train`.
- `tools/envs/README.verl-env.md`: uv environment notes and override examples.

## Training
- `mindgames/training/contracts.py`: shared `GameStep` / `EpisodeStepResult` training contracts.
- `mindgames/training/specs.py`: single source of truth for per-game defaults and game-specific training behavior.
- `mindgames/training/episode.py`: backend-agnostic episode runner over the three supported games.
- `mindgames/training/verl_adapter.py`: VERL-specific dataset, reward, and interaction adapter layer.
- `mindgames/training/verl_launch.py`: run-plan construction, dataset materialization, and VERL launch helpers.
- `tools/train/train_mindgames_verl.py`: thin CLI wrapper around the training-core launch helpers.
- `tools/train/train_mindgames_verl.sh`: shell launcher for the pure VERL CLI.

## Rollouts
- `mindgames/training/rollouts.py`: shared rollout trace helpers and mindgames episode execution.
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
