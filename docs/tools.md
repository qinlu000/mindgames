# Tools Overview

This branch documents only the code that still belongs to the three-game pure VERL mainline.

## Environment
- `tools/envs/create_verl_env.sh`: sync the current worktree `.venv` with `uv sync --extra train`.
- `tools/envs/README.verl-env.md`: uv environment notes and override examples.

## Training
- `mindgames/verl_training.py`: pure VERL interaction + reward helpers for the supported games.
- `tools/train/train_mindgames_verl.py`: shared CLI that materializes JSONL tasks and launches `verl.trainer.main_ppo`.
- `tools/train/train_mindgames_verl.sh`: shell launcher for the pure VERL CLI.

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
