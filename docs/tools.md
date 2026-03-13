# Tools Overview

This doc summarizes the scripts under `tools/`. Files are grouped into subfolders.

## Rollout
- `tools/rollout/run_rollouts.py`: run offline rollouts for any env and write JSONL (supports mixed human+LLM seats via `--human-players` + `--llm-agent`).
- `tools/rollout/rollout_hanabi_gym.sh`: vLLM rollout server for Hanabi GRPO (gym env).
- `tools/rollout/run_rollout_server.sh`: generic vLLM rollout server wrapper.
- `tools/rollout/run_hanabi_qwen3_8b_vllm_500.sh`: run 500 Hanabi episodes via a local vLLM server.
- `tools/rollout/run_hanabi_qwen3_235b_thinking_parallel.sh`: parallel Hanabi rollouts with Qwen3-235B Thinking (OpenAI-compatible endpoint).
- `tools/rollout/run_hanabi_partner_pool_matrix.sh`: start one vLLM per model and run NxN Hanabi cross-play matrix for partner-pool research.
- `tools/rollout/watch_then_run_partner_pool_matrix.sh`: wait for a batch eval to finish, then auto-launch partner-pool NxN matrix.
- `tools/rollout/rollout_utils.py`: shared helpers for compact rollout/episode formats.
- `tools/rollout/summarize_rollouts.py`: summarize rollouts JSONL into metrics.
- `tools/rollout/split_rollouts_jsonl.py`: split rollouts JSONL into per-episode files.
- `tools/rollout/hanabi_gym_plugin.py`: gym env plugin for Hanabi GRPO rollout server.
- `tools/rollout/watch_hanabi_batch_30m.sh`: monitor batch progress every 30 minutes and auto-generate final report when done.

## Data prep
- `tools/data/rollouts_to_sft_jsonl.py`: convert rollouts JSONL to SFT JSONL.
- `tools/data/prepare_hitom_grpo_dataset.py`: prepare Hi-ToM GRPO dataset.
- `tools/data/jsonl_to_json.py`: convert JSONL to a JSON array file.
- `tools/data/view_jsonl.py`: pretty-print JSONL for terminal viewing.

## Training
- `tools/train/train_rlhf_base.sh`: canonical GRPO base entrypoint for ms-swift wrappers.
- `tools/train/train_grpo_base.sh`: compatibility alias to `train_rlhf_base.sh`.
- `tools/train/train_grpo_msswift.sh`: compatibility alias to `train_grpo_base.sh`.
- `tools/train/train_hanabi_rlhf_simple.sh`: canonical Hanabi GRPO wrapper.
- `tools/train/train_grpo_hanabi_server_simple.sh`: compatibility alias to Hanabi GRPO mode.
- `tools/train/train_grpo_hanabi_server_wandb.sh`: Hanabi GRPO wrapper with W&B defaults.
- `tools/train/train_grpo_hanabi_marshal.sh`: Hanabi MARSHAL-style wrapper.
- `tools/train/train_grpo_msswift_server_wandb.sh`: compatibility alias to Hanabi W&B wrapper.
- `tools/train/train_sft_msswift.sh`: SFT training entrypoint (ms-swift).

## Serving
- `tools/serve/serve_qwen3_8b.sh`: launch a local vLLM server for Qwen3-8B.
- `tools/serve/serve_qwen3vl_4b_instruct.sh`: launch a local vLLM server for Qwen3-VL-4B-Instruct.
- `tools/serve/serve_qwen3vl_4b_thinking.sh`: launch a local vLLM server for Qwen3-VL-4B-Thinking.

## Analysis
- `tools/analysis/probe_fact_leakage.py`: knowledge-only probe for TruthAndDeception.

## Experiment control
- `tools/exp/expctl.py`: experiment registry CLI (render/prepare/run).

## Plugins
- `tools/swift_plugins/`: ms-swift plugins (reward, dataset hooks).
