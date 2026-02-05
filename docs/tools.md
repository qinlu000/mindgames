# Tools Overview

This doc summarizes the scripts under `tools/`. Files are grouped into subfolders.

## Rollout
- `tools/rollout/run_rollouts.py`: run offline rollouts for any env and write JSONL.
- `tools/rollout/rollout_hanabi_gym.sh`: vLLM rollout server for Hanabi GRPO (gym env).
- `tools/rollout/run_rollout_server.sh`: generic vLLM rollout server wrapper.
- `tools/rollout/run_hanabi_qwen3_8b_vllm_500.sh`: run 500 Hanabi episodes via a local vLLM server.
- `tools/rollout/rollout_utils.py`: shared helpers for compact rollout/episode formats.
- `tools/rollout/summarize_rollouts.py`: summarize rollouts JSONL into metrics.
- `tools/rollout/split_rollouts_jsonl.py`: split rollouts JSONL into per-episode files.
- `tools/rollout/hanabi_gym_plugin.py`: gym env plugin for Hanabi GRPO rollout server.

## Data prep
- `tools/data/rollouts_to_sft_jsonl.py`: convert rollouts JSONL to SFT JSONL.
- `tools/data/prepare_hitom_grpo_dataset.py`: prepare Hi-ToM GRPO dataset.
- `tools/data/jsonl_to_json.py`: convert JSONL to a JSON array file.
- `tools/data/view_jsonl.py`: pretty-print JSONL for terminal viewing.

## Training
- `tools/train/train_grpo_msswift.sh`: GRPO training entrypoint (ms-swift).
- `tools/train/train_grpo_msswift_server_wandb.sh`: GRPO training with external vLLM server + W&B.
- `tools/train/train_sft_msswift.sh`: SFT training entrypoint (ms-swift).
- `tools/train/train_sft_trl.py`: SFT training (TRL, python).

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
