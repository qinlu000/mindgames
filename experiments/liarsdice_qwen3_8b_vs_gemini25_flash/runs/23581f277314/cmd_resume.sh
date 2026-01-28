#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id LiarsDice-v0-train --num-players 2 --episodes 30 --seed 0 --agent openai:qwen3-8b --agent openai:gemini-2.5-flash --system-prompt 'You are an expert Liar'"'"'s Dice player. Output exactly ONE valid action in the required bracket format (either [Bid: Q, F] or [Call]) and nothing else.'   --timeout 120 --out experiments/liarsdice_qwen3_8b_vs_gemini25_flash/runs/23581f277314/rollouts.jsonl --resume
python tools/summarize_rollouts.py experiments/liarsdice_qwen3_8b_vs_gemini25_flash/runs/23581f277314/rollouts.jsonl --json > experiments/liarsdice_qwen3_8b_vs_gemini25_flash/runs/23581f277314/summary.json
