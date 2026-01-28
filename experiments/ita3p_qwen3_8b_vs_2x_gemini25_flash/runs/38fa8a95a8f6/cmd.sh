#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id IteratedTwoThirdsAverage3P-v0 --env-kwargs '{"num_rounds":5}' --num-players 3 --episodes 30 --seed 0 --agent openai:qwen3-8b --agent openai:gemini-2.5-flash --agent openai:gemini-2.5-flash --system-prompt 'You are playing a numeric guessing game. Output ONLY your guess as a single bracketed number like [42.0]. No extra words, no explanation.'   --timeout 120 --episode-json-dir experiments/ita3p_qwen3_8b_vs_2x_gemini25_flash/runs/38fa8a95a8f6/episodes --out experiments/ita3p_qwen3_8b_vs_2x_gemini25_flash/runs/38fa8a95a8f6/rollouts.jsonl
python tools/summarize_rollouts.py experiments/ita3p_qwen3_8b_vs_2x_gemini25_flash/runs/38fa8a95a8f6/rollouts.jsonl --json > experiments/ita3p_qwen3_8b_vs_2x_gemini25_flash/runs/38fa8a95a8f6/summary.json
