#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id IteratedTwoThirdsAverage-v0 --env-kwargs '{"num_rounds":5}' --num-players 2 --episodes 50 --seed 0 --agent openai:qwen3-8b --agent openai:gemini-2.5-flash --system-prompt 'You are playing a numeric guessing game. Output ONLY your guess as a single bracketed number like [42.0]. No extra words, no explanation.'   --timeout 120 --episode-json-dir experiments/ita2p_qwen3_8b_vs_gemini25_flash/runs/a6a2c32ccac1/episodes --out experiments/ita2p_qwen3_8b_vs_gemini25_flash/runs/a6a2c32ccac1/rollouts.jsonl
python tools/summarize_rollouts.py experiments/ita2p_qwen3_8b_vs_gemini25_flash/runs/a6a2c32ccac1/rollouts.jsonl --json > experiments/ita2p_qwen3_8b_vs_gemini25_flash/runs/a6a2c32ccac1/summary.json
