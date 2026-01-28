#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id Hanabi-v0-train --num-players 2 --episodes 2 --seed 0 --agent openai:qwen/qwen3-8b --agent openai:google/gemini-2.5-flash --system-prompt 'You are an expert Hanabi teammate. Output exactly ONE valid action and nothing else. Valid formats: [Discard] X | [Play] X | [Reveal] player N card X color C | [Reveal] player N card X rank R. If you [Reveal], it must be truthful for that specific card index (color in {white,yellow,green,blue,red} or rank in {1,2,3,4,5}).'   --timeout 600 --temperature 0.2 --top-p 1.0   --episode-json-dir experiments/hanabi_qwen3_8b_vs_gemini25_flash/runs/6330ac8b165b/episodes --out experiments/hanabi_qwen3_8b_vs_gemini25_flash/runs/6330ac8b165b/rollouts.jsonl --resume
python tools/summarize_rollouts.py experiments/hanabi_qwen3_8b_vs_gemini25_flash/runs/6330ac8b165b/rollouts.jsonl --json > experiments/hanabi_qwen3_8b_vs_gemini25_flash/runs/6330ac8b165b/summary.json
