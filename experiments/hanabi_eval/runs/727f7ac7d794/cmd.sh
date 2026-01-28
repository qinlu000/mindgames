#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id Hanabi-v0-train --num-players 2 --episodes 20 --seed 0 --agent qwen:Qwen/Qwen3-VL-4B-Instruct --agent qwen:Qwen/Qwen3-VL-4B-Instruct --system-prompt 'You are an expert Hanabi teammate. Output exactly ONE valid action and nothing else. Valid formats: [Discard] X | [Play] X | [Reveal] player N card X color C | [Reveal] player N card X rank R' --openai-base-url http://127.0.0.1:8010/v1 --openai-api-key dummy --timeout 120 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.0 --presence-penalty 1.5       --out experiments/hanabi_eval/runs/727f7ac7d794/rollouts.jsonl
python tools/summarize_rollouts.py experiments/hanabi_eval/runs/727f7ac7d794/rollouts.jsonl --json > experiments/hanabi_eval/runs/727f7ac7d794/summary.json
