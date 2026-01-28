#!/usr/bin/env bash
set -euo pipefail

python tools/run_rollouts.py --env-id LiarsDice-v0-train --num-players 2 --episodes 20 --seed 0 --agent qwen:Qwen/Qwen3-VL-4B-Instruct --agent qwen:Qwen/Qwen3-VL-4B-Instruct --system-prompt 'You are an expert Liar'"'"'s Dice player. Output exactly ONE valid action in the required bracket format (either [Bid: Q, F] or [Call]) and nothing else.' --openai-base-url http://127.0.0.1:8010/v1 --openai-api-key dummy --timeout 120 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.0 --presence-penalty 1.5       --out experiments/liarsdice_eval/runs/7e1d0f92361b/rollouts.jsonl
python tools/summarize_rollouts.py experiments/liarsdice_eval/runs/7e1d0f92361b/rollouts.jsonl --json > experiments/liarsdice_eval/runs/7e1d0f92361b/summary.json
