#!/usr/bin/env bash
set -euo pipefail

python tools/probe_fact_leakage.py --facts mindgames/envs/TruthAndDeception/facts.json --agent qwen:Qwen/Qwen3-VL-4B-Thinking     --seed 0 --num-items 200 --prompt-style tag --system-prompt 'You are doing a strict multiple-choice knowledge check. Output ONLY the final choice in the required format.' --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.0 --presence-penalty 0.0 --gen-seed 1234    --max-tokens 4096 --max-new-tokens 128 --output-minimal  --out-jsonl experiments/qwen3vl_fact_probe_all/runs/a5335ce00b58/probe.jsonl --out-summary experiments/qwen3vl_fact_probe_all/runs/a5335ce00b58/summary.json
