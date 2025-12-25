#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-4B-Instruct via vLLM (OpenAI-compatible server)
#
# Defaults (override via env vars):
#   CUDA_VISIBLE_DEVICES=1
#   HOST=0.0.0.0
#   PORT=8010
#   API_KEY=dummy
#   MAX_MODEL_LEN=16384   # aligns with HF "VL" out_seq_length recommendation
#   GPU_MEMORY_UTILIZATION=0.85
#   MAX_NUM_SEQS=16
#
# Example:
#   CUDA_VISIBLE_DEVICES=1 PORT=8011 bash tools/serve_qwen3vl_4b_instruct.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
API_KEY="${API_KEY:-dummy}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
uv run vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"

