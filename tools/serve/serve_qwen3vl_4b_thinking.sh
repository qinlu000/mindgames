#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-4B-Thinking via vLLM (OpenAI-compatible server)
#
# Defaults:
#   - GPU: 2
#   - Port: 8010
#
# Override examples:
#   CUDA_VISIBLE_DEVICES=1 PORT=8011 bash tools/serve/serve_qwen3vl_4b_thinking.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Thinking}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
API_KEY="${API_KEY:-dummy}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
uv run vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-seqs 16
