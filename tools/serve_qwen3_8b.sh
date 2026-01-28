#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B (text) via vLLM (OpenAI-compatible server)
#
# This repo is set up to run vLLM via `uv run` (see `mindgames/README.md`).
#
# Defaults (override via env vars):
#   CUDA_VISIBLE_DEVICES=0
#   HOST=0.0.0.0
#   PORT=8020
#   API_KEY=dummy
#   MODEL=Qwen/Qwen3-8B-Instruct
#   DTYPE=bfloat16
#   TENSOR_PARALLEL_SIZE=1
#   MAX_MODEL_LEN=16384
#   GPU_MEMORY_UTILIZATION=0.90
#   MAX_NUM_SEQS=16
#
# Multi-GPU example (2× GPUs, more headroom / throughput):
#   CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=32 \
#     bash tools/serve_qwen3_8b.sh

MODEL="${MODEL:-Qwen/Qwen3-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8020}"
API_KEY="${API_KEY:-dummy}"

DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
  --dtype "$DTYPE" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"

