#!/usr/bin/env bash
set -euo pipefail

# ms-swift rollout server for Hanabi gym (GRPO).
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   HOST=127.0.0.1
#   PORT=8000
#   GYM_ENV=hanabi_env
#   VLLM_TENSOR_PARALLEL_SIZE=1
#   VLLM_DATA_PARALLEL_SIZE=1
#   VLLM_MAX_NUM_SEQS=16
#   VLLM_ENABLE_LORA=true
#   VLLM_MAX_LORA_RANK=8
#   CUDA_VISIBLE_DEVICES=0
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GYM_ENV="${GYM_ENV:-hanabi_env}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"

if command -v uv >/dev/null 2>&1; then
  SWIFT_CMD=(uv run swift)
elif [ -x ".venv/bin/swift" ]; then
  SWIFT_CMD=(.venv/bin/swift)
elif command -v swift >/dev/null 2>&1; then
  SWIFT_CMD=(swift)
else
  echo "swift not found. Install ms-swift or run: uv add \"ms-swift[all]\"" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
"${SWIFT_CMD[@]}" rollout \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --use_gym_env true \
  --gym_env "$GYM_ENV" \
  --multi_turn_scheduler gym_scheduler \
  --external_plugins tools/rollout/hanabi_gym_plugin.py \
  --vllm_use_async_engine true \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_data_parallel_size "$VLLM_DATA_PARALLEL_SIZE" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_enable_lora "$VLLM_ENABLE_LORA" \
  --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK"
