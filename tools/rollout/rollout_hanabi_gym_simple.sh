#!/usr/bin/env bash
set -euo pipefail

# Simple Hanabi GRPO rollout server launcher (no machine-specific workarounds).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#   HOST=127.0.0.1 PORT=8000 \
#   CONTEXT_MANAGER=hanabi_recent_turns \
#   HANABI_CTX_MAX_TURNS=1 \
#   HANABI_CTX_KEEP_SYSTEM=true \
#   VLLM_TENSOR_PARALLEL_SIZE=1 \
#   VLLM_DATA_PARALLEL_SIZE=1 \
#   VLLM_MAX_MODEL_LEN=18000 \
#   VLLM_MAX_NUM_SEQS=16 \
#   bash tools/rollout/rollout_hanabi_gym_simple.sh

if [ -z "${MODEL:-}" ]; then
  if [ -d "/workspace/models/Qwen3-8B" ]; then
    MODEL="/workspace/models/Qwen3-8B"
  else
    MODEL="Qwen/Qwen3-8B"
  fi
else
  MODEL="${MODEL}"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GYM_ENV="${GYM_ENV:-hanabi_env}"
CONTEXT_MANAGER="${CONTEXT_MANAGER:-hanabi_recent_turns}"
HANABI_CTX_MAX_TURNS="${HANABI_CTX_MAX_TURNS:-1}"
HANABI_CTX_KEEP_SYSTEM="${HANABI_CTX_KEEP_SYSTEM:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-18000}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"
VLLM_USE_ASYNC_ENGINE="${VLLM_USE_ASYNC_ENGINE:-true}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

if command -v uv >/dev/null 2>&1; then
  SWIFT_CMD=(uv run swift)
elif [ -x ".venv/bin/swift" ]; then
  SWIFT_CMD=(.venv/bin/swift)
elif command -v swift >/dev/null 2>&1; then
  SWIFT_CMD=(swift)
else
  echo "swift not found. Please install ms-swift first." >&2
  exit 1
fi

echo "[hanabi-rollout-simple] model=$MODEL host=$HOST port=$PORT cuda=$CUDA_VISIBLE_DEVICES tp=$VLLM_TENSOR_PARALLEL_SIZE dp=$VLLM_DATA_PARALLEL_SIZE max_model_len=$VLLM_MAX_MODEL_LEN max_num_seqs=$VLLM_MAX_NUM_SEQS ctx=$CONTEXT_MANAGER ctx_max_turns=$HANABI_CTX_MAX_TURNS"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
HANABI_CTX_MAX_TURNS="$HANABI_CTX_MAX_TURNS" \
HANABI_CTX_KEEP_SYSTEM="$HANABI_CTX_KEEP_SYSTEM" \
"${SWIFT_CMD[@]}" rollout \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --use_gym_env true \
  --gym_env "$GYM_ENV" \
  --context_manager "$CONTEXT_MANAGER" \
  --multi_turn_scheduler gym_scheduler \
  --external_plugins tools/rollout/hanabi_gym_plugin.py \
  --vllm_use_async_engine "$VLLM_USE_ASYNC_ENGINE" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_data_parallel_size "$VLLM_DATA_PARALLEL_SIZE" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_enable_lora "$VLLM_ENABLE_LORA" \
  --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK"
