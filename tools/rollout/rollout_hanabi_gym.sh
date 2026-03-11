#!/usr/bin/env bash
set -euo pipefail

# ms-swift rollout server for Hanabi gym (GRPO).
#
# Defaults (override via env vars):
#   MODEL=/workspace/models/Qwen3-8B (if exists), otherwise Qwen/Qwen3-8B
#   HOST=127.0.0.1
#   PORT=8000
#   GYM_ENV=hanabi_env
#   CONTEXT_MANAGER=hanabi_recent_turns
#   HANABI_CTX_MAX_TURNS=1
#   HANABI_CTX_KEEP_SYSTEM=true
#   CUDA_VISIBLE_DEVICES=<auto: first half of visible GPUs, e.g. 0-4 on 10 GPUs>
#   VLLM_TENSOR_PARALLEL_SIZE=1
#   VLLM_DATA_PARALLEL_SIZE=<auto: visible_gpus / tp, e.g. 5>
#   VLLM_MAX_MODEL_LEN=16384
#   VLLM_MAX_NUM_SEQS=16
#   VLLM_ENABLE_LORA=true
#   VLLM_MAX_LORA_RANK=8
#   VLLM_USE_ASYNC_ENGINE=true
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1
#   DRY_RUN=false

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GYM_ENV="${GYM_ENV:-hanabi_env}"
CONTEXT_MANAGER="${CONTEXT_MANAGER:-hanabi_recent_turns}"
MULTI_TURN_SCHEDULER="${MULTI_TURN_SCHEDULER:-hanabi_gym_scheduler}"
HANABI_CTX_MAX_TURNS="${HANABI_CTX_MAX_TURNS:-1}"
HANABI_CTX_KEEP_SYSTEM="${HANABI_CTX_KEEP_SYSTEM:-true}"

VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"
VLLM_USE_ASYNC_ENGINE="${VLLM_USE_ASYNC_ENGINE:-true}"
DRY_RUN="${DRY_RUN:-false}"

_count_csv_items() {
  local raw="$1"
  if [ -z "$raw" ]; then
    echo 0
    return
  fi
  local item
  local count=0
  IFS=',' read -r -a _items <<< "$raw"
  for item in "${_items[@]}"; do
    if [ -n "${item//[[:space:]]/}" ]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

_gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
}

_build_range_csv() {
  local start="$1"
  local end="$2"
  local values=()
  local i
  for ((i = start; i <= end; i++)); do
    values+=("$i")
  done
  local out
  out="$(IFS=,; echo "${values[*]}")"
  echo "$out"
}

if [ -z "${MODEL:-}" ]; then
  if [ -d "/workspace/models/Qwen3-8B" ]; then
    MODEL="/workspace/models/Qwen3-8B"
  else
    MODEL="Qwen/Qwen3-8B"
  fi
else
  MODEL="${MODEL}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  total_gpus="$(_gpu_count)"
  if [ "$total_gpus" -lt 1 ]; then
    echo "No GPUs detected. Set CUDA_VISIBLE_DEVICES explicitly." >&2
    exit 1
  fi
  # Default split: rollout uses the first half, train uses the second half.
  rollout_gpus=$((total_gpus / 2))
  if [ "$rollout_gpus" -lt 1 ]; then
    rollout_gpus=1
  fi
  CUDA_VISIBLE_DEVICES="$(_build_range_csv 0 $((rollout_gpus - 1)))"
fi

visible_gpu_count="$(_count_csv_items "$CUDA_VISIBLE_DEVICES")"
if [ "$visible_gpu_count" -lt 1 ]; then
  echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
if [ "$VLLM_TENSOR_PARALLEL_SIZE" -lt 1 ]; then
  echo "VLLM_TENSOR_PARALLEL_SIZE must be >= 1, got $VLLM_TENSOR_PARALLEL_SIZE" >&2
  exit 1
fi
if [ "$visible_gpu_count" -lt "$VLLM_TENSOR_PARALLEL_SIZE" ]; then
  echo "Not enough GPUs for TP=$VLLM_TENSOR_PARALLEL_SIZE (visible=$visible_gpu_count)." >&2
  exit 1
fi

if [ -z "$VLLM_DATA_PARALLEL_SIZE" ]; then
  if [ $((visible_gpu_count % VLLM_TENSOR_PARALLEL_SIZE)) -ne 0 ]; then
    echo "Visible GPUs ($visible_gpu_count) must be divisible by TP ($VLLM_TENSOR_PARALLEL_SIZE) when DP is auto." >&2
    exit 1
  fi
  VLLM_DATA_PARALLEL_SIZE="$((visible_gpu_count / VLLM_TENSOR_PARALLEL_SIZE))"
fi
if [ "$VLLM_DATA_PARALLEL_SIZE" -lt 1 ]; then
  echo "VLLM_DATA_PARALLEL_SIZE must be >= 1, got $VLLM_DATA_PARALLEL_SIZE" >&2
  exit 1
fi

required_gpus="$((VLLM_TENSOR_PARALLEL_SIZE * VLLM_DATA_PARALLEL_SIZE))"
if [ "$required_gpus" -gt "$visible_gpu_count" ]; then
  echo "TP*DP=$required_gpus exceeds visible GPUs=$visible_gpu_count." >&2
  exit 1
fi
if [ "$required_gpus" -lt "$visible_gpu_count" ]; then
  echo "WARN: TP*DP=$required_gpus leaves $((visible_gpu_count - required_gpus)) GPU(s) unused." >&2
fi

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

echo "[hanabi-rollout] model=$MODEL host=$HOST port=$PORT cuda=$CUDA_VISIBLE_DEVICES tp=$VLLM_TENSOR_PARALLEL_SIZE dp=$VLLM_DATA_PARALLEL_SIZE max_model_len=$VLLM_MAX_MODEL_LEN max_num_seqs=$VLLM_MAX_NUM_SEQS ctx=$CONTEXT_MANAGER ctx_max_turns=$HANABI_CTX_MAX_TURNS"

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
HANABI_CTX_MAX_TURNS="$HANABI_CTX_MAX_TURNS" \
HANABI_CTX_KEEP_SYSTEM="$HANABI_CTX_KEEP_SYSTEM" \
"${SWIFT_CMD[@]}" rollout \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --use_gym_env true \
  --gym_env "$GYM_ENV" \
  --context_manager "$CONTEXT_MANAGER" \
  --multi_turn_scheduler "$MULTI_TURN_SCHEDULER" \
  --external_plugins tools/rollout/hanabi_gym_plugin.py \
  --vllm_use_async_engine "$VLLM_USE_ASYNC_ENGINE" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_data_parallel_size "$VLLM_DATA_PARALLEL_SIZE" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_enable_lora "$VLLM_ENABLE_LORA" \
  --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK"
