#!/usr/bin/env bash
set -euo pipefail

# Hanabi wrapper (wandb): auto-select train GPUs + call simple wrapper.
#
# Defaults (override via env vars):
#   MODEL=/workspace/models/Qwen3-8B (if exists), otherwise Qwen/Qwen3-8B
#   CUDA_VISIBLE_DEVICES=<auto second half>
#   NPROC_PER_NODE=<auto from train GPUs>
#   DATASET=data/hanabi.grpo.jsonl
#   OUTPUT_DIR=output/qwen3-8b-hanabi-grpo
#   NUM_GENERATIONS=<auto: 2 * NPROC_PER_NODE>
#   GENERATION_BATCH_SIZE=<auto: NUM_GENERATIONS * NPROC_PER_NODE>
#   STEPS_PER_GENERATION=
#   MAX_LENGTH=4096
#   MAX_COMPLETION_LENGTH=64
#   MAX_STEPS=500
#   VLLM_SERVER_HOST=127.0.0.1
#   VLLM_SERVER_PORT=8000
#   VLLM_SERVER_GROUP_PORT=
#   VLLM_SERVER_TIMEOUT=
#   REPORT_TO=wandb
#   RUN_NAME=grpo-hanabi
#   WANDB_PROJECT=mindgames
#   WANDB_ENTITY=
#   WANDB_MODE=online
#   WANDB_NAME=$RUN_NAME
#   DRY_RUN=false

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
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  total_gpus="$(_gpu_count)"
  if [ "$total_gpus" -lt 1 ]; then
    echo "No GPUs detected. Set CUDA_VISIBLE_DEVICES explicitly." >&2
    exit 1
  fi
  train_start=$((total_gpus / 2))
  train_end=$((total_gpus - 1))
  if [ "$train_start" -gt "$train_end" ]; then
    train_start=0
    train_end=0
  fi
  CUDA_VISIBLE_DEVICES="$(_build_range_csv "$train_start" "$train_end")"
fi

train_gpu_count="$(_count_csv_items "$CUDA_VISIBLE_DEVICES")"
if [ "$train_gpu_count" -lt 1 ]; then
  echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$train_gpu_count}"
DATASET="${DATASET:-data/hanabi.grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hanabi-grpo}"

NUM_GENERATIONS="${NUM_GENERATIONS:-$((NPROC_PER_NODE * 2))}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$((NUM_GENERATIONS * NPROC_PER_NODE))}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_TURNS="${MAX_TURNS:-}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"

VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_GROUP_PORT="${VLLM_SERVER_GROUP_PORT:-}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-}"

REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-grpo-hanabi}"
WANDB_PROJECT="${WANDB_PROJECT:-mindgames}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
EXTRA_SWIFT_ARGS="${EXTRA_SWIFT_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

if [ "$REPORT_TO" = "wandb" ] && [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARN: WANDB_API_KEY is empty while WANDB_MODE=online. Switching to offline." >&2
  WANDB_MODE="offline"
fi

echo "[hanabi-train] model=$MODEL server=${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT} cuda=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE num_generations=$NUM_GENERATIONS gen_batch=${GENERATION_BATCH_SIZE:-auto} steps_per_generation=${STEPS_PER_GENERATION:-none}"

MODEL="$MODEL" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" \
DATASET="$DATASET" OUTPUT_DIR="$OUTPUT_DIR" \
NUM_GENERATIONS="$NUM_GENERATIONS" GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
STEPS_PER_GENERATION="$STEPS_PER_GENERATION" \
MAX_LENGTH="$MAX_LENGTH" MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" MAX_STEPS="$MAX_STEPS" \
MAX_TURNS="$MAX_TURNS" LOG_COMPLETIONS="$LOG_COMPLETIONS" \
VLLM_SERVER_HOST="$VLLM_SERVER_HOST" VLLM_SERVER_PORT="$VLLM_SERVER_PORT" \
VLLM_SERVER_GROUP_PORT="$VLLM_SERVER_GROUP_PORT" VLLM_SERVER_TIMEOUT="$VLLM_SERVER_TIMEOUT" \
REPORT_TO="$REPORT_TO" RUN_NAME="$RUN_NAME" \
WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" WANDB_MODE="$WANDB_MODE" WANDB_NAME="$WANDB_NAME" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
EXTRA_SWIFT_ARGS="$EXTRA_SWIFT_ARGS" DRY_RUN="$DRY_RUN" \
bash tools/train/train_grpo_hanabi_server_simple.sh
