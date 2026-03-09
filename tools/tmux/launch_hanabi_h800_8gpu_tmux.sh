#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ROLLOUT_SESSION="${ROLLOUT_SESSION:-hanabi_rollout_h800}"
TRAIN_SESSION="${TRAIN_SESSION:-hanabi_train_h800}"

MODEL="${MODEL:-/workspace/models/Qwen3-8B}"
DATASET="${DATASET:-data/hanabi.grpo.jsonl}"
RUN_NAME="${RUN_NAME:-hanabi-grpo-h800-8gpu-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/$RUN_NAME}"

HOST="${HOST:-127.0.0.1}"
PORTS="${PORTS:-8100,8101,8102,8103}"
ROLLOUT_GPU_LIST="${ROLLOUT_GPU_LIST:-0,1,2,3}"
TRAIN_GPU_LIST="${TRAIN_GPU_LIST:-4,5,6,7}"

CONTEXT_MANAGER="${CONTEXT_MANAGER:-hanabi_recent_turns}"
HANABI_CTX_MAX_TURNS="${HANABI_CTX_MAX_TURNS:-1}"
HANABI_CTX_KEEP_SYSTEM="${HANABI_CTX_KEEP_SYSTEM:-true}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_USE_ASYNC_ENGINE="${VLLM_USE_ASYNC_ENGINE:-true}"

MAX_LENGTH="${MAX_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-13000}"
NUM_GENERATIONS="${NUM_GENERATIONS:-10}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-40}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-20}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
REPORT_TO="${REPORT_TO:-wandb}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-1800}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
USE_DEEPSPEED="${USE_DEEPSPEED:-false}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/tools/train/deepspeed_zero3_bf16.json}"

HEALTH_RETRIES="${HEALTH_RETRIES:-180}"
HEALTH_SLEEP_SEC="${HEALTH_SLEEP_SEC:-2}"
FORCE_RESTART="${FORCE_RESTART:-true}"

EXTRA_SWIFT_ARGS="${EXTRA_SWIFT_ARGS:---vllm_server_pass_dataset true}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

parse_csv() {
  local raw="$1"
  local -n out_ref="$2"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  out_ref=($raw)
}

join_by_comma() {
  local -n arr_ref="$1"
  local joined=""
  for item in "${arr_ref[@]}"; do
    if [ -z "$joined" ]; then
      joined="$item"
    else
      joined="${joined},${item}"
    fi
  done
  echo "$joined"
}

if [ ! -d "$MODEL" ]; then
  MODEL="Qwen/Qwen3-8B"
fi

if is_true "$USE_DEEPSPEED"; then
  if [[ " $EXTRA_SWIFT_ARGS " != *" --deepspeed "* ]]; then
    EXTRA_SWIFT_ARGS="$EXTRA_SWIFT_ARGS --deepspeed $DEEPSPEED_CONFIG"
  fi
fi

cd "$ROOT_DIR"

parse_csv "$PORTS" PORT_ARR
parse_csv "$ROLLOUT_GPU_LIST" ROLLOUT_GPU_ARR
parse_csv "$TRAIN_GPU_LIST" TRAIN_GPU_ARR

if [ "${#PORT_ARR[@]}" -eq 0 ]; then
  echo "PORTS is empty" >&2
  exit 1
fi
if [ "${#PORT_ARR[@]}" -ne "${#ROLLOUT_GPU_ARR[@]}" ]; then
  echo "PORTS count (${#PORT_ARR[@]}) must match ROLLOUT_GPU_LIST count (${#ROLLOUT_GPU_ARR[@]})." >&2
  exit 1
fi
if [ "${#TRAIN_GPU_ARR[@]}" -eq 0 ]; then
  echo "TRAIN_GPU_LIST is empty" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${#TRAIN_GPU_ARR[@]}}"

if [ -z "${VLLM_SERVER_HOST:-}" ]; then
  HOST_ARR=()
  for _ in "${PORT_ARR[@]}"; do
    HOST_ARR+=("$HOST")
  done
  VLLM_SERVER_HOST="$(join_by_comma HOST_ARR)"
fi

if [ -z "${VLLM_SERVER_GROUP_PORT:-}" ]; then
  GROUP_PORT_ARR=()
  for idx in "${!PORT_ARR[@]}"; do
    GROUP_PORT_ARR+=("$((63051 + idx))")
  done
  VLLM_SERVER_GROUP_PORT="$(join_by_comma GROUP_PORT_ARR)"
fi

echo "[h800-8gpu] rollout_gpus=$ROLLOUT_GPU_LIST train_gpus=$TRAIN_GPU_LIST ports=$PORTS run=$RUN_NAME"

FORCE_RESTART="$FORCE_RESTART" \
SESSION="$ROLLOUT_SESSION" \
MODEL="$MODEL" \
HOST="$HOST" \
PORTS="$PORTS" \
GPU_LIST="$ROLLOUT_GPU_LIST" \
CONTEXT_MANAGER="$CONTEXT_MANAGER" \
HANABI_CTX_MAX_TURNS="$HANABI_CTX_MAX_TURNS" \
HANABI_CTX_KEEP_SYSTEM="$HANABI_CTX_KEEP_SYSTEM" \
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
VLLM_USE_ASYNC_ENGINE="$VLLM_USE_ASYNC_ENGINE" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
bash tools/tmux/start_hanabi_rollout_tmux.sh

for port in "${PORT_ARR[@]}"; do
  ok=false
  for ((i = 1; i <= HEALTH_RETRIES; i++)); do
    if curl -fsS "http://${HOST}:${port}/health/" >/dev/null 2>&1; then
      ok=true
      break
    fi
    sleep "$HEALTH_SLEEP_SEC"
  done
  if [ "$ok" != "true" ]; then
    echo "rollout server health check failed: ${HOST}:${port}" >&2
    exit 1
  fi
done

FORCE_RESTART="$FORCE_RESTART" \
SESSION="$TRAIN_SESSION" \
MODEL="$MODEL" \
DATASET="$DATASET" \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
VLLM_SERVER_HOST="$VLLM_SERVER_HOST" \
VLLM_SERVER_PORT="$PORTS" \
VLLM_SERVER_GROUP_PORT="$VLLM_SERVER_GROUP_PORT" \
VLLM_SERVER_TIMEOUT="$VLLM_SERVER_TIMEOUT" \
NUM_GENERATIONS="$NUM_GENERATIONS" \
GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
MAX_LENGTH="$MAX_LENGTH" \
MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH" \
MAX_STEPS="$MAX_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
LOG_COMPLETIONS="$LOG_COMPLETIONS" \
REPORT_TO="$REPORT_TO" \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
EXTRA_SWIFT_ARGS="$EXTRA_SWIFT_ARGS" \
bash tools/tmux/start_hanabi_train_long_tmux.sh

echo
echo "H800 8-GPU GRPO started."
echo "Attach rollout: tmux attach -t $ROLLOUT_SESSION"
echo "Attach train:   tmux attach -t $TRAIN_SESSION"
echo "Status:         PORTS=$PORTS ROLLOUT_SESSION=$ROLLOUT_SESSION TRAIN_SESSION=$TRAIN_SESSION bash tools/tmux/status_hanabi_tmux.sh"
