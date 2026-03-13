#!/usr/bin/env bash
set -euo pipefail

# ms-swift GRPO base launcher used by the Hanabi wrappers.
#
# Defaults (override via env vars):
#   RLHF_TYPE=grpo
#   SWIFT_BIN=                       # explicit swift binary, e.g. /workspace/mindgames/.venv-grpo/bin/swift
#   MODEL=/workspace/models/Qwen3-8B (if exists), otherwise Qwen/Qwen3-8B
#   ADAPTERS=                        # comma or space separated SFT LoRA adapters for the train model
#   REF_ADAPTERS=                    # comma or space separated reference adapters
#   REF_MODEL=
#   REF_MODEL_TYPE=
#   REF_MODEL_REVISION=
#   DATASET=data/hanabi.grpo.jsonl
#   OUTPUT_DIR=output/qwen3-8b-hanabi-grpo
#   TUNER_TYPE=lora
#   USE_VLLM=true
#   VLLM_MODE=server                 # server | colocate
#   VLLM_SERVER_HOST=127.0.0.1       # comma-separated supported
#   VLLM_SERVER_PORT=8000            # comma-separated supported
#   VLLM_SERVER_GROUP_PORT=          # optional comma-separated
#   VLLM_SERVER_TIMEOUT=
#   VLLM_GPU_MEMORY_UTILIZATION=
#   VLLM_TENSOR_PARALLEL_SIZE=
#   NUM_GENERATIONS=8
#   GENERATION_BATCH_SIZE=           # mutually exclusive with STEPS_PER_GENERATION
#   STEPS_PER_GENERATION=
#   MAX_LENGTH=4096
#   MAX_COMPLETION_LENGTH=64
#   ENABLE_THINKING=
#   LEARNING_RATE=
#   BETA=
#   ASYNC_GENERATE=false
#   COMPLETION_LENGTH_LIMIT_SCOPE=   # total | per_round
#   VLLM_SERVER_PASS_DATASET=
#   SOFT_MAX_LENGTH=
#   OVERLONG_FILTER=
#   NUM_TRAIN_EPOCHS=
#   MAX_STEPS=500
#   SAVE_STEPS=500
#   MAX_TURNS=
#   REPORT_TO=wandb
#   RUN_NAME=
#   REWARD_FUNCS=                    # comma or space separated
#   EXTERNAL_PLUGINS=                # comma or space separated
#   LOG_COMPLETIONS=true
#   EXTRA_SWIFT_ARGS=
#   CUDA_VISIBLE_DEVICES=0
#   NPROC_PER_NODE=1
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1
#   TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
#   DRY_RUN=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=tools/train/_common.sh
source "$SCRIPT_DIR/_common.sh"

RLHF_TYPE="${RLHF_TYPE:-grpo}"
SWIFT_BIN="${SWIFT_BIN:-}"
MODEL="${MODEL:-$(resolve_default_qwen3_8b_model)}"
ADAPTERS="${ADAPTERS:-}"
REF_ADAPTERS="${REF_ADAPTERS:-$ADAPTERS}"
REF_MODEL="${REF_MODEL:-}"
REF_MODEL_TYPE="${REF_MODEL_TYPE:-}"
REF_MODEL_REVISION="${REF_MODEL_REVISION:-}"
DATASET="${DATASET:-data/hanabi.grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hanabi-grpo}"
TUNER_TYPE="${TUNER_TYPE:-lora}"

USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-server}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_GROUP_PORT="${VLLM_SERVER_GROUP_PORT:-}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
NO_PROXY="${NO_PROXY:-}"
no_proxy="${no_proxy:-}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-}"

MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
BETA="${BETA:-}"
ASYNC_GENERATE="${ASYNC_GENERATE:-false}"
COMPLETION_LENGTH_LIMIT_SCOPE="${COMPLETION_LENGTH_LIMIT_SCOPE:-}"
VLLM_SERVER_PASS_DATASET="${VLLM_SERVER_PASS_DATASET:-}"
SOFT_MAX_LENGTH="${SOFT_MAX_LENGTH:-}"
OVERLONG_FILTER="${OVERLONG_FILTER:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_TURNS="${MAX_TURNS:-}"

REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-}"
REWARD_FUNCS="${REWARD_FUNCS:-}"
EXTERNAL_PLUGINS="${EXTERNAL_PLUGINS:-}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
EXTRA_SWIFT_ARGS="${EXTRA_SWIFT_ARGS:-}"
PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
DRY_RUN="${DRY_RUN:-false}"

resolve_swift_cmd "$SWIFT_BIN" SWIFT_CMD

if ! is_pos_int "$NPROC_PER_NODE"; then
  echo "NPROC_PER_NODE must be a positive integer, got '$NPROC_PER_NODE'" >&2
  exit 1
fi

parse_list "$REWARD_FUNCS" REWARD_FUNCS_ARR
parse_list "$EXTERNAL_PLUGINS" EXTERNAL_PLUGINS_ARR
parse_list "$ADAPTERS" ADAPTERS_ARR
parse_list "$REF_ADAPTERS" REF_ADAPTERS_ARR
parse_list "$VLLM_SERVER_HOST" VLLM_SERVER_HOST_ARR
parse_list "$VLLM_SERVER_PORT" VLLM_SERVER_PORT_ARR
parse_list "$VLLM_SERVER_GROUP_PORT" VLLM_SERVER_GROUP_PORT_ARR

case "$RLHF_TYPE" in
  grpo) ;;
  *)
    echo "train_rlhf_base.sh only supports RLHF_TYPE=grpo." >&2
    exit 1
    ;;
esac

if ! is_pos_int "$NUM_GENERATIONS"; then
  echo "NUM_GENERATIONS must be a positive integer, got '$NUM_GENERATIONS'" >&2
  exit 1
fi
if [ -n "$GENERATION_BATCH_SIZE" ] && [ -n "$STEPS_PER_GENERATION" ]; then
  echo "GENERATION_BATCH_SIZE and STEPS_PER_GENERATION are mutually exclusive." >&2
  exit 1
fi
if [ -n "$GENERATION_BATCH_SIZE" ] && ! is_pos_int "$GENERATION_BATCH_SIZE"; then
  echo "GENERATION_BATCH_SIZE must be a positive integer, got '$GENERATION_BATCH_SIZE'" >&2
  exit 1
fi
if [ -n "$STEPS_PER_GENERATION" ] && ! is_pos_int "$STEPS_PER_GENERATION"; then
  echo "STEPS_PER_GENERATION must be a positive integer, got '$STEPS_PER_GENERATION'" >&2
  exit 1
fi
if [ -z "$GENERATION_BATCH_SIZE" ] && [ -z "$STEPS_PER_GENERATION" ]; then
  GENERATION_BATCH_SIZE="$NUM_GENERATIONS"
fi
if [ -n "$GENERATION_BATCH_SIZE" ]; then
  if [ $((GENERATION_BATCH_SIZE % NUM_GENERATIONS)) -ne 0 ]; then
    echo "GENERATION_BATCH_SIZE ($GENERATION_BATCH_SIZE) must be divisible by NUM_GENERATIONS ($NUM_GENERATIONS)." >&2
    exit 1
  fi
  if [ $((GENERATION_BATCH_SIZE % NPROC_PER_NODE)) -ne 0 ]; then
    echo "WARN: GENERATION_BATCH_SIZE ($GENERATION_BATCH_SIZE) is not divisible by NPROC_PER_NODE ($NPROC_PER_NODE)." >&2
  fi
fi

if [ "$RLHF_TYPE" = "grpo" ] && is_true "$USE_VLLM" && [ "$VLLM_MODE" = "server" ]; then
  if [ "${#VLLM_SERVER_HOST_ARR[@]}" -ne "${#VLLM_SERVER_PORT_ARR[@]}" ]; then
    echo "VLLM_SERVER_HOST count (${#VLLM_SERVER_HOST_ARR[@]}) must match VLLM_SERVER_PORT count (${#VLLM_SERVER_PORT_ARR[@]})." >&2
    exit 1
  fi
  if [ "${#VLLM_SERVER_GROUP_PORT_ARR[@]}" -gt 0 ] && [ "${#VLLM_SERVER_GROUP_PORT_ARR[@]}" -ne "${#VLLM_SERVER_HOST_ARR[@]}" ]; then
    echo "VLLM_SERVER_GROUP_PORT count (${#VLLM_SERVER_GROUP_PORT_ARR[@]}) must match server count (${#VLLM_SERVER_HOST_ARR[@]})." >&2
    exit 1
  fi

  NO_PROXY_EXTRAS="127.0.0.1,localhost,::1"
  for host in "${VLLM_SERVER_HOST_ARR[@]}"; do
    NO_PROXY_EXTRAS="${NO_PROXY_EXTRAS},${host}"
  done
  if [ -n "$NO_PROXY" ]; then
    NO_PROXY="${NO_PROXY},${NO_PROXY_EXTRAS}"
  else
    NO_PROXY="${NO_PROXY_EXTRAS}"
  fi
  if [ -n "$no_proxy" ]; then
    no_proxy="${no_proxy},${NO_PROXY_EXTRAS}"
  else
    no_proxy="${NO_PROXY_EXTRAS}"
  fi
fi

CMD=(
  "${SWIFT_CMD[@]}" rlhf
  --rlhf_type "$RLHF_TYPE"
  --model "$MODEL"
  --tuner_type "$TUNER_TYPE"
  --use_vllm "$USE_VLLM"
  --dataset "$DATASET"
  --output_dir "$OUTPUT_DIR"
  --max_length "$MAX_LENGTH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --max_steps "$MAX_STEPS"
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
)

append_arg_if_set CMD --num_train_epochs "$NUM_TRAIN_EPOCHS"
append_list_arg_if_any CMD --adapters ADAPTERS_ARR
append_list_arg_if_any CMD --ref_adapters REF_ADAPTERS_ARR
append_arg_if_set CMD --ref_model "$REF_MODEL"
append_arg_if_set CMD --ref_model_type "$REF_MODEL_TYPE"
append_arg_if_set CMD --ref_model_revision "$REF_MODEL_REVISION"
append_arg_if_set CMD --report_to "$REPORT_TO"
append_arg_if_set CMD --run_name "$RUN_NAME"
append_arg_if_set CMD --enable_thinking "$ENABLE_THINKING"
append_arg_if_set CMD --learning_rate "$LEARNING_RATE"
append_arg_if_set CMD --beta "$BETA"
append_list_arg_if_any CMD --external_plugins EXTERNAL_PLUGINS_ARR

CMD+=(--num_generations "$NUM_GENERATIONS")
append_arg_if_set CMD --async_generate "$ASYNC_GENERATE"
append_arg_if_set CMD --completion_length_limit_scope "$COMPLETION_LENGTH_LIMIT_SCOPE"
append_arg_if_set CMD --vllm_server_pass_dataset "$VLLM_SERVER_PASS_DATASET"
append_arg_if_set CMD --soft_max_length "$SOFT_MAX_LENGTH"
append_arg_if_set CMD --overlong_filter "$OVERLONG_FILTER"
append_arg_if_set CMD --max_turns "$MAX_TURNS"
append_arg_if_set CMD --log_completions "$LOG_COMPLETIONS"
append_arg_if_set CMD --steps_per_generation "$STEPS_PER_GENERATION"
append_arg_if_set CMD --generation_batch_size "$GENERATION_BATCH_SIZE"
append_list_arg_if_any CMD --reward_funcs REWARD_FUNCS_ARR

if is_true "$USE_VLLM"; then
  CMD+=(--vllm_mode "$VLLM_MODE")
  if [ "$VLLM_MODE" = "server" ]; then
    append_list_arg_if_any CMD --vllm_server_host VLLM_SERVER_HOST_ARR
    append_list_arg_if_any CMD --vllm_server_port VLLM_SERVER_PORT_ARR
    append_list_arg_if_any CMD --vllm_server_group_port VLLM_SERVER_GROUP_PORT_ARR
    append_arg_if_set CMD --vllm_server_timeout "$VLLM_SERVER_TIMEOUT"
  elif [ "$VLLM_MODE" = "colocate" ]; then
    append_arg_if_set CMD --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    append_arg_if_set CMD --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  else
    echo "VLLM_MODE must be 'server' or 'colocate', got '$VLLM_MODE'" >&2
    exit 1
  fi
fi

if [ -n "$EXTRA_SWIFT_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_SWIFT_ARR=($EXTRA_SWIFT_ARGS)
  CMD+=("${EXTRA_SWIFT_ARR[@]}")
fi

if is_true "$DRY_RUN"; then
  printf '[rlhf-base] ' >&2
  printf '%q ' "${CMD[@]}" >&2
  printf '\n' >&2
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
NO_PROXY="$NO_PROXY" \
no_proxy="$no_proxy" \
PYTHONPATH="$PYTHONPATH" \
"${CMD[@]}"
