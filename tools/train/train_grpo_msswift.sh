#!/usr/bin/env bash
set -euo pipefail

# ms-swift GRPO (Qwen3-8B + Hi-ToM by default).
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   DATASET=hf::Hi-ToM/Hi-ToM_Dataset
#   TRAIN_TYPE=lora   # ms-swift 3.x (use TUNER_TYPE for 4.x)
#   TUNER_TYPE=lora   # ms-swift 4.x (ignored by 3.x)
#   OUTPUT_DIR=output/qwen3-8b-hitom-grpo
#   REWARD_FUNCS=hitom_accuracy
#   REWARD_MODEL=
#   EXTERNAL_PLUGINS=tools/swift_plugins/hitom_dataset.py,tools/swift_plugins/hitom_reward.py
#   NUM_GENERATIONS=16
#   GENERATION_BATCH_SIZE=16
#   MAX_LENGTH=4096
#   MAX_PROMPT_LENGTH=        # alias for MAX_LENGTH
#   MAX_COMPLETION_LENGTH=8192
#   NUM_TRAIN_EPOCHS=
#   MAX_STEPS=
#   REPORT_TO=tensorboard   # set to wandb to enable W&B
#   RUN_NAME=
#   EXTERNAL_PLUGINS=
#   USE_VLLM=true
#   VLLM_MODE=colocate
#   VLLM_SERVER_HOST=
#   VLLM_SERVER_PORT=
#   CUDA_VISIBLE_DEVICES=0
#   NPROC_PER_NODE=1
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1
#
# Multi-GPU example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
#     bash tools/train/train_grpo_msswift.sh
#
# Hanabi gym example (requires rollout server):
#   DATASET=data/hanabi.grpo.jsonl VLLM_MODE=server \
#   VLLM_SERVER_HOST=127.0.0.1 VLLM_SERVER_PORT=8000 \
#   REWARD_FUNCS= EXTERNAL_PLUGINS= \
#     bash tools/train/train_grpo_msswift.sh

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-hf::Hi-ToM/Hi-ToM_Dataset}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
TUNER_TYPE="${TUNER_TYPE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hitom-grpo}"

REWARD_FUNCS="${REWARD_FUNCS-hitom_accuracy}"
REWARD_MODEL="${REWARD_MODEL:-}"
EXTERNAL_PLUGINS="${EXTERNAL_PLUGINS-tools/swift_plugins/hitom_dataset.py,tools/swift_plugins/hitom_reward.py}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$NUM_GENERATIONS}"
MAX_LENGTH="${MAX_LENGTH:-${MAX_PROMPT_LENGTH:-4096}}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-8192}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-}"
REPORT_TO="${REPORT_TO:-tensorboard}"
RUN_NAME="${RUN_NAME:-}"

USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

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

# reward funcs can be space- or comma-separated
REWARD_FUNCS="${REWARD_FUNCS//,/ }"
IFS=' ' read -r -a REWARD_FUNCS_ARR <<< "$REWARD_FUNCS"
EXTERNAL_PLUGINS="${EXTERNAL_PLUGINS//,/ }"
IFS=' ' read -r -a EXTERNAL_PLUGINS_ARR <<< "$EXTERNAL_PLUGINS"

EXTRA_ARGS=()
if [ "${#EXTERNAL_PLUGINS_ARR[@]}" -gt 0 ] && [ -n "${EXTERNAL_PLUGINS_ARR[0]}" ]; then
  EXTRA_ARGS+=(--external_plugins "${EXTERNAL_PLUGINS_ARR[@]}")
fi
if [ "${#REWARD_FUNCS_ARR[@]}" -gt 0 ] && [ -n "${REWARD_FUNCS_ARR[0]}" ]; then
  EXTRA_ARGS+=(--reward_funcs "${REWARD_FUNCS_ARR[@]}")
fi
if [ -n "$REWARD_MODEL" ]; then
  EXTRA_ARGS+=(--reward_model "$REWARD_MODEL")
fi
if [ -n "$REPORT_TO" ]; then
  EXTRA_ARGS+=(--report_to "$REPORT_TO")
fi
if [ -n "$RUN_NAME" ]; then
  EXTRA_ARGS+=(--run_name "$RUN_NAME")
fi
if [ -n "$MAX_LENGTH" ]; then
  EXTRA_ARGS+=(--max_length "$MAX_LENGTH")
fi
if [ -n "$MAX_COMPLETION_LENGTH" ]; then
  EXTRA_ARGS+=(--max_completion_length "$MAX_COMPLETION_LENGTH")
fi
if [ -n "$NUM_TRAIN_EPOCHS" ]; then
  EXTRA_ARGS+=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
fi
if [ -n "$MAX_STEPS" ]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi

if [ "$VLLM_MODE" = "server" ]; then
  if [ -n "$VLLM_SERVER_HOST" ]; then
    EXTRA_ARGS+=(--vllm_server_host "$VLLM_SERVER_HOST")
  fi
  if [ -n "$VLLM_SERVER_PORT" ]; then
    EXTRA_ARGS+=(--vllm_server_port "$VLLM_SERVER_PORT")
  fi
fi

TRAIN_ARGS=()
if [ -n "$TUNER_TYPE" ]; then
  TRAIN_ARGS+=(--tuner_type "$TUNER_TYPE")
else
  TRAIN_ARGS+=(--train_type "$TRAIN_TYPE")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
"${SWIFT_CMD[@]}" rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  "${TRAIN_ARGS[@]}" \
  --use_vllm "$USE_VLLM" \
  --vllm_mode "$VLLM_MODE" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  "${EXTRA_ARGS[@]}"
