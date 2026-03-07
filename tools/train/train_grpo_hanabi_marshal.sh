#!/usr/bin/env bash
set -euo pipefail

# MARSHAL-style Hanabi training wrapper.
#
# What this script does:
# 1) Prepare marshal dataset.
# 2) Launch Hanabi wrapper with marshal-oriented GRPO knobs via EXTRA_SWIFT_ARGS.

BASE_DATASET="${BASE_DATASET:-data/hanabi.grpo.jsonl}"
DATASET="${DATASET:-data/hanabi.grpo.marshal.jsonl}"
REBUILD_DATASET="${REBUILD_DATASET:-false}"

MARSHAL_DENSE_REWARD="${MARSHAL_DENSE_REWARD:-true}"
MARSHAL_FUSE_PENALTY="${MARSHAL_FUSE_PENALTY:-0.0}"
MARSHAL_INVALID_PENALTY="${MARSHAL_INVALID_PENALTY:-0.0}"
MARSHAL_AGENT_NORM="${MARSHAL_AGENT_NORM:-true}"
MARSHAL_AGENT_NORM_METHOD="${MARSHAL_AGENT_NORM_METHOD:-mean_std}"
MARSHAL_AGENT_NORM_WARMUP="${MARSHAL_AGENT_NORM_WARMUP:-8}"
MARSHAL_AGENT_NORM_CLIP="${MARSHAL_AGENT_NORM_CLIP:-}"

if [ ! -f "$DATASET" ] || [ "$REBUILD_DATASET" = "true" ]; then
  if command -v uv >/dev/null 2>&1; then
    PREP_CMD=(uv run python tools/data/prepare_hanabi_marshal_dataset.py)
  else
    PREP_CMD=(python tools/data/prepare_hanabi_marshal_dataset.py)
  fi

  PREP_ARGS=(
    --input "$BASE_DATASET"
    --output "$DATASET"
    --marshal-dense-reward "$MARSHAL_DENSE_REWARD"
    --marshal-fuse-penalty "$MARSHAL_FUSE_PENALTY"
    --marshal-invalid-penalty "$MARSHAL_INVALID_PENALTY"
    --marshal-agent-norm "$MARSHAL_AGENT_NORM"
    --marshal-agent-norm-method "$MARSHAL_AGENT_NORM_METHOD"
    --marshal-agent-norm-warmup "$MARSHAL_AGENT_NORM_WARMUP"
  )
  if [ -n "$MARSHAL_AGENT_NORM_CLIP" ]; then
    PREP_ARGS+=(--marshal-agent-norm-clip "$MARSHAL_AGENT_NORM_CLIP")
  fi

  "${PREP_CMD[@]}" "${PREP_ARGS[@]}"
fi

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-reinforce_plus_plus}"
SCALE_REWARDS="${SCALE_REWARDS:-none}"
WHITEN_REWARDS="${WHITEN_REWARDS:-false}"
VLLM_SERVER_PASS_DATASET="${VLLM_SERVER_PASS_DATASET:-true}"

RUN_NAME="${RUN_NAME:-grpo-hanabi-marshal-style}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-}"
MAX_TURNS="${MAX_TURNS:-}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"

MARSHAL_SWIFT_ARGS="--advantage_estimator ${ADVANTAGE_ESTIMATOR} --scale_rewards ${SCALE_REWARDS} --whiten_rewards ${WHITEN_REWARDS} --vllm_server_pass_dataset ${VLLM_SERVER_PASS_DATASET}"
if [ -n "${EXTRA_SWIFT_ARGS:-}" ]; then
  MARSHAL_SWIFT_ARGS="${MARSHAL_SWIFT_ARGS} ${EXTRA_SWIFT_ARGS}"
fi

DATASET="$DATASET" \
RUN_NAME="$RUN_NAME" \
STEPS_PER_GENERATION="$STEPS_PER_GENERATION" \
GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
MAX_TURNS="$MAX_TURNS" \
LOG_COMPLETIONS="$LOG_COMPLETIONS" \
EXTRA_SWIFT_ARGS="$MARSHAL_SWIFT_ARGS" \
bash tools/train/train_grpo_hanabi_server_simple.sh
