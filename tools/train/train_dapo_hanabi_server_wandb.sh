#!/usr/bin/env bash
set -euo pipefail

# Hanabi DAPO wrapper with auto GPU split + W&B defaults.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LOSS_TYPE="${LOSS_TYPE:-dapo}"
BETA="${BETA:-0}"
RUN_NAME="${RUN_NAME:-dapo-hanabi}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hanabi-dapo}"
EXTRA_SWIFT_ARGS="${EXTRA_SWIFT_ARGS:-}"

DAPO_SWIFT_ARGS="--loss_type ${LOSS_TYPE}"
if [ -n "$EXTRA_SWIFT_ARGS" ]; then
  DAPO_SWIFT_ARGS="${DAPO_SWIFT_ARGS} ${EXTRA_SWIFT_ARGS}"
fi

exec env \
  BETA="$BETA" \
  RUN_NAME="$RUN_NAME" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  EXTRA_SWIFT_ARGS="$DAPO_SWIFT_ARGS" \
  bash "$SCRIPT_DIR/train_grpo_hanabi_server_wandb.sh"
