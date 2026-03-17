#!/usr/bin/env bash
set -euo pipefail

# Hanabi DAPO wrapper for ms-swift.
#
# DAPO in ms-swift 4.x uses the GRPO entrypoint with DAPO-specific loss knobs.
# This wrapper keeps the existing Hanabi gym rollout flow and injects the
# recommended DAPO defaults for ms-swift 4.0.0:
#   --loss_type dapo
#   --beta 0
#
# Optional overrides:
#   LOSS_TYPE=dapo
#   BETA=0
#   EXTRA_SWIFT_ARGS="..."

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
  bash "$SCRIPT_DIR/train_hanabi_rlhf_simple.sh"
