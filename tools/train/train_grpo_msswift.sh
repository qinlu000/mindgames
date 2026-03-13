#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: forwards to the canonical RLHF base launcher.
# Canonical entrypoint is tools/train/train_rlhf_base.sh.

if [ -n "${TRAIN_TYPE:-}" ] && [ -z "${TUNER_TYPE:-}" ]; then
  TUNER_TYPE="$TRAIN_TYPE"
  export TUNER_TYPE
fi
if [ -n "${SWIFT_EXTRA_ARGS:-}" ] && [ -z "${EXTRA_SWIFT_ARGS:-}" ]; then
  EXTRA_SWIFT_ARGS="$SWIFT_EXTRA_ARGS"
  export EXTRA_SWIFT_ARGS
fi
if [ -n "${GRPO_SWIFT_BIN:-}" ] && [ -z "${SWIFT_BIN:-}" ]; then
  SWIFT_BIN="$GRPO_SWIFT_BIN"
  export SWIFT_BIN
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_rlhf_base.sh" "$@"
