#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: Hanabi GRPO server-mode launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env RLHF_TYPE=grpo bash "$SCRIPT_DIR/train_hanabi_rlhf_simple.sh" "$@"
