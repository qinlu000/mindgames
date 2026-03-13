#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: forwards to the canonical RLHF base launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_rlhf_base.sh" "$@"
