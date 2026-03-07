#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper.
# Canonical Hanabi entrypoint: tools/train/train_grpo_hanabi_server_wandb.sh

exec bash tools/train/train_grpo_hanabi_server_wandb.sh "$@"
