#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

printf '[mindgames-verl-env] syncing uv project env at: %s\n' "$ENV_DIR"
uv sync --extra train

printf '[mindgames-verl-env] verifying imports\n'
uv run --extra train python - <<'INNER'
import flash_attn
import torch
import verl
import vllm

print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
print("torch", getattr(torch, "__version__", "unknown"))
print("verl", getattr(verl, "__version__", "unknown"))
print("vllm", getattr(vllm, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
INNER

printf '[mindgames-verl-env] done\n'
printf '[mindgames-verl-env] activate with: source "%s/bin/activate"\n' "$ENV_DIR"
