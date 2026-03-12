#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv-agent-lightning-verl}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
SYNC_PROJECT="${SYNC_PROJECT:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
VLLM_VERSION="${VLLM_VERSION:-0.10.2}"
VERL_VERSION="${VERL_VERSION:-0.5.0}"
AGENTLIGHTNING_SPEC="${AGENTLIGHTNING_SPEC:-agentlightning[verl]==0.2.2}"

if [[ "$SYNC_PROJECT" == "auto" ]]; then
  if [[ -x "$ENV_DIR/bin/python" ]]; then
    SYNC_PROJECT="0"
  else
    SYNC_PROJECT="1"
  fi
fi

if [[ "$SYNC_PROJECT" == "1" ]]; then
  echo "[agent-lightning-verl-env] creating isolated environment at: $ENV_DIR"
  UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --extra agents
else
  echo "[agent-lightning-verl-env] reusing existing environment at: $ENV_DIR"
fi

echo "[agent-lightning-verl-env] installing official weight-training stack"
"$ENV_DIR/bin/python" -m ensurepip --upgrade
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" --upgrade pip
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" wheel ninja psutil
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX_URL"
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" flash-attn --no-build-isolation
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" "vllm==${VLLM_VERSION}"
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" "verl==${VERL_VERSION}"
"$ENV_DIR/bin/python" -m pip install --cache-dir "$PIP_CACHE_DIR" "$AGENTLIGHTNING_SPEC"

# Current upstream resolution may leave an unreferenced pyvers package behind.
if "$ENV_DIR/bin/python" -m pip show pyvers >/dev/null 2>&1; then
  "$ENV_DIR/bin/python" -m pip uninstall -y pyvers >/dev/null
fi
"$ENV_DIR/bin/python" -m pip check

echo "[agent-lightning-verl-env] verifying imports"
"$ENV_DIR/bin/python" - <<'PY'
import agentlightning
import torch
import verl
import vllm

print("agentlightning", getattr(agentlightning, "__version__", "unknown"))
print("torch", getattr(torch, "__version__", "unknown"))
print("verl", getattr(verl, "__version__", "unknown"))
print("vllm", getattr(vllm, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
PY

echo "[agent-lightning-verl-env] done"
echo "[agent-lightning-verl-env] activate with: source \"$ENV_DIR/bin/activate\""
