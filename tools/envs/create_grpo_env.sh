#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv-grpo}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CLEAR_ENV="${CLEAR_ENV:-false}"

GRPO_MS_SWIFT_VERSION="${GRPO_MS_SWIFT_VERSION:-4.0.0}"
GRPO_TRL_VERSION="${GRPO_TRL_VERSION:-0.24.0}"
GRPO_VLLM_VERSION="${GRPO_VLLM_VERSION:-0.10.2}"
GRPO_DEEPSPEED_VERSION="${GRPO_DEEPSPEED_VERSION:-0.18.7}"
GRPO_TRANSFORMERS_VERSION="${GRPO_TRANSFORMERS_VERSION:-4.57.6}"
GRPO_ACCELERATE_VERSION="${GRPO_ACCELERATE_VERSION:-1.12.0}"

echo "Creating isolated GRPO env at: $ENV_DIR"
if [ -f "$ENV_DIR/pyvenv.cfg" ]; then
  case "${CLEAR_ENV}" in
    1|true|TRUE|yes|YES|on|ON)
      uv venv "$ENV_DIR" --python "$PYTHON_BIN" --clear
      ;;
    *)
      echo "Reusing existing virtual environment at: $ENV_DIR"
      ;;
  esac
else
  uv venv "$ENV_DIR" --python "$PYTHON_BIN"
fi

PY="$ENV_DIR/bin/python"

echo "Installing GRPO-compatible package set"
uv pip install --python "$PY" --upgrade pip setuptools wheel
uv pip install --python "$PY" \
  "ms-swift==${GRPO_MS_SWIFT_VERSION}" \
  "trl==${GRPO_TRL_VERSION}" \
  "vllm==${GRPO_VLLM_VERSION}" \
  "deepspeed==${GRPO_DEEPSPEED_VERSION}" \
  "transformers==${GRPO_TRANSFORMERS_VERSION}" \
  "accelerate==${GRPO_ACCELERATE_VERSION}"

echo "Verifying installed versions"
"$PY" - <<'PY'
import importlib.metadata as md

for pkg in ["ms-swift", "trl", "vllm", "deepspeed", "transformers", "accelerate", "torch"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception as e:
        print(f"{pkg}=not-found ({e})")
PY

"$ENV_DIR/bin/swift" rlhf --help >/dev/null
"$ENV_DIR/bin/swift" rollout --help >/dev/null

cat <<EOF

GRPO env is ready.
Activate:
  source "$ENV_DIR/bin/activate"

Or run directly:
  "$ENV_DIR/bin/swift" rlhf --help
  "$ENV_DIR/bin/swift" rollout --help
EOF
