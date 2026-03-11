#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv-agent-lightning}"

echo "[agent-lightning-env] creating isolated environment at: $ENV_DIR"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --extra agents --extra agent-lightning "$@"
echo "[agent-lightning-env] done"
echo "[agent-lightning-env] activate with: source \"$ENV_DIR/bin/activate\""
