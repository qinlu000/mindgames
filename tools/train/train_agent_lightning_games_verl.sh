#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[agent-lightning-verl-env] missing uv project env: $ENV_DIR" >&2
  echo "[agent-lightning-verl-env] create it first with: UV_PROJECT_ENVIRONMENT=\"$ENV_DIR\" bash tools/envs/create_agent_lightning_verl_env.sh" >&2
  exit 1
fi

exec uv run --extra agents --extra train python tools/train/train_agent_lightning_games_verl.py "$@"
