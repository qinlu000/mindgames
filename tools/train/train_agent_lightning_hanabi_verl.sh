#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${AGENT_LIGHTNING_VERL_ENV_DIR:-${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv-agent-lightning-verl}}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[agent-lightning-verl-env] missing environment: $ENV_DIR" >&2
  echo "[agent-lightning-verl-env] create it first with: ENV_DIR=\"$ENV_DIR\" bash tools/envs/create_agent_lightning_verl_env.sh" >&2
  exit 1
fi

exec "$ENV_DIR/bin/python" tools/train/train_agent_lightning_hanabi_verl.py "$@"
