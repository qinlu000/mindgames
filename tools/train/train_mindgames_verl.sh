#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${UV_PROJECT_ENVIRONMENT:-$ROOT_DIR/.venv}"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[mindgames-verl-env] missing project env: $ENV_DIR" >&2
  echo "[mindgames-verl-env] create it first with: UV_PROJECT_ENVIRONMENT=\"$ENV_DIR\" bash tools/envs/create_verl_env.sh" >&2
  exit 1
fi

LOCAL_NO_PROXY_ENTRIES=(localhost 127.0.0.1 ::1)
if command -v hostname >/dev/null 2>&1; then
  while read -r ip; do
    [[ -n "$ip" ]] || continue
    LOCAL_NO_PROXY_ENTRIES+=("$ip")
  done < <(hostname -I 2>/dev/null | tr ' ' '\n')
fi
LOCAL_NO_PROXY_CSV="$(IFS=,; echo "${LOCAL_NO_PROXY_ENTRIES[*]}")"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY_CSV"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY_CSV"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

exec "$ENV_DIR/bin/python" tools/train/train_mindgames_verl.py "$@"
