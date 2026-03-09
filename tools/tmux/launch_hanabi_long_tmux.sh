#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ROLLOUT_SESSION="${ROLLOUT_SESSION:-hanabi_rollout}"
TRAIN_SESSION="${TRAIN_SESSION:-hanabi_train_long}"

PORTS="${PORTS:-8100,8101,8102,8103,8104}"
ROLLOUT_GPU_LIST="${ROLLOUT_GPU_LIST:-0,1,2,3,4}"
RUN_NAME="${RUN_NAME:-hanabi-grpo-long-$(date +%Y%m%d_%H%M%S)}"

STOP_FOREGROUND="${STOP_FOREGROUND:-true}"
HEALTH_RETRIES="${HEALTH_RETRIES:-120}"
HEALTH_SLEEP_SEC="${HEALTH_SLEEP_SEC:-2}"
GPU_IDLE_TIMEOUT_SEC="${GPU_IDLE_TIMEOUT_SEC:-180}"
GPU_IDLE_MB_THRESHOLD="${GPU_IDLE_MB_THRESHOLD:-2000}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

parse_csv() {
  local raw="$1"
  local -n out_ref="$2"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  out_ref=($raw)
}

kill_by_match() {
  local match="$1"
  local pids
  pids="$(pgrep -f "$match" || true)"
  if [ -z "$pids" ]; then
    return 0
  fi
  echo "Stopping processes matching: $match"
  echo "$pids" | xargs -r kill -TERM
  sleep 5
  pids="$(pgrep -f "$match" || true)"
  if [ -n "$pids" ]; then
    echo "$pids" | xargs -r kill -KILL
  fi
}

parse_csv "$PORTS" PORT_ARR
parse_csv "$ROLLOUT_GPU_LIST" ROLLOUT_GPU_ARR

wait_rollout_gpus_idle() {
  local timeout_sec="$1"
  local threshold_mb="$2"
  local deadline
  deadline=$((SECONDS + timeout_sec))

  while [ "$SECONDS" -lt "$deadline" ]; do
    local ok=true
    local gpu_info
    gpu_info="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$gpu_info" ]; then
      sleep 2
      continue
    fi

    for gpu in "${ROLLOUT_GPU_ARR[@]}"; do
      local used
      used="$(echo "$gpu_info" | awk -F',' -v idx="$gpu" '$1 ~ ("^"idx"$") {gsub(/ /, "", $2); print $2}')"
      if [ -n "$used" ] && [ "$used" -gt "$threshold_mb" ]; then
        ok=false
        break
      fi
    done

    if [ "$ok" = "true" ]; then
      return 0
    fi
    sleep 2
  done

  return 1
}

if is_true "$STOP_FOREGROUND"; then
  kill_by_match "swift rlhf --rlhf_type grpo"
  kill_by_match "swift/cli/rlhf.py --rlhf_type grpo"
  kill_by_match "torch.distributed.run --nproc_per_node"
  for port in "${PORT_ARR[@]}"; do
    kill_by_match "swift rollout .*--port ${port}( |$)"
  done
  kill_by_match "VLLM::EngineCore"
  if ! wait_rollout_gpus_idle "$GPU_IDLE_TIMEOUT_SEC" "$GPU_IDLE_MB_THRESHOLD"; then
    echo "Rollout GPUs still busy after cleanup timeout." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    exit 1
  fi
fi

cd "$ROOT_DIR"

FORCE_RESTART=true SESSION="$ROLLOUT_SESSION" PORTS="$PORTS" \
  bash tools/tmux/start_hanabi_rollout_tmux.sh

for port in "${PORT_ARR[@]}"; do
  ok=false
  for ((i = 1; i <= HEALTH_RETRIES; i++)); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ok=true
      break
    fi
    sleep "$HEALTH_SLEEP_SEC"
  done
  if [ "$ok" != "true" ]; then
    echo "Rollout server port ${port} health check failed." >&2
    exit 1
  fi
done

FORCE_RESTART=true SESSION="$TRAIN_SESSION" RUN_NAME="$RUN_NAME" \
  bash tools/tmux/start_hanabi_train_long_tmux.sh

echo
echo "All set."
echo "Sessions:"
tmux ls
echo
echo "Attach training: tmux attach -t $TRAIN_SESSION"
echo "Attach rollout:  tmux attach -t $ROLLOUT_SESSION"
