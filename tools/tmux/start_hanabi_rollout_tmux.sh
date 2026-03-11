#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SESSION="${SESSION:-hanabi_rollout}"
MODEL="${MODEL:-/workspace/models/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORTS="${PORTS:-8100,8101,8102,8103,8104}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
CONTEXT_MANAGER="${CONTEXT_MANAGER:-hanabi_recent_turns}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
MAX_TURNS="${MAX_TURNS:-}"
HANABI_CTX_MAX_TURNS="${HANABI_CTX_MAX_TURNS:-1}"
HANABI_CTX_KEEP_SYSTEM="${HANABI_CTX_KEEP_SYSTEM:-true}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"
VLLM_USE_ASYNC_ENGINE="${VLLM_USE_ASYNC_ENGINE:-true}"
SWIFT_BIN="${SWIFT_BIN:-}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
STARTUP_GAP_SEC="${STARTUP_GAP_SEC:-2}"
FORCE_RESTART="${FORCE_RESTART:-false}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/tmux/$SESSION}"

if [ ! -d "$MODEL" ]; then
  MODEL="Qwen/Qwen3-8B"
fi

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

parse_csv "$PORTS" PORT_ARR
parse_csv "$GPU_LIST" GPU_ARR

if [ "${#PORT_ARR[@]}" -eq 0 ]; then
  echo "PORTS is empty" >&2
  exit 1
fi
if [ "${#PORT_ARR[@]}" -ne "${#GPU_ARR[@]}" ]; then
  echo "PORTS count (${#PORT_ARR[@]}) must match GPU_LIST count (${#GPU_ARR[@]})." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if is_true "$FORCE_RESTART"; then
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session '$SESSION' already exists. Set FORCE_RESTART=true to recreate." >&2
    exit 0
  fi
fi

for idx in "${!PORT_ARR[@]}"; do
  port="${PORT_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]}"
  win_name="r${idx}"
  log_file="$LOG_DIR/rollout_${port}.log"
  cmd="cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=$gpu MODEL=\"$MODEL\" HOST=\"$HOST\" PORT=$port CONTEXT_MANAGER=\"$CONTEXT_MANAGER\" ENABLE_THINKING=\"$ENABLE_THINKING\" MAX_TURNS=\"$MAX_TURNS\" HANABI_CTX_MAX_TURNS=\"$HANABI_CTX_MAX_TURNS\" HANABI_CTX_KEEP_SYSTEM=\"$HANABI_CTX_KEEP_SYSTEM\" VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS VLLM_ENABLE_LORA=$VLLM_ENABLE_LORA VLLM_MAX_LORA_RANK=$VLLM_MAX_LORA_RANK VLLM_USE_ASYNC_ENGINE=$VLLM_USE_ASYNC_ENGINE SWIFT_BIN=\"$SWIFT_BIN\" NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_IB_DISABLE=$NCCL_IB_DISABLE bash tools/rollout/rollout_hanabi_gym_simple.sh 2>&1 | tee -a \"$log_file\""

  if [ "$idx" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "$win_name" "$cmd"
    tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
  else
    tmux new-window -d -t "$SESSION:" -n "$win_name" "$cmd"
  fi
  sleep "$STARTUP_GAP_SEC"
done

echo "Started rollout tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
tmux list-windows -t "$SESSION"
