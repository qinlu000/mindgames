#!/usr/bin/env bash
set -euo pipefail

ROLLOUT_SESSION="${ROLLOUT_SESSION:-hanabi_rollout}"
TRAIN_SESSION="${TRAIN_SESSION:-hanabi_train_long}"
PORTS="${PORTS:-8100,8101,8102,8103,8104}"
TAIL_LINES="${TAIL_LINES:-30}"

parse_csv() {
  local raw="$1"
  local -n out_ref="$2"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  out_ref=($raw)
}

parse_csv "$PORTS" PORT_ARR

echo "== tmux sessions =="
tmux ls || true

echo
echo "== rollout health =="
for port in "${PORT_ARR[@]}"; do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "port ${port}: ok"
  else
    echo "port ${port}: down"
  fi
done

echo
echo "== train tail (${TRAIN_SESSION}) =="
if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  tmux capture-pane -pt "${TRAIN_SESSION}:train" | tail -n "$TAIL_LINES"
else
  echo "session not found: $TRAIN_SESSION"
fi

echo
echo "== rollout tails (${ROLLOUT_SESSION}) =="
if tmux has-session -t "$ROLLOUT_SESSION" 2>/dev/null; then
  while IFS= read -r win; do
    echo "--- ${win} ---"
    tmux capture-pane -pt "${ROLLOUT_SESSION}:${win}" | tail -n 8
  done < <(tmux list-windows -t "$ROLLOUT_SESSION" -F '#W')
else
  echo "session not found: $ROLLOUT_SESSION"
fi
