#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-900}" # 15 min
TRAIN_SESSION="${TRAIN_SESSION:-hanabi_sft_ep16_sp4_main}"
TRAIN_LOG="${TRAIN_LOG:-output/logs/hanabi_sft_ep16_sp4_main_ep16-sp4-20260226-143118.log}"
TRAIN_PGREP_PATTERN="${TRAIN_PGREP_PATTERN:-swift sft .*qwen3-8b-hanabi-ge9-think-lora-4gpu-ws1-maxlen8192-right-ep16-sp4}"
WANDB_ENTITY_TARGET="${WANDB_ENTITY_TARGET:-}"
WANDB_PROJECT_TARGET="${WANDB_PROJECT_TARGET:-}"

MS_REPO_ID="${MS_REPO_ID:-qwen3-8b-hanabi-ge9-think-lora-4gpu-ws1-maxlen8192-right-ep16-sp4-merged}"
MS_COMMIT_MESSAGE="${MS_COMMIT_MESSAGE:-auto upload from ${TRAIN_SESSION}}"
MERGED_DIR="${MERGED_DIR:-output/merged/${MS_REPO_ID}}"

WATCH_LOG="${WATCH_LOG:-output/logs/watch_sft_post_${TRAIN_SESSION}.log}"
STATE_DIR="${STATE_DIR:-output/state}"
STATE_FILE="${STATE_DIR}/${TRAIN_SESSION}.post.done"

mkdir -p "$(dirname "$WATCH_LOG")" "$(dirname "$MERGED_DIR")" "$STATE_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"
}

choose_wandb_cmd() {
  if [ -x ".venv/bin/wandb" ]; then
    echo ".venv/bin/wandb"
    return
  fi
  echo "wandb"
}

get_last_offline_run() {
  if [ -f "$TRAIN_LOG" ]; then
    rg -o "/[^[:space:]]*wandb/offline-run-[^[:space:]]+" "$TRAIN_LOG" | tail -n 1 || true
  fi
}

get_last_model_checkpoint() {
  if [ -f "$TRAIN_LOG" ]; then
    local ckpt
    ckpt="$(rg -o "last_model_checkpoint:\\s+/[^[:space:]]+" "$TRAIN_LOG" | tail -n 1 | awk '{print $2}' || true)"
    if [ -n "${ckpt:-}" ] && [ -d "$ckpt" ]; then
      echo "$ckpt"
      return
    fi
    ckpt="$(rg -o "/[^[:space:]]+/checkpoint-[0-9]+" "$TRAIN_LOG" | tail -n 1 || true)"
    if [ -n "${ckpt:-}" ] && [ -d "$ckpt" ]; then
      echo "$ckpt"
    fi
  fi
}

is_training_running() {
  pgrep -af "$TRAIN_PGREP_PATTERN" >/dev/null 2>&1
}

is_training_success() {
  [ -f "$TRAIN_LOG" ] || return 1
  rg -q "last_model_checkpoint:\\s+/" "$TRAIN_LOG" &&
    rg -q "End time of running main:" "$TRAIN_LOG"
}

is_training_failed() {
  [ -f "$TRAIN_LOG" ] || return 1
  rg -q "Traceback \\(most recent call last\\)|ChildFailedError|OutOfMemoryError" "$TRAIN_LOG"
}

sync_wandb() {
  local wandb_cmd
  wandb_cmd="$(choose_wandb_cmd)"
  if ! command -v "${wandb_cmd%% *}" >/dev/null 2>&1 && [ "$wandb_cmd" = "wandb" ]; then
    log "WARN: wandb command not found, skip sync."
    return 0
  fi

  local run_dir
  run_dir="$(get_last_offline_run)"
  if [ -z "${run_dir:-}" ] || [ ! -d "$run_dir" ]; then
    run_dir="$(ls -1dt wandb/offline-run-* 2>/dev/null | head -n 1 || true)"
  fi

  if [ -z "${run_dir:-}" ] || [ ! -d "$run_dir" ]; then
    log "WARN: no offline wandb run found, skip sync."
    return 0
  fi

  local sync_args=()
  if [ -n "${WANDB_ENTITY_TARGET:-}" ]; then
    sync_args+=(--entity "$WANDB_ENTITY_TARGET")
  fi
  if [ -n "${WANDB_PROJECT_TARGET:-}" ]; then
    sync_args+=(--project "$WANDB_PROJECT_TARGET")
  fi

  log "Start wandb sync: $run_dir (entity=${WANDB_ENTITY_TARGET:-default}, project=${WANDB_PROJECT_TARGET:-default})"
  if ! $wandb_cmd sync "${sync_args[@]}" "$run_dir" 2>&1 | tee -a "$WATCH_LOG"; then
    log "ERROR: wandb sync failed."
    return 1
  fi
  log "wandb sync finished."
}

merge_lora() {
  local ckpt_dir="$1"
  log "Start local merge: ckpt_dir=$ckpt_dir -> $MERGED_DIR"
  .venv/bin/swift export \
    --ckpt_dir "$ckpt_dir" \
    --merge_lora true \
    --device_map auto \
    --output_dir "$MERGED_DIR" \
    --exist_ok true \
    2>&1 | tee -a "$WATCH_LOG"
  log "Local merge finished."
}

push_modelscope() {
  local ms_token="${MODELSCOPE_API_TOKEN:-}"
  if [ -z "${ms_token:-}" ]; then
    ms_token="$(
      .venv/bin/python - <<'PY' 2>/dev/null || true
from modelscope.hub.api import ModelScopeConfig
tok = ModelScopeConfig.get_token()
if tok:
    print(tok)
PY
    )"
  fi
  if [ -z "${ms_token:-}" ]; then
    log "ERROR: no ModelScope token found in MODELSCOPE_API_TOKEN or local modelscope config."
    return 1
  fi
  log "Start push to ModelScope: repo=$MS_REPO_ID"
  .venv/bin/swift export \
    --model "$MERGED_DIR" \
    --push_to_hub true \
    --use_hf false \
    --hub_model_id "$MS_REPO_ID" \
    --hub_token "$ms_token" \
    --commit_message "$MS_COMMIT_MESSAGE" \
    2>&1 | tee -a "$WATCH_LOG"
  log "Push to ModelScope finished."
}

if [ -f "$STATE_FILE" ]; then
  log "state exists ($STATE_FILE), watcher exits."
  exit 0
fi

log "watcher started: interval=${CHECK_INTERVAL_SEC}s"
log "train_session=$TRAIN_SESSION"
log "train_log=$TRAIN_LOG"
log "wandb_target=${WANDB_ENTITY_TARGET:-default}/${WANDB_PROJECT_TARGET:-default}"
log "modelscope_repo=$MS_REPO_ID"
log "merged_dir=$MERGED_DIR"

while true; do
  if is_training_running; then
    log "training still running, sleep ${CHECK_INTERVAL_SEC}s."
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  log "training process not found, checking train log."
  if ! [ -f "$TRAIN_LOG" ]; then
    log "WARN: train log not found yet, sleep ${CHECK_INTERVAL_SEC}s."
    sleep "$CHECK_INTERVAL_SEC"
    continue
  fi

  if is_training_success; then
    ckpt_dir="$(get_last_model_checkpoint || true)"
    if [ -z "${ckpt_dir:-}" ] || [ ! -d "$ckpt_dir" ]; then
      log "ERROR: cannot find last checkpoint from train log."
      exit 1
    fi

    sync_wandb
    merge_lora "$ckpt_dir"
    push_modelscope

    date '+%F %T' >"$STATE_FILE"
    log "all post actions completed. state saved to $STATE_FILE"
    exit 0
  fi

  if is_training_failed; then
    log "ERROR: detected training failure markers in log. stop post actions."
    exit 1
  fi

  log "training not running but success/failure markers are incomplete, sleep ${CHECK_INTERVAL_SEC}s."
  sleep "$CHECK_INTERVAL_SEC"
done
