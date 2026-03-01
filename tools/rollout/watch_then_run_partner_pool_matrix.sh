#!/usr/bin/env bash
set -euo pipefail

# Wait until a Hanabi batch eval fully completes, then launch partner-pool
# cross-play matrix research automatically.
#
# Usage:
#   OUT_ROOT=outputs/hanabi_merged_4x100_YYYYMMDD_HHMMSS \
#   bash tools/rollout/watch_then_run_partner_pool_matrix.sh
#
# Optional env vars:
#   EXPECT_MODELS=4
#   CHECK_INTERVAL=300
#   MODELS_FILE=<default: ${OUT_ROOT}/models.txt>
#   MATRIX_OUT=outputs/hanabi_partner_pool_from_<out_root>_<ts>
#   MATRIX_CUDA_VISIBLE_DEVICES=0,1,2,3
#   MATRIX_EPISODES=50
#   MATRIX_SEED=0
#   MATRIX_BASE_PORT=9100
#   MATRIX_GPU_MEM_UTIL=0.90
#   MATRIX_MAX_MODEL_LEN=8192
#   MATRIX_MAX_NUM_SEQS=8
#   MATRIX_TEMPERATURE=0.6 MATRIX_TOP_P=0.95 MATRIX_TOP_K=20
#   MATRIX_DISABLE_THINKING=0
#   LOG_FILE=<default: ${OUT_ROOT}/research_phase1.log>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-}"
EXPECT_MODELS="${EXPECT_MODELS:-4}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"

if [[ -z "$OUT_ROOT" ]]; then
  echo "OUT_ROOT is required." >&2
  exit 1
fi
if [[ ! -d "$OUT_ROOT" ]]; then
  echo "OUT_ROOT not found: $OUT_ROOT" >&2
  exit 1
fi

MODELS_FILE="${MODELS_FILE:-$OUT_ROOT/models.txt}"
ts="$(date +%Y%m%d_%H%M%S)"
MATRIX_OUT="${MATRIX_OUT:-outputs/hanabi_partner_pool_from_$(basename "$OUT_ROOT")_${ts}}"
MATRIX_CUDA_VISIBLE_DEVICES="${MATRIX_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MATRIX_EPISODES="${MATRIX_EPISODES:-50}"
MATRIX_SEED="${MATRIX_SEED:-0}"
MATRIX_BASE_PORT="${MATRIX_BASE_PORT:-9100}"
MATRIX_GPU_MEM_UTIL="${MATRIX_GPU_MEM_UTIL:-0.90}"
MATRIX_MAX_MODEL_LEN="${MATRIX_MAX_MODEL_LEN:-8192}"
MATRIX_MAX_NUM_SEQS="${MATRIX_MAX_NUM_SEQS:-8}"
MATRIX_TEMPERATURE="${MATRIX_TEMPERATURE:-0.6}"
MATRIX_TOP_P="${MATRIX_TOP_P:-0.95}"
MATRIX_TOP_K="${MATRIX_TOP_K:-20}"
MATRIX_DISABLE_THINKING="${MATRIX_DISABLE_THINKING:-0}"

LOG_FILE="${LOG_FILE:-$OUT_ROOT/research_phase1.log}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

count_summaries() {
  find "$OUT_ROOT" -maxdepth 2 -type f -name 'summary.json' | wc -l
}

count_active_rollouts() {
  pgrep -af "tools/rollout/run_rollouts.py" | rg -F "$OUT_ROOT" | wc -l || true
}

has_leaderboard() {
  [[ -f "$OUT_ROOT/leaderboard.json" ]] && echo "yes" || echo "no"
}

log "watch started: OUT_ROOT=$OUT_ROOT EXPECT_MODELS=$EXPECT_MODELS CHECK_INTERVAL=$CHECK_INTERVAL"
while true; do
  summaries="$(count_summaries)"
  active_rollouts="$(count_active_rollouts)"
  leaderboard="$(has_leaderboard)"
  log "status: summaries=${summaries}/${EXPECT_MODELS} active_rollouts=${active_rollouts} leaderboard=${leaderboard}"

  if [[ "$summaries" -ge "$EXPECT_MODELS" && "$active_rollouts" -eq 0 && "$leaderboard" == "yes" ]]; then
    break
  fi
  sleep "$CHECK_INTERVAL"
done

if [[ ! -f "$MODELS_FILE" ]]; then
  log "MODELS_FILE not found: $MODELS_FILE"
  exit 1
fi

model_count="$(awk 'NF{c++} END{print c+0}' "$MODELS_FILE")"
if [[ "$model_count" -lt 2 ]]; then
  log "MODELS_FILE has fewer than 2 models: $MODELS_FILE"
  exit 1
fi

log "batch completed. launching partner-pool matrix."
log "matrix config: OUT=$MATRIX_OUT MODELS=$model_count EPISODES=$MATRIX_EPISODES SEED=$MATRIX_SEED CUDA_VISIBLE_DEVICES=$MATRIX_CUDA_VISIBLE_DEVICES"

MODEL_ROOT="output/merged" \
MODELS_FILE="$MODELS_FILE" \
OUT_DIR="$MATRIX_OUT" \
CUDA_VISIBLE_DEVICES="$MATRIX_CUDA_VISIBLE_DEVICES" \
EPISODES="$MATRIX_EPISODES" \
SEED="$MATRIX_SEED" \
BASE_PORT="$MATRIX_BASE_PORT" \
GPU_MEM_UTIL="$MATRIX_GPU_MEM_UTIL" \
VLLM_MAX_MODEL_LEN="$MATRIX_MAX_MODEL_LEN" \
VLLM_MAX_NUM_SEQS="$MATRIX_MAX_NUM_SEQS" \
TEMPERATURE="$MATRIX_TEMPERATURE" \
TOP_P="$MATRIX_TOP_P" \
TOP_K="$MATRIX_TOP_K" \
DISABLE_THINKING="$MATRIX_DISABLE_THINKING" \
bash tools/rollout/run_hanabi_partner_pool_matrix.sh 2>&1 | tee -a "$LOG_FILE"

log "partner-pool matrix finished: $MATRIX_OUT"
log "matrix file: $MATRIX_OUT/matrix.tsv"
log "hard partners: $MATRIX_OUT/hard_partners.txt"
