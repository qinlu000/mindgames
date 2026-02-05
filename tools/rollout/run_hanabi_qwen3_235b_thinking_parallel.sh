#!/usr/bin/env bash
set -euo pipefail

# Parallel Hanabi rollouts with OpenAI-compatible agent (OpenRouter by default).
# Usage: bash tools/rollout/run_hanabi_qwen3_235b_thinking_parallel.sh
#
# Defaults (override via env vars):
#   MODEL=qwen/qwen3-235b-a22b-thinking-2507
#   EPISODES=100
#   WORKERS=50
#   SEED=0
#   OUT_ROOT=outputs/hanabi_qwen3_235b_thinking_parallel
#   TEMPERATURE=0.2
#   TOP_P=1.0
#   MAX_TOKENS=8192
#   NUM_PLAYERS=2
#   SUMMARIZE=1
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1
#
# Required:
#   OPENAI_API_KEY must be set (use your OpenRouter key when BASE_URL is OpenRouter).

MODEL="${MODEL:-qwen/qwen3-235b-a22b-thinking-2507}"
EPISODES="${EPISODES:-100}"
WORKERS="${WORKERS:-50}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/hanabi_qwen3_235b_thinking_parallel}"

TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
SUMMARIZE="${SUMMARIZE:-1}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required (use your OpenRouter key when BASE_URL is OpenRouter)." >&2
  exit 1
fi

if [[ "$WORKERS" -lt 1 ]]; then
  echo "WORKERS must be >= 1" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

base=$((EPISODES / WORKERS))
rem=$((EPISODES % WORKERS))
offset="${SEED}"

pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  eps="${base}"
  if [[ "$w" -lt "$rem" ]]; then
    eps=$((eps + 1))
  fi
  if [[ "$eps" -le 0 ]]; then
    continue
  fi

  out_dir="${OUT_ROOT}/worker_${w}"
  mkdir -p "${out_dir}"

  (
    uv run python tools/rollout/run_rollouts.py \
      --env-id Hanabi-v0-train \
      --num-players "${NUM_PLAYERS}" \
      --episodes "${eps}" \
      --seed "${offset}" \
      --agent "openai:${MODEL}" \
      --agent "openai:${MODEL}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --max-tokens "${MAX_TOKENS}" \
      --episode-json-dir "${out_dir}/episodes" \
      --out "${out_dir}/rollouts.jsonl" \
      > "${out_dir}/run.log" 2>&1
  ) &

  pids+=("$!")
  offset=$((offset + eps))
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "One or more workers failed. Check ${OUT_ROOT}/worker_*/run.log" >&2
  exit 1
fi

if [[ "${SUMMARIZE}" == "1" || "${SUMMARIZE}" == "true" ]]; then
  mapfile -t files < <(ls -1 "${OUT_ROOT}"/worker_*/rollouts.jsonl 2>/dev/null || true)
  if [[ "${#files[@]}" -gt 0 ]]; then
    uv run python tools/rollout/summarize_rollouts.py "${files[@]}" --json > "${OUT_ROOT}/summary.json"
  fi
fi
