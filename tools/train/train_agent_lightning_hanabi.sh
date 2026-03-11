#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/workspace/models/Qwen3-8B}"
AGENT_KIND="${AGENT_KIND:-qwen}"
ENV_ID="${ENV_ID:-Hanabi-v0-train}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
TRAIN_EPISODES="${TRAIN_EPISODES:-128}"
VAL_EPISODES="${VAL_EPISODES:-32}"
TRAIN_SEED="${TRAIN_SEED:-0}"
VAL_SEED="${VAL_SEED:-100000}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-60}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY_S="${RETRY_DELAY_S:-0.5}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
REWARD_SCALE="${REWARD_SCALE:-25}"

uv run --extra agent-lightning --extra agents \
  python tools/train/train_agent_lightning_hanabi.py \
  --model "$MODEL" \
  --agent-kind "$AGENT_KIND" \
  --env-id "$ENV_ID" \
  --num-players "$NUM_PLAYERS" \
  --train-episodes "$TRAIN_EPISODES" \
  --val-episodes "$VAL_EPISODES" \
  --train-seed "$TRAIN_SEED" \
  --val-seed "$VAL_SEED" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --request-timeout-s "$REQUEST_TIMEOUT_S" \
  --max-retries "$MAX_RETRIES" \
  --retry-delay-s "$RETRY_DELAY_S" \
  --enable-thinking "$ENABLE_THINKING" \
  --reward-scale "$REWARD_SCALE" \
  "$@"
