#!/usr/bin/env bash
set -euo pipefail

# Run 500 Hanabi episodes with Qwen3-8B via a local vLLM server.
# Usage: bash tools/rollout/run_hanabi_qwen3_8b_vllm_500.sh

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

EPISODES="${EPISODES:-500}"
SEED="${SEED:-0}"
OUT_DIR="${OUT_DIR:-outputs/hanabi_qwen3_8b_500}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES

uv run vllm serve "${MODEL}" \
  --host "${HOST}" --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --trust-remote-code \
  --dtype bfloat16 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    kill "${VLLM_PID}"
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

python - "${PORT}" <<'PY'
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/models"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(2)
sys.exit("vLLM server did not become ready in time.")
PY

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

disable_thinking_flag=()
if [[ "${DISABLE_THINKING}" == "1" || "${DISABLE_THINKING}" == "true" ]]; then
  disable_thinking_flag=(--disable-thinking)
fi

max_tokens_flag=()
if [[ -n "${MAX_TOKENS}" && "${MAX_TOKENS}" != "None" && "${MAX_TOKENS}" != "null" ]]; then
  max_tokens_flag=(--max-tokens "${MAX_TOKENS}")
fi

uv run python tools/rollout/run_rollouts.py \
  --env-id Hanabi-v0-train \
  --num-players 2 \
  --episodes "${EPISODES}" \
  --seed "${SEED}" \
  --agent "qwen:${MODEL}" \
  --agent "qwen:${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  "${max_tokens_flag[@]}" \
  "${disable_thinking_flag[@]}" \
  --resume \
  --episode-json-dir "${OUT_DIR}/episodes" \
  --out "${OUT_DIR}/rollouts.jsonl"

uv run python tools/rollout/summarize_rollouts.py "${OUT_DIR}/rollouts.jsonl" \
  --json > "${OUT_DIR}/summary.json"
