#!/usr/bin/env bash
set -euo pipefail

# Data-parallel Hanabi rollouts against ONE vLLM server (OpenAI-compatible).
#
# This starts a single vLLM server (optionally tensor-parallel across multiple GPUs),
# then shards EPISODES across WORKERS that concurrently call the same base_url.
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   ENV_ID=Hanabi-v0-train
#   NUM_PLAYERS=2
#   EPISODES=100
#   SEED=0
#   WORKERS=8
#   DISABLE_THINKING=1
#   EXTRA_BODY='{"min_p":0.0}'
#   SYSTEM_PROMPT=<optional; if set, passed to run_rollouts.py>
#
# vLLM server:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   HOST=127.0.0.1          # client connects to this host
#   BIND_HOST=${HOST}       # vLLM binds to this host (set 0.0.0.0 if needed)
#   PORT=8000
#   TP=<auto from CUDA_VISIBLE_DEVICES>
#   GPU_MEM_UTIL=0.90
#   MAX_MODEL_LEN=8192
#   MAX_NUM_SEQS=32
#   DTYPE=bfloat16
#   VLLM_API_KEY=dummy
#
# Output:
#   OUT_DIR=<required> (or set RUN_DIR)
#   ${OUT_DIR}/worker_<w>/rollouts.jsonl
#   ${OUT_DIR}/rollouts.jsonl
#   ${OUT_DIR}/summary.json

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ENV_ID="${ENV_ID:-Hanabi-v0-train}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
EXTRA_BODY="${EXTRA_BODY:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HOST="${HOST:-127.0.0.1}"
BIND_HOST="${BIND_HOST:-${HOST}}"
PORT="${PORT:-8000}"
TP="${TP:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
REASONING_PARSER="${REASONING_PARSER:-}"

OUT_DIR="${OUT_DIR:-${RUN_DIR:-}}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "OUT_DIR is required (or set RUN_DIR)." >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

if [[ "${NUM_PLAYERS}" -lt 1 ]]; then
  echo "NUM_PLAYERS must be >= 1" >&2
  exit 1
fi
if [[ "${EPISODES}" -lt 1 ]]; then
  echo "EPISODES must be >= 1" >&2
  exit 1
fi
if [[ "${WORKERS}" -lt 1 ]]; then
  echo "WORKERS must be >= 1" >&2
  exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
  PY=(.venv/bin/python)
  if [[ -x ".venv/bin/vllm" ]]; then
    VLLM=(.venv/bin/vllm)
  else
    VLLM=(vllm)
  fi
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
  VLLM=(uv run vllm)
else
  PY=(python)
  VLLM=(vllm)
fi

if [[ -z "${TP}" ]]; then
  IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${#_gpus[@]}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES must include at least 1 GPU id" >&2
    exit 1
  fi
  TP="${#_gpus[@]}"
fi

VLLM_LOG="${VLLM_LOG:-${OUT_DIR}/vllm.log}"
mkdir -p "$(dirname "${VLLM_LOG}")"

export CUDA_VISIBLE_DEVICES

echo "Starting vLLM server on ${BIND_HOST}:${PORT} (client: ${HOST}:${PORT}) TP=${TP} GPUs=${CUDA_VISIBLE_DEVICES}"
reasoning_flag=()
if [[ -n "${REASONING_PARSER}" ]]; then
  reasoning_flag=(--reasoning-parser "${REASONING_PARSER}")
fi
"${VLLM[@]}" serve "${MODEL}" \
  --host "${BIND_HOST}" --port "${PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  "${reasoning_flag[@]}" \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

echo "Waiting for vLLM to become ready... (log: ${VLLM_LOG})"
ready=0
for _ in $(seq 1 240); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "vLLM exited early. Check: ${VLLM_LOG}" >&2
    exit 1
  fi

  if "${PY[@]}" - "${HOST}" "${PORT}" "${VLLM_API_KEY}" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
api_key = sys.argv[3] if len(sys.argv) > 3 else ""
url = f"http://{host}:{port}/v1/models"
req = urllib.request.Request(url)
if api_key:
    req.add_header("Authorization", f"Bearer {api_key}")
with urllib.request.urlopen(req, timeout=2) as resp:
    raise SystemExit(0 if resp.status == 200 else 1)
PY
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "vLLM did not become ready in time. Check: ${VLLM_LOG}" >&2
  exit 1
fi

base_url="http://${HOST}:${PORT}/v1"
echo "vLLM ready: ${base_url}"

disable_thinking_flag=()
if [[ "${DISABLE_THINKING}" == "1" || "${DISABLE_THINKING,,}" == "true" ]]; then
  disable_thinking_flag=(--disable-thinking)
fi

system_prompt_flag=()
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  system_prompt_flag=(--system-prompt "${SYSTEM_PROMPT}")
fi

extra_body_flag=()
if [[ -n "${EXTRA_BODY}" ]]; then
  extra_body_flag=(--extra-body "${EXTRA_BODY}")
fi

max_tokens_flag=()
if [[ -n "${MAX_TOKENS}" && "${MAX_TOKENS}" != "None" && "${MAX_TOKENS}" != "null" ]]; then
  max_tokens_flag=(--max-tokens "${MAX_TOKENS}")
fi

echo "Launching ${WORKERS} workers for ${EPISODES} episodes (single server concurrency)..."

base=$((EPISODES / WORKERS))
rem=$((EPISODES % WORKERS))
global_ep=0

pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  eps="${base}"
  if [[ "${w}" -lt "${rem}" ]]; then
    eps=$((eps + 1))
  fi
  if [[ "${eps}" -le 0 ]]; then
    continue
  fi

  out_dir="${OUT_DIR}/worker_${w}"
  mkdir -p "${out_dir}"

  echo "Launching worker ${w}: episodes=${eps} episode_id_offset=${global_ep} base_url=${base_url}"

  (
    set +e
    echo "Worker ${w} start: $(date -Is)"
    "${PY[@]}" tools/rollout/run_rollouts.py \
      --env-id "${ENV_ID}" \
      --num-players "${NUM_PLAYERS}" \
      --episodes "${eps}" \
      --seed $((SEED + global_ep)) \
      --episode-id-offset "${global_ep}" \
      --agent "openai:${MODEL}" \
      --openai-base-url "${base_url}" \
      --openai-api-key "${VLLM_API_KEY}" \
      "${system_prompt_flag[@]}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --top-k "${TOP_K}" \
      "${extra_body_flag[@]}" \
      "${max_tokens_flag[@]}" \
      "${disable_thinking_flag[@]}" \
      --episode-json-dir "${out_dir}/episodes" \
      --out "${out_dir}/rollouts.jsonl"
    rc=$?
    echo "Worker ${w} done: $(date -Is) exit_code=${rc}"
    exit "${rc}"
  ) > "${out_dir}/run.log" 2>&1 &

  pids+=("$!")
  global_ep=$((global_ep + eps))
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "One or more workers failed. Check ${OUT_DIR}/worker_*/run.log" >&2
  exit 1
fi

mapfile -t worker_files < <(ls -1 "${OUT_DIR}"/worker_*/rollouts.jsonl 2>/dev/null || true)
if [[ "${#worker_files[@]}" -eq 0 ]]; then
  echo "No worker rollouts found under ${OUT_DIR}/worker_*/rollouts.jsonl" >&2
  exit 1
fi

cat "${worker_files[@]}" > "${OUT_DIR}/rollouts.jsonl"
"${PY[@]}" tools/rollout/summarize_rollouts.py "${worker_files[@]}" --json > "${OUT_DIR}/summary.json"

echo "Done."
echo "Rollouts: ${OUT_DIR}/rollouts.jsonl"
echo "Summary:  ${OUT_DIR}/summary.json"
echo "vLLM log: ${VLLM_LOG}"
