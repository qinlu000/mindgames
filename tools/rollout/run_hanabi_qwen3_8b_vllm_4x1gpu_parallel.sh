#!/usr/bin/env bash
set -euo pipefail

# Data-parallel Hanabi rollouts with multiple 1-GPU vLLM replicas.
#
# This starts one vLLM server per GPU (TP=1), then shards EPISODES across WORKERS.
# Each worker runs `tools/rollout/run_rollouts.py` and targets a specific replica.
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   ENV_ID=Hanabi-v0-train
#   NUM_PLAYERS=2
#   EPISODES=100
#   SEED=0
#   DISABLE_THINKING=1
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3   # used to select GPUs (one server per id)
#   HOST=127.0.0.1
#   BASE_PORT=8000                 # replica ports: BASE_PORT + i
#
#   GPU_MEM_UTIL=0.90
#   MAX_MODEL_LEN=8192
#   MAX_NUM_SEQS=32
#   DTYPE=bfloat16
#
#   WORKERS=<auto: 2x num_replicas>
#   OUT_DIR=<required when running standalone; expctl sets RUN_DIR/OUT_DIR>
#   SYSTEM_PROMPT=<optional; if set, passed to run_rollouts.py>
#
# Outputs:
#   ${OUT_DIR}/worker_<w>/rollouts.jsonl
#   ${OUT_DIR}/rollouts.jsonl              (concatenated)
#   ${OUT_DIR}/summary.json

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ENV_ID="${ENV_ID:-Hanabi-v0-train}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
EXTRA_BODY="${EXTRA_BODY:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HOST="${HOST:-127.0.0.1}"
BIND_HOST="${BIND_HOST:-${HOST}}"
BASE_PORT="${BASE_PORT:-8000}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
DTYPE="${DTYPE:-bfloat16}"

OUT_DIR="${OUT_DIR:-${RUN_DIR:-}}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

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

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must include at least 1 GPU id" >&2
  exit 1
fi
NUM_REPLICAS="${#GPU_IDS[@]}"

WORKERS="${WORKERS:-$((NUM_REPLICAS * 2))}"
if [[ "${WORKERS}" -lt 1 ]]; then
  echo "WORKERS must be >= 1" >&2
  exit 1
fi

VLLM_LOG_DIR="${VLLM_LOG_DIR:-${OUT_DIR}/vllm_logs}"
mkdir -p "${VLLM_LOG_DIR}"

vllm_pids=()
replica_ports=()

cleanup() {
  # Kill replicas in reverse order (best effort).
  for ((i=${#vllm_pids[@]}-1; i>=0; i--)); do
    pid="${vllm_pids[$i]}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" || true
      wait "${pid}" || true
    fi
  done
}
trap cleanup EXIT

for i in $(seq 0 $((NUM_REPLICAS - 1))); do
  gpu="${GPU_IDS[$i]}"
  port=$((BASE_PORT + i))
  replica_ports+=("${port}")

  log="${VLLM_LOG_DIR}/replica_${i}_gpu${gpu}_port${port}.log"
  echo "Starting vLLM replica ${i} on GPU ${gpu} port ${port} (log: ${log})"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${VLLM[@]}" serve "${MODEL}" \
    --host "${BIND_HOST}" --port "${port}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --trust-remote-code \
    --dtype "${DTYPE}" \
    > "${log}" 2>&1 &
  vllm_pids+=("$!")
done

echo "Waiting for ${NUM_REPLICAS} replicas to become ready..."
for i in $(seq 0 $((NUM_REPLICAS - 1))); do
  port="${replica_ports[$i]}"
  ready=0
  for _ in $(seq 1 180); do
    pid="${vllm_pids[$i]}"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Replica ${i} exited early (port ${port}). Check: ${VLLM_LOG_DIR}/replica_${i}_*.log" >&2
      exit 1
    fi

    if "${PY[@]}" - "${HOST}" "${port}" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
url = f"http://{host}:{port}/v1/models"
with urllib.request.urlopen(url, timeout=2) as resp:
    raise SystemExit(0 if resp.status == 200 else 1)
PY
      ready=1
      break
    fi
    sleep 1
  done

  if [[ "${ready}" -ne 1 ]]; then
    echo "Replica ${i} did not become ready in time (port ${port})." >&2
    exit 1
  fi
done

echo "All replicas ready. Launching ${WORKERS} rollout workers for ${EPISODES} episodes..."

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

  replica_idx=$((w % NUM_REPLICAS))
  port="${replica_ports[$replica_idx]}"
  base_url="http://${HOST}:${port}/v1"

  out_dir="${OUT_DIR}/worker_${w}"
  mkdir -p "${out_dir}"

  echo "Launching worker ${w}: episodes=${eps} episode_id_offset=${global_ep} replica=${replica_idx} base_url=${base_url}"

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
      --openai-api-key "${OPENAI_API_KEY:-dummy}" \
      "${system_prompt_flag[@]}" \
      --temperature "${TEMPERATURE:-0.6}" \
      --top-p "${TOP_P:-0.95}" \
      --top-k "${TOP_K:-20}" \
      "${extra_body_flag[@]}" \
      --max-tokens "${MAX_TOKENS:-32}" \
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
