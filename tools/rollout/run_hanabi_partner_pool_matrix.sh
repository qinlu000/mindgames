#!/usr/bin/env bash
set -euo pipefail

# Cross-play matrix runner for Hanabi partner-pool research.
#
# What it does:
# 1) Starts one vLLM server per model (TP=1, one GPU per model).
# 2) Runs all ordered model pairs (A as player0, B as player1).
# 3) Saves pair rollouts + summaries.
# 4) Writes matrix and hard-partner reports.
#
# Usage:
#   MODEL_ROOT=output/merged \
#   OUT_DIR=outputs/hanabi_partner_pool_$(date +%Y%m%d_%H%M%S) \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash tools/rollout/run_hanabi_partner_pool_matrix.sh
#
# Optional:
#   MODELS_FILE=path/to/models.txt    # one model path per line
#   EPISODES=50
#   BASE_PORT=9100
#   VLLM_MAX_MODEL_LEN=8192
#   VLLM_MAX_NUM_SEQS=8
#   GPU_MEM_UTIL=0.90
#   TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20
#   DISABLE_THINKING=0               # passed to run_rollouts.py globally

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_ROOT="${MODEL_ROOT:-output/merged}"
MODELS_FILE="${MODELS_FILE:-}"
OUT_DIR="${OUT_DIR:-outputs/hanabi_partner_pool_$(date +%Y%m%d_%H%M%S)}"

ENV_ID="${ENV_ID:-Hanabi-v0-train}"
EPISODES="${EPISODES:-50}"
SEED="${SEED:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
HOST="${HOST:-127.0.0.1}"
BIND_HOST="${BIND_HOST:-${HOST}}"
BASE_PORT="${BASE_PORT:-9100}"
API_KEY="${API_KEY:-dummy}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
DTYPE="${DTYPE:-bfloat16}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"

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

mkdir -p "$OUT_DIR"

model_list_file="$OUT_DIR/models.txt"
if [[ -n "$MODELS_FILE" ]]; then
  if [[ ! -f "$MODELS_FILE" ]]; then
    echo "MODELS_FILE not found: $MODELS_FILE" >&2
    exit 1
  fi
  awk 'NF{print $0}' "$MODELS_FILE" > "$model_list_file"
else
  if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "MODEL_ROOT not found: $MODEL_ROOT" >&2
    exit 1
  fi
  find "$MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d | sort > "$model_list_file"
fi

mapfile -t MODELS < "$model_list_file"
if [[ "${#MODELS[@]}" -lt 2 ]]; then
  echo "Need at least 2 models. Found ${#MODELS[@]}." >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#GPUS[@]}" -lt "${#MODELS[@]}" ]]; then
  echo "Need at least ${#MODELS[@]} GPUs (one per model), got ${#GPUS[@]} via CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

echo "OUT_DIR=$OUT_DIR"
echo "ENV_ID=$ENV_ID EPISODES=$EPISODES"
echo "MODELS=${#MODELS[@]}"

server_pids=()
server_urls=()
server_logs=()

cleanup() {
  for ((i=${#server_pids[@]}-1; i>=0; i--)); do
    pid="${server_pids[$i]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
      wait "$pid" || true
    fi
  done
}
trap cleanup EXIT

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  model_name="$(basename "$model")"
  gpu="${GPUS[$i]}"
  port=$((BASE_PORT + i))
  url="http://${HOST}:${port}/v1"
  log="$OUT_DIR/vllm_${i}_${model_name}.log"
  server_urls+=("$url")
  server_logs+=("$log")

  echo "Start vLLM[$i] model=$model_name gpu=$gpu port=$port"
  CUDA_VISIBLE_DEVICES="$gpu" \
  "${VLLM[@]}" serve "$model" \
    --host "$BIND_HOST" --port "$port" \
    --api-key "$API_KEY" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    > "$log" 2>&1 &
  server_pids+=("$!")
done

echo "Wait for ${#MODELS[@]} vLLM servers..."
for i in "${!MODELS[@]}"; do
  pid="${server_pids[$i]}"
  url="${server_urls[$i]}"
  host="${HOST}"
  port=$((BASE_PORT + i))
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "vLLM[$i] exited early. log=${server_logs[$i]}" >&2
      exit 1
    fi
    if "${PY[@]}" - "$host" "$port" "$API_KEY" <<'PY' >/dev/null 2>&1; then
import sys
import urllib.request
host = sys.argv[1]
port = int(sys.argv[2])
api_key = sys.argv[3]
req = urllib.request.Request(f"http://{host}:{port}/v1/models")
if api_key:
    req.add_header("Authorization", f"Bearer {api_key}")
with urllib.request.urlopen(req, timeout=2) as r:
    raise SystemExit(0 if r.status == 200 else 1)
PY
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "vLLM[$i] not ready in time: $url (log=${server_logs[$i]})" >&2
    exit 1
  fi
  echo "vLLM[$i] ready: $url"
done

mkdir -p "$OUT_DIR/pairs"

for i in "${!MODELS[@]}"; do
  for j in "${!MODELS[@]}"; do
    m0="${MODELS[$i]}"
    m1="${MODELS[$j]}"
    n0="$(basename "$m0")"
    n1="$(basename "$m1")"
    pair_dir="$OUT_DIR/pairs/${n0}__vs__${n1}"
    mkdir -p "$pair_dir"
    echo "Run pair: $n0 (P0) vs $n1 (P1)"

    disable_flag=()
    if [[ "$DISABLE_THINKING" == "1" || "${DISABLE_THINKING,,}" == "true" ]]; then
      disable_flag=(--disable-thinking)
    fi

    "${PY[@]}" tools/rollout/run_rollouts.py \
      --env-id "$ENV_ID" \
      --num-players 2 \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --agent "openai:$m0" \
      --agent "openai:$m1" \
      --agent-openai-base-url "${server_urls[$i]}" \
      --agent-openai-base-url "${server_urls[$j]}" \
      --agent-openai-api-key "$API_KEY" \
      --agent-openai-api-key "$API_KEY" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      "${disable_flag[@]}" \
      --episode-json-dir "$pair_dir/episodes" \
      --out "$pair_dir/rollouts.jsonl" \
      > "$pair_dir/run.log" 2>&1

    "${PY[@]}" tools/rollout/summarize_rollouts.py "$pair_dir/rollouts.jsonl" --json > "$pair_dir/summary.json"
  done
done

"${PY[@]}" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1]).resolve()
pairs_dir = out_dir / "pairs"
pair_dirs = sorted([p for p in pairs_dir.iterdir() if p.is_dir() and "__vs__" in p.name])

models = sorted(set([x for p in pair_dirs for x in p.name.split("__vs__")]))
idx = {m: i for i, m in enumerate(models)}

matrix = [[None for _ in models] for _ in models]
pair_rows = []
for d in pair_dirs:
    m0, m1 = d.name.split("__vs__", 1)
    s = d / "summary.json"
    if not s.exists():
        continue
    data = json.loads(s.read_text(encoding="utf-8"))
    envs = data.get("envs") or {}
    if not envs:
        continue
    env = envs[next(iter(envs))]
    score = env.get("avg_score_if_coop")
    episodes = env.get("episodes")
    matrix[idx[m0]][idx[m1]] = score
    pair_rows.append(
        {
            "p0": m0,
            "p1": m1,
            "episodes": episodes,
            "avg_score_if_coop": score,
            "summary": str(s),
        }
    )

hard_partners = []
for m in models:
    i = idx[m]
    vals = [(models[j], matrix[i][j]) for j in range(len(models)) if matrix[i][j] is not None]
    vals_sorted = sorted(vals, key=lambda x: (x[1], x[0]))
    hard_partners.append({"model": m, "hardest_3_partners": vals_sorted[:3]})

out_json = {
    "models": models,
    "matrix_avg_score_if_coop": matrix,
    "pairs": pair_rows,
    "hard_partners": hard_partners,
}
(out_dir / "matrix.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# TSV matrix
lines = []
lines.append("\t" + "\t".join(models))
for i, m in enumerate(models):
    row = [m]
    for j in range(len(models)):
        v = matrix[i][j]
        row.append("" if v is None else str(v))
    lines.append("\t".join(row))
(out_dir / "matrix.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Hard partner text
txt = []
for item in hard_partners:
    txt.append(item["model"])
    for name, score in item["hardest_3_partners"]:
        txt.append(f"  - {name}: {score}")
    txt.append("")
(out_dir / "hard_partners.txt").write_text("\n".join(txt), encoding="utf-8")

print(f"matrix_json={out_dir / 'matrix.json'}")
print(f"matrix_tsv={out_dir / 'matrix.tsv'}")
print(f"hard_partners={out_dir / 'hard_partners.txt'}")
PY

echo "Done."
echo "OUT_DIR=$OUT_DIR"
echo "Pairs dir: $OUT_DIR/pairs"
echo "Matrix:    $OUT_DIR/matrix.tsv"
echo "Hardest:   $OUT_DIR/hard_partners.txt"

