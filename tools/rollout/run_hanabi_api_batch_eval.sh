#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate many API models on Hanabi (self-play: same model as P0/P1).
#
# Defaults target the user request:
# - use all models from MODELS_FILE (MIN_MODELS defaults to file count)
# - 10 episodes per model
# - OpenAI-compatible API call path
#
# Usage:
#   OPENAI_API_KEY=... \
#   OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
#   MODELS_FILE=data/hanabi_api_models_15.txt \
#   bash tools/rollout/run_hanabi_api_batch_eval.sh
#
# Optional common overrides:
#   OUT_ROOT=outputs/hanabi_api_batch_$(date +%Y%m%d_%H%M%S)
#   EPISODES=10
#   MIN_MODELS=11               # optional lower bound; default is model file count
#   AGENT_KIND=openai              # openai|qwen|openrouter|gemini|...
#   MODEL_GEN_FILE=path/to/model_gen_overrides.json
#   PARALLEL_JOBS=15               # >1: run models in parallel
#   CONTINUE_ON_ERROR=1
#   DRY_RUN=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODELS_FILE="${MODELS_FILE:-data/hanabi_api_models_15.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/hanabi_api_batch_$(date +%Y%m%d_%H%M%S)}"

ENV_ID="${ENV_ID:-Hanabi-v0-train}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
EPISODES="${EPISODES:-10}"
SEED="${SEED:-0}"
RANDOMIZE_SEED_PER_MODEL="${RANDOMIZE_SEED_PER_MODEL:-0}"
MIN_MODELS="${MIN_MODELS:-}"

AGENT_KIND="${AGENT_KIND:-openai}"
MODEL_GEN_FILE="${MODEL_GEN_FILE:-}"

TIMEOUT="${TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_INITIAL_DELAY="${RETRY_INITIAL_DELAY:-0}"
RETRY_MAX_DELAY="${RETRY_MAX_DELAY:-0}"
# By default do not send sampling params; let provider-side model defaults apply.
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
STREAM="${STREAM:-0}"

OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

safe_name() {
  local s="$1"
  s="${s//\//__}"
  s="${s//:/_}"
  s="${s// /_}"
  s="$(printf '%s' "$s" | sed 's/[^A-Za-z0-9._-]/_/g')"
  printf '%s' "$s"
}

rand_seed() {
  echo $(( (RANDOM << 16) | RANDOM ))
}

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "MODELS_FILE not found: $MODELS_FILE" >&2
  exit 1
fi
if [[ -n "$MODEL_GEN_FILE" && ! -f "$MODEL_GEN_FILE" ]]; then
  echo "MODEL_GEN_FILE not found: $MODEL_GEN_FILE" >&2
  exit 1
fi

if ! [[ "$EPISODES" =~ ^[0-9]+$ ]] || [[ "$EPISODES" -lt 1 ]]; then
  echo "EPISODES must be a positive integer, got: $EPISODES" >&2
  exit 1
fi
if [[ -n "$MIN_MODELS" ]]; then
  if ! [[ "$MIN_MODELS" =~ ^[0-9]+$ ]] || [[ "$MIN_MODELS" -lt 1 ]]; then
    echo "MIN_MODELS must be a positive integer, got: $MIN_MODELS" >&2
    exit 1
  fi
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "PARALLEL_JOBS must be a positive integer, got: $PARALLEL_JOBS" >&2
  exit 1
fi

mapfile -t MODELS < <(
  awk '
    {
      line=$0
      sub(/^[ \t]+/, "", line)
      sub(/[ \t]+$/, "", line)
      if (line == "" || line ~ /^#/) next
      print line
    }
  ' "$MODELS_FILE"
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models found in ${MODELS_FILE}" >&2
  exit 1
fi
if [[ -z "$MIN_MODELS" ]]; then
  MIN_MODELS="${#MODELS[@]}"
fi
if [[ "${#MODELS[@]}" -lt "$MIN_MODELS" ]]; then
  echo "Need at least ${MIN_MODELS} models, found ${#MODELS[@]} in ${MODELS_FILE}" >&2
  exit 1
fi

if ! is_true "$DRY_RUN"; then
  case "${AGENT_KIND,,}" in
    openai|qwen)
      if [[ -z "$OPENAI_API_KEY" ]]; then
        echo "OPENAI_API_KEY is required for AGENT_KIND=${AGENT_KIND}." >&2
        exit 1
      fi
      ;;
    openrouter)
      if [[ -z "$OPENROUTER_API_KEY" ]]; then
        echo "OPENROUTER_API_KEY is required for AGENT_KIND=openrouter." >&2
        exit 1
      fi
      ;;
  esac
fi

if [[ -x ".venv/bin/python" ]]; then
  PY=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python)
fi

mkdir -p "$OUT_ROOT"
printf "%s\n" "${MODELS[@]}" > "$OUT_ROOT/models.txt"
printf "model\tseed\n" > "$OUT_ROOT/model_seeds.tsv"

declare -a MODEL_SEEDS=()
for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  model_seed="$SEED"
  if is_true "$RANDOMIZE_SEED_PER_MODEL"; then
    model_seed="$(rand_seed)"
  fi
  MODEL_SEEDS[$i]="$model_seed"
  printf "%s\t%s\n" "$model" "$model_seed" >> "$OUT_ROOT/model_seeds.tsv"
done

echo "Hanabi API batch eval started"
echo "OUT_ROOT=$OUT_ROOT"
echo "MODELS_FILE=$MODELS_FILE"
echo "MODELS=${#MODELS[@]}"
echo "MIN_MODELS=$MIN_MODELS"
echo "EPISODES=$EPISODES"
echo "AGENT_KIND=$AGENT_KIND"
echo "ENV_ID=$ENV_ID"
echo "DRY_RUN=$DRY_RUN"
echo "PARALLEL_JOBS=$PARALLEL_JOBS"
echo "STREAM=$STREAM"

if [[ "$PARALLEL_JOBS" -gt 1 ]] && ! is_true "$CONTINUE_ON_ERROR"; then
  echo "WARN: PARALLEL_JOBS>1 with CONTINUE_ON_ERROR=0 may still run already queued models." >&2
fi

run_one_model() {
  local i="$1"
  local model="${MODELS[$i]}"
  local model_seed="${MODEL_SEEDS[$i]}"

  local idx_tag
  local model_tag
  local model_out
  idx_tag="$(printf '%03d' "$i")"
  model_tag="${idx_tag}_$(safe_name "$model")"
  model_out="$OUT_ROOT/$model_tag"
  mkdir -p "$model_out"
  printf "%s\n" "$model" > "$model_out/model.txt"

  local temperature_flag=()
  local top_p_flag=()
  local top_k_flag=()
  local max_tokens_flag=()
  local disable_flag=()
  local stream_flag=()
  local base_url_flag=()
  local api_key_flag=()
  local agent_gen_flags=()
  temperature_flag=()
  top_p_flag=()
  top_k_flag=()
  max_tokens_flag=()
  disable_flag=()
  stream_flag=()
  base_url_flag=()
  api_key_flag=()
  agent_gen_flags=()

  if [[ -n "$TEMPERATURE" && "$TEMPERATURE" != "null" && "$TEMPERATURE" != "None" ]]; then
    temperature_flag=(--temperature "$TEMPERATURE")
  fi
  if [[ -n "$TOP_P" && "$TOP_P" != "null" && "$TOP_P" != "None" ]]; then
    top_p_flag=(--top-p "$TOP_P")
  fi
  if [[ -n "$TOP_K" && "$TOP_K" != "null" && "$TOP_K" != "None" ]]; then
    top_k_flag=(--top-k "$TOP_K")
  fi
  if [[ -n "$MAX_TOKENS" && "$MAX_TOKENS" != "null" && "$MAX_TOKENS" != "None" ]]; then
    max_tokens_flag=(--max-tokens "$MAX_TOKENS")
  fi
  if is_true "$DISABLE_THINKING"; then
    disable_flag=(--disable-thinking)
  fi
  if is_true "$STREAM"; then
    stream_flag=(--stream)
  fi

  local model_gen_json=""
  model_gen_json=""
  if [[ -n "$MODEL_GEN_FILE" ]]; then
    model_gen_json="$("${PY[@]}" - "$MODEL_GEN_FILE" "$model" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
model = sys.argv[2]
try:
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception as e:
    raise SystemExit(f"Invalid MODEL_GEN_FILE JSON: {e}")

obj = None
if isinstance(data, dict):
    cand = data.get(model)
    if isinstance(cand, dict):
        obj = cand
elif isinstance(data, list):
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("model") != model:
            continue
        # Accept one of these shapes:
        # {"model": "...", "gen": {...}}
        # {"model": "...", "agent_gen": {...}}
        # {"model": "...", ...gen keys...}
        if isinstance(item.get("gen"), dict):
            obj = item["gen"]
        elif isinstance(item.get("agent_gen"), dict):
            obj = item["agent_gen"]
        else:
            obj = {k: v for k, v in item.items() if k != "model"}
        break
else:
    raise SystemExit("MODEL_GEN_FILE must be a JSON object or JSON array.")

if obj is None:
    print("")
elif not isinstance(obj, dict):
    raise SystemExit(f"Override for model={model!r} must be a JSON object.")
else:
    print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
PY
)"
    if [[ -n "$model_gen_json" ]]; then
      # One per player (this script uses self-play: same model on P0/P1).
      agent_gen_flags=(--agent-gen "$model_gen_json" --agent-gen "$model_gen_json")
    fi
  fi

  case "${AGENT_KIND,,}" in
    openai|qwen)
      if [[ -n "$OPENAI_BASE_URL" ]]; then
        base_url_flag=(--openai-base-url "$OPENAI_BASE_URL")
      fi
      if [[ -n "$OPENAI_API_KEY" ]]; then
        api_key_flag=(--openai-api-key "$OPENAI_API_KEY")
      fi
      ;;
  esac

  echo
  echo "===== $(date -Is) model=${model} seed=${model_seed} ====="

  if is_true "$DRY_RUN"; then
    echo "DRY_RUN: would run ${AGENT_KIND}:${model} for ${EPISODES} episodes."
    return 0
  fi

  local rc=0
  rc=0
  (
    "${PY[@]}" tools/rollout/run_rollouts.py \
      --env-id "$ENV_ID" \
      --num-players "$NUM_PLAYERS" \
      --episodes "$EPISODES" \
      --seed "$model_seed" \
      --agent "${AGENT_KIND}:${model}" \
      --agent "${AGENT_KIND}:${model}" \
      "${agent_gen_flags[@]}" \
      --timeout "$TIMEOUT" \
      --max-retries "$MAX_RETRIES" \
      --retry-initial-delay "$RETRY_INITIAL_DELAY" \
      --retry-max-delay "$RETRY_MAX_DELAY" \
      "${stream_flag[@]}" \
      "${temperature_flag[@]}" \
      "${top_p_flag[@]}" \
      "${top_k_flag[@]}" \
      "${max_tokens_flag[@]}" \
      "${disable_flag[@]}" \
      "${base_url_flag[@]}" \
      "${api_key_flag[@]}" \
      --episode-json-dir "$model_out/episodes" \
      --out "$model_out/rollouts.jsonl"
  ) > "$model_out/run.log" 2>&1 || rc=$?

  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi

  "${PY[@]}" tools/rollout/summarize_rollouts.py "$model_out/rollouts.jsonl" --json > "$model_out/summary.json"
}

failed_models=()

if [[ "$PARALLEL_JOBS" -le 1 ]]; then
  for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    rc=0
    run_one_model "$i" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "Model run failed: $model (exit=${rc})" >&2
      failed_models+=("$model")
      if ! is_true "$CONTINUE_ON_ERROR"; then
        echo "Stopping on first failure. Set CONTINUE_ON_ERROR=1 to continue." >&2
        break
      fi
    fi
  done
else
  status_dir="$OUT_ROOT/.status_codes"
  mkdir -p "$status_dir"
  rm -f "$status_dir"/*.code 2>/dev/null || true

  for i in "${!MODELS[@]}"; do
    code_file="$status_dir/$(printf '%03d' "$i").code"
    (
      set +e
      run_one_model "$i"
      rc=$?
      printf "%s\n" "$rc" > "$code_file"
      exit 0
    ) &

    while true; do
      running_jobs="$(jobs -pr | wc -l | tr -d ' ')"
      if [[ "$running_jobs" -lt "$PARALLEL_JOBS" ]]; then
        break
      fi
      sleep 0.2
    done
  done

  wait || true

  for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    code_file="$status_dir/$(printf '%03d' "$i").code"
    if [[ ! -f "$code_file" ]]; then
      echo "Model run failed: $model (missing status code)" >&2
      failed_models+=("$model")
      continue
    fi
    rc="$(cat "$code_file")"
    if ! [[ "$rc" =~ ^[0-9]+$ ]]; then
      echo "Model run failed: $model (invalid status code: $rc)" >&2
      failed_models+=("$model")
      continue
    fi
    if [[ "$rc" -ne 0 ]]; then
      echo "Model run failed: $model (exit=${rc})" >&2
      failed_models+=("$model")
    fi
  done
fi

"${PY[@]}" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1]).resolve()
rows = []
for model_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        continue
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    envs = data.get("envs") or {}
    if not envs:
        continue
    env_id = next(iter(envs))
    env = envs.get(env_id) or {}
    by_player = env.get("by_player") or {}
    p0 = by_player.get("0") or {}
    p1 = by_player.get("1") or {}
    model_id_path = model_dir / "model.txt"
    model_id = model_id_path.read_text(encoding="utf-8").strip() if model_id_path.exists() else model_dir.name

    rows.append(
        {
            "model": model_id,
            "run_dir": model_dir.name,
            "env_id": env_id,
            "episodes": env.get("episodes"),
            "avg_score_if_coop": env.get("avg_score_if_coop"),
            "perfect_score_rate_if_coop": env.get("perfect_score_rate_if_coop"),
            "p0_invalid_rate": p0.get("invalid_rate"),
            "p1_invalid_rate": p1.get("invalid_rate"),
        }
    )

def score_key(row: dict) -> float:
    v = row.get("avg_score_if_coop")
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0

rows.sort(key=lambda r: (-score_key(r), r.get("model", "")))

json_path = out_root / "leaderboard.json"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

header = [
    "model",
    "run_dir",
    "env_id",
    "episodes",
    "avg_score_if_coop",
    "perfect_score_rate_if_coop",
    "p0_invalid_rate",
    "p1_invalid_rate",
]
lines = ["\t".join(header)]
for row in rows:
    lines.append("\t".join("" if row.get(k) is None else str(row.get(k)) for k in header))

tsv_path = out_root / "leaderboard.tsv"
tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"leaderboard_json={json_path}")
print(f"leaderboard_tsv={tsv_path}")
PY

echo
echo "Hanabi API batch eval finished."
echo "Output root:  $OUT_ROOT"
echo "Model list:   $OUT_ROOT/models.txt"
echo "Seed map:     $OUT_ROOT/model_seeds.tsv"
echo "Leaderboard:  $OUT_ROOT/leaderboard.json"
echo "Leaderboard:  $OUT_ROOT/leaderboard.tsv"

if [[ "${#failed_models[@]}" -gt 0 ]]; then
  echo "Failed models: ${failed_models[*]}" >&2
  exit 1
fi
