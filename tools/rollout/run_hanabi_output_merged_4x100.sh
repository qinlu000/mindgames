#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate local merged Hanabi models under output/merged.
#
# Default behavior:
# - Enumerate 4 model dirs under output/merged (strict count check enabled)
# - Run 100 Hanabi episodes per model
# - Reuse existing single-server parallel runner per model
# - Emit per-model rollouts/summary and a top-level leaderboard
#
# Usage:
#   bash tools/rollout/run_hanabi_output_merged_4x100.sh
#
# Common overrides:
#   MERGED_ROOT=output/merged
#   OUT_ROOT=outputs/hanabi_merged_4x100_$(date +%Y%m%d_%H%M%S)
#   EPISODES=100
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   WORKERS=25
#   PORT=8000
#   CONTINUE_ON_ERROR=1
#   STRICT_FOUR=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MERGED_ROOT="${MERGED_ROOT:-output/merged}"
OUT_ROOT="${OUT_ROOT:-outputs/hanabi_merged_4x100_$(date +%Y%m%d_%H%M%S)}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-tools/rollout/run_hanabi_qwen3_8b_vllm_1server_parallel.sh}"

ENV_ID="${ENV_ID:-Hanabi-v0-train}"
NUM_PLAYERS="${NUM_PLAYERS:-2}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
# SEED:
# - integer: fixed base seed
# - random/auto: pick a random base seed at runtime
# RANDOMIZE_SEED_PER_MODEL:
# - 0: all models share the same base seed (fair comparison)
# - 1: each model gets an independent random base seed
RANDOMIZE_SEED_PER_MODEL="${RANDOMIZE_SEED_PER_MODEL:-0}"

# auto: model dir name contains "no-think" -> 1, otherwise 0
DISABLE_THINKING_DEFAULT="${DISABLE_THINKING_DEFAULT:-auto}"
# REASONING_PARSER:
# - If REASONING_PARSER is explicitly set, all models use that value.
# - Otherwise, REASONING_PARSER_DEFAULT=auto enables parser "qwen3" for think models
#   (disable_thinking=0), and leaves it unset for no-think models.
REASONING_PARSER_DEFAULT="${REASONING_PARSER_DEFAULT:-auto}"

STRICT_FOUR="${STRICT_FOUR:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$MERGED_ROOT" ]]; then
  echo "MERGED_ROOT not found: $MERGED_ROOT" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "RUNNER_SCRIPT not found: $RUNNER_SCRIPT" >&2
  exit 1
fi

mapfile -t MODEL_DIRS < <(find "$MERGED_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)
if [[ "${#MODEL_DIRS[@]}" -eq 0 ]]; then
  echo "No model directories found under $MERGED_ROOT" >&2
  exit 1
fi

if [[ "$STRICT_FOUR" == "1" || "${STRICT_FOUR,,}" == "true" ]]; then
  if [[ "${#MODEL_DIRS[@]}" -ne 4 ]]; then
    echo "Expected exactly 4 models under $MERGED_ROOT, got ${#MODEL_DIRS[@]}." >&2
    echo "Set STRICT_FOUR=0 to run with current count." >&2
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT"
printf "%s\n" "${MODEL_DIRS[@]}" > "$OUT_ROOT/models.txt"

rand_seed() {
  echo $(( (RANDOM << 16) | RANDOM ))
}

BASE_SEED=""
seed_lc="${SEED,,}"
if [[ "$seed_lc" == "random" || "$seed_lc" == "auto" ]]; then
  BASE_SEED="$(rand_seed)"
elif [[ "$SEED" =~ ^[0-9]+$ ]]; then
  BASE_SEED="$SEED"
else
  echo "Invalid SEED=$SEED (use integer or random/auto)" >&2
  exit 1
fi

echo "Batch Hanabi eval started"
echo "MERGED_ROOT=$MERGED_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "EPISODES=$EPISODES"
echo "SEED_INPUT=$SEED"
echo "BASE_SEED=$BASE_SEED"
echo "RANDOMIZE_SEED_PER_MODEL=$RANDOMIZE_SEED_PER_MODEL"
echo "RUNNER_SCRIPT=$RUNNER_SCRIPT"
echo "MODELS=${#MODEL_DIRS[@]}"
echo "DRY_RUN=$DRY_RUN"

seed_map="$OUT_ROOT/model_seeds.tsv"
echo -e "model\tseed" > "$seed_map"

failed_models=()
model_idx=0

for model_dir in "${MODEL_DIRS[@]}"; do
  model_name="$(basename "$model_dir")"
  model_out="$OUT_ROOT/$model_name"
  mkdir -p "$model_out"

  model_seed="$BASE_SEED"
  if [[ "$RANDOMIZE_SEED_PER_MODEL" == "1" || "${RANDOMIZE_SEED_PER_MODEL,,}" == "true" ]]; then
    model_seed="$(rand_seed)"
  fi
  echo -e "${model_name}\t${model_seed}" >> "$seed_map"

  disable_thinking="${DISABLE_THINKING:-}"
  if [[ -z "$disable_thinking" ]]; then
    if [[ "$DISABLE_THINKING_DEFAULT" == "auto" ]]; then
      model_lc="${model_name,,}"
      if [[ "$model_lc" == *"no-think"* ]]; then
        disable_thinking="1"
      else
        disable_thinking="0"
      fi
    else
      disable_thinking="$DISABLE_THINKING_DEFAULT"
    fi
  fi

  model_reasoning_parser="${REASONING_PARSER:-}"
  if [[ -z "$model_reasoning_parser" ]]; then
    rp_def_lc="${REASONING_PARSER_DEFAULT,,}"
    if [[ "$rp_def_lc" == "auto" ]]; then
      if [[ "$disable_thinking" == "1" || "${disable_thinking,,}" == "true" ]]; then
        model_reasoning_parser=""
      else
        model_reasoning_parser="qwen3"
      fi
    elif [[ "$rp_def_lc" == "none" || "$rp_def_lc" == "off" || "$rp_def_lc" == "false" ]]; then
      model_reasoning_parser=""
    else
      model_reasoning_parser="$REASONING_PARSER_DEFAULT"
    fi
  fi

  echo
  echo "===== $(date -Is) model=${model_name} seed=${model_seed} disable_thinking=${disable_thinking} reasoning_parser=${model_reasoning_parser:-<unset>} ====="

  if [[ "$DRY_RUN" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    echo "DRY_RUN: would run MODEL=$model_dir OUT_DIR=$model_out EPISODES=$EPISODES SEED=$model_seed REASONING_PARSER=${model_reasoning_parser:-<unset>}"
    model_idx=$((model_idx + 1))
    continue
  fi

  rc=0
  (
    OUT_DIR="$model_out" \
    MODEL="$model_dir" \
    ENV_ID="$ENV_ID" \
    NUM_PLAYERS="$NUM_PLAYERS" \
    EPISODES="$EPISODES" \
    SEED="$model_seed" \
    DISABLE_THINKING="$disable_thinking" \
    REASONING_PARSER="$model_reasoning_parser" \
    bash "$RUNNER_SCRIPT"
  ) 2>&1 | tee "$model_out/batch.log" || rc=$?

  if [[ "$rc" -ne 0 ]]; then
    echo "Model run failed: ${model_name} (exit=${rc})" >&2
    failed_models+=("$model_name")
    if [[ "$CONTINUE_ON_ERROR" != "1" && "${CONTINUE_ON_ERROR,,}" != "true" ]]; then
      echo "Stopping on first failure. Set CONTINUE_ON_ERROR=1 to keep going." >&2
      break
    fi
  fi

  model_idx=$((model_idx + 1))
done

if [[ -x ".venv/bin/python" ]]; then
  PY=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python)
fi

"${PY[@]}" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows = []
for model_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
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
    rows.append(
        {
            "model": model_dir.name,
            "env_id": env_id,
            "episodes": env.get("episodes"),
            "avg_score_if_coop": env.get("avg_score_if_coop"),
            "perfect_score_rate_if_coop": env.get("perfect_score_rate_if_coop"),
            "p0_invalid_rate": p0.get("invalid_rate"),
            "p1_invalid_rate": p1.get("invalid_rate"),
        }
    )

def _score_key(row):
    s = row.get("avg_score_if_coop")
    if isinstance(s, (int, float)):
        return float(s)
    return -1.0

rows.sort(key=lambda r: (-_score_key(r), r.get("model", "")))

json_path = out_root / "leaderboard.json"
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

header = [
    "model",
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
echo "Batch Hanabi eval finished."
echo "Output root: $OUT_ROOT"
echo "Model list:  $OUT_ROOT/models.txt"
echo "Seed map:    $OUT_ROOT/model_seeds.tsv"
echo "Leaderboard: $OUT_ROOT/leaderboard.json"
echo "Leaderboard: $OUT_ROOT/leaderboard.tsv"

if [[ "${#failed_models[@]}" -gt 0 ]]; then
  echo "Failed models: ${failed_models[*]}" >&2
  exit 1
fi
