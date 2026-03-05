#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: Hanabi GRPO training + external vLLM rollout server.
# Start a rollout server separately before running this script.
#
# Defaults (override via env vars):
#   MODEL=/workspace/models/Qwen3-8B (if exists), otherwise Qwen/Qwen3-8B
#   CUDA_VISIBLE_DEVICES=<auto: second half of visible GPUs, e.g. 5-9 on 10 GPUs>
#   NPROC_PER_NODE=<auto: equals number of train GPUs>
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1
#   DATASET=data/hanabi.grpo.jsonl
#   OUTPUT_DIR=output/qwen3-8b-hanabi-grpo
#   NUM_GENERATIONS=<auto: 2 * NPROC_PER_NODE for heavy-thinking setup>
#   GENERATION_BATCH_SIZE=32
#   MAX_LENGTH=16384
#   MAX_COMPLETION_LENGTH=16384
#   NUM_TRAIN_EPOCHS=
#   MAX_STEPS=500
#   VLLM_SERVER_HOST=127.0.0.1
#   VLLM_SERVER_PORT=8000
#   VLLM_SERVER_GROUP_PORT=        # optional, auto-pick if empty. Supports comma-separated lists.
#   REPORT_TO=wandb
#   RUN_NAME=grpo-hanabi
#   WANDB_PROJECT=mindgames
#   WANDB_API_KEY=你的key
#   WANDB_ENTITY=
#   WANDB_MODE=online
#   WANDB_LOG_MODEL=checkpoint
#   WANDB_WATCH=false
#   WANDB_NAME=$RUN_NAME
#   UPLOAD_CKPT_TO_WANDB=true
#   CKPT_ARTIFACT_NAME=${RUN_NAME}-ckpt
#   CKPT_ARTIFACT_ALIASES=latest,end
#   HF_TOKEN=
#   HF_REPO_ID=
#   HUB_STRATEGY=end
#   HUB_PRIVATE_REPO=false
#   PUSH_TO_HUB=false
#   USE_HF=false
#   DRY_RUN=false

_count_csv_items() {
  local raw="$1"
  if [ -z "$raw" ]; then
    echo 0
    return
  fi
  local item
  local count=0
  IFS=',' read -r -a _items <<< "$raw"
  for item in "${_items[@]}"; do
    if [ -n "${item//[[:space:]]/}" ]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

_gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
}

_build_range_csv() {
  local start="$1"
  local end="$2"
  local values=()
  local i
  for ((i = start; i <= end; i++)); do
    values+=("$i")
  done
  local out
  out="$(IFS=,; echo "${values[*]}")"
  echo "$out"
}

_is_local_host() {
  local host="$1"
  [ "$host" = "127.0.0.1" ] || [ "$host" = "localhost" ] || [ "$host" = "::1" ]
}

_all_local_hosts_csv() {
  local raw="$1"
  local item
  if [ -z "$raw" ]; then
    return 1
  fi
  IFS=',' read -r -a _items <<< "$raw"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ -z "$item" ]; then
      continue
    fi
    if ! _is_local_host "$item"; then
      return 1
    fi
  done
  return 0
}

_has_local_host_csv() {
  local raw="$1"
  local item
  if [ -z "$raw" ]; then
    return 1
  fi
  IFS=',' read -r -a _items <<< "$raw"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [ -z "$item" ]; then
      continue
    fi
    if _is_local_host "$item"; then
      return 0
    fi
  done
  return 1
}

_pick_free_port() {
  local start="${1:-51216}"
  local end="${2:-51350}"
  python - "$start" "$end" <<'PY'
import socket
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
for port in range(start, end + 1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        pass
    else:
        print(port)
        s.close()
        raise SystemExit(0)
    s.close()
raise SystemExit(1)
PY
}

_pick_free_ports_csv() {
  local count="${1:-1}"
  local start="${2:-51216}"
  local end="${3:-51350}"
  local ports=()
  local next_start="$start"
  local picked

  while [ "${#ports[@]}" -lt "$count" ]; do
    if ! picked="$(_pick_free_port "$next_start" "$end")"; then
      return 1
    fi
    ports+=("$picked")
    next_start=$((picked + 1))
  done

  local out
  out="$(IFS=,; echo "${ports[*]}")"
  echo "$out"
}

if [ -z "${MODEL:-}" ]; then
  if [ -d "/workspace/models/Qwen3-8B" ]; then
    MODEL="/workspace/models/Qwen3-8B"
  else
    MODEL="Qwen/Qwen3-8B"
  fi
else
  MODEL="${MODEL}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  total_gpus="$(_gpu_count)"
  if [ "$total_gpus" -lt 1 ]; then
    echo "No GPUs detected. Set CUDA_VISIBLE_DEVICES explicitly." >&2
    exit 1
  fi
  # Default split: rollout uses first half, train uses second half.
  train_start=$((total_gpus / 2))
  train_end=$((total_gpus - 1))
  if [ "$train_start" -gt "$train_end" ]; then
    train_start=0
    train_end=0
  fi
  CUDA_VISIBLE_DEVICES="$(_build_range_csv "$train_start" "$train_end")"
fi

detected_train_gpus="$(_count_csv_items "$CUDA_VISIBLE_DEVICES")"
if [ "$detected_train_gpus" -lt 1 ]; then
  echo "CUDA_VISIBLE_DEVICES resolved to zero GPUs: '$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$detected_train_gpus}"
if [ "$NPROC_PER_NODE" -lt 1 ]; then
  echo "NPROC_PER_NODE must be >= 1, got $NPROC_PER_NODE" >&2
  exit 1
fi

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
DATASET="${DATASET:-data/hanabi.grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hanabi-grpo}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-16384}"
if [ -z "${NUM_GENERATIONS:-}" ]; then
  # Heuristic: with longer completion lengths, reduce group size to stabilize throughput.
  if [ "$MAX_COMPLETION_LENGTH" -ge 16384 ]; then
    NUM_GENERATIONS="$((NPROC_PER_NODE * 2))"
  elif [ "$MAX_COMPLETION_LENGTH" -ge 8192 ]; then
    NUM_GENERATIONS="$((NPROC_PER_NODE * 3))"
  elif [ "$MAX_COMPLETION_LENGTH" -ge 4096 ]; then
    NUM_GENERATIONS="$((NPROC_PER_NODE * 4))"
  else
    NUM_GENERATIONS="$((NPROC_PER_NODE * 5))"
  fi
else
  NUM_GENERATIONS="${NUM_GENERATIONS}"
fi
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-500}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_GROUP_PORT="${VLLM_SERVER_GROUP_PORT:-}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-grpo-hanabi}"
WANDB_PROJECT="${WANDB_PROJECT:-mindgames}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-checkpoint}"
WANDB_WATCH="${WANDB_WATCH:-false}"
WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
UPLOAD_CKPT_TO_WANDB="${UPLOAD_CKPT_TO_WANDB:-true}"
CKPT_ARTIFACT_NAME="${CKPT_ARTIFACT_NAME:-${RUN_NAME}-ckpt}"
CKPT_ARTIFACT_ALIASES="${CKPT_ARTIFACT_ALIASES:-latest,end}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HUB_STRATEGY="${HUB_STRATEGY:-end}"
HUB_PRIVATE_REPO="${HUB_PRIVATE_REPO:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
USE_HF="${USE_HF:-false}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
DRY_RUN="${DRY_RUN:-false}"

if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY
fi

if [ "$REPORT_TO" = "wandb" ] && [ "$WANDB_MODE" = "online" ] && [ -z "$WANDB_API_KEY" ]; then
  echo "WARN: WANDB_API_KEY is empty while REPORT_TO=wandb and WANDB_MODE=online. Switching to WANDB_MODE=offline." >&2
  WANDB_MODE="offline"
fi

if [ "$PUSH_TO_HUB" = "true" ]; then
  if [ -z "$HF_TOKEN" ]; then
    echo "PUSH_TO_HUB=true requires HF_TOKEN." >&2
    exit 1
  fi
  if [ -z "$HF_REPO_ID" ]; then
    echo "PUSH_TO_HUB=true requires HF_REPO_ID (e.g. <user_or_org>/qwen3-8b-hanabi-grpo)." >&2
    exit 1
  fi
fi

if [ -z "$VLLM_SERVER_GROUP_PORT" ]; then
  host_count="$(_count_csv_items "$VLLM_SERVER_HOST")"
  if [ "$host_count" -lt 1 ]; then
    host_count=1
  fi
  if _all_local_hosts_csv "$VLLM_SERVER_HOST"; then
    if [ "$host_count" -eq 1 ]; then
      if picked_port="$(_pick_free_port 51216 51350)"; then
        VLLM_SERVER_GROUP_PORT="$picked_port"
      fi
    elif picked_ports="$(_pick_free_ports_csv "$host_count" 51216 51350)"; then
      VLLM_SERVER_GROUP_PORT="$picked_ports"
    fi
  fi
fi

if _has_local_host_csv "$VLLM_SERVER_HOST"; then
  NO_PROXY="${NO_PROXY:-}"
  no_proxy="${no_proxy:-}"
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
  export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
fi

echo "[hanabi-train] model=$MODEL server=${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT} group_port=${VLLM_SERVER_GROUP_PORT:-auto} cuda=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE num_generations=$NUM_GENERATIONS gen_batch=$GENERATION_BATCH_SIZE max_length=$MAX_LENGTH max_completion_length=$MAX_COMPLETION_LENGTH"

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

REPORT_TO="$REPORT_TO" RUN_NAME="$RUN_NAME" WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" WANDB_MODE="$WANDB_MODE" \
WANDB_LOG_MODEL="$WANDB_LOG_MODEL" WANDB_WATCH="$WANDB_WATCH" WANDB_NAME="$WANDB_NAME" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
MODEL="$MODEL" \
VLLM_MODE=server \
VLLM_SERVER_HOST="$VLLM_SERVER_HOST" VLLM_SERVER_PORT="$VLLM_SERVER_PORT" \
VLLM_SERVER_GROUP_PORT="$VLLM_SERVER_GROUP_PORT" \
DATASET="$DATASET" OUTPUT_DIR="$OUTPUT_DIR" \
NUM_GENERATIONS="$NUM_GENERATIONS" GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
MAX_LENGTH="$MAX_LENGTH" MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" MAX_STEPS="$MAX_STEPS" \
PUSH_TO_HUB="$PUSH_TO_HUB" USE_HF="$USE_HF" HUB_TOKEN="$HF_TOKEN" HUB_MODEL_ID="$HF_REPO_ID" \
HUB_STRATEGY="$HUB_STRATEGY" HUB_PRIVATE_REPO="$HUB_PRIVATE_REPO" \
REWARD_FUNCS= EXTERNAL_PLUGINS= \
bash tools/train/train_grpo_msswift.sh

if [ "$REPORT_TO" = "wandb" ] && [ "$UPLOAD_CKPT_TO_WANDB" = "true" ]; then
  if command -v uv >/dev/null 2>&1; then
    PY_CMD=(uv run python)
  elif [ -x ".venv/bin/python" ]; then
    PY_CMD=(.venv/bin/python)
  else
    PY_CMD=(python)
  fi

  OUTPUT_DIR="$OUTPUT_DIR" \
  WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
  RUN_NAME="$RUN_NAME" CKPT_ARTIFACT_NAME="$CKPT_ARTIFACT_NAME" \
  CKPT_ARTIFACT_ALIASES="$CKPT_ARTIFACT_ALIASES" \
  "${PY_CMD[@]}" - <<'PY'
import os
import sys
from pathlib import Path

out_dir = Path(os.environ["OUTPUT_DIR"])
if not out_dir.exists():
    print(f"WARN: OUTPUT_DIR not found, skip W&B ckpt upload: {out_dir}", file=sys.stderr)
    sys.exit(0)

try:
    import wandb
except Exception as exc:
    print(f"WARN: wandb unavailable, skip ckpt upload: {exc}", file=sys.stderr)
    sys.exit(0)

project = os.environ.get("WANDB_PROJECT", "mindgames")
entity = os.environ.get("WANDB_ENTITY") or None
run_name = os.environ.get("RUN_NAME", "grpo-hanabi")
artifact_name = os.environ.get("CKPT_ARTIFACT_NAME") or f"{run_name}-ckpt"
aliases = [x.strip() for x in os.environ.get("CKPT_ARTIFACT_ALIASES", "latest,end").split(",") if x.strip()]

with wandb.init(project=project, entity=entity, job_type="checkpoint_upload", name=f"{run_name}-ckpt-upload") as run:
    art = wandb.Artifact(name=artifact_name, type="model", metadata={"output_dir": str(out_dir)})
    art.add_dir(str(out_dir))
    run.log_artifact(art, aliases=aliases)
    print(f"Uploaded W&B artifact: {artifact_name} aliases={aliases}")
PY
fi
