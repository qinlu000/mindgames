#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: Hanabi GRPO training on GPUs 4-7 + external vLLM rollout server + W&B.
# Start a vLLM rollout server separately (e.g., on GPUs 0-3) before running this script.
#
# Defaults (override via env vars):
#   CUDA_VISIBLE_DEVICES=4,5,6,7
#   NPROC_PER_NODE=4
#   NCCL_P2P_DISABLE=0
#   NCCL_IB_DISABLE=0
#   DATASET=data/hanabi.grpo.jsonl
#   OUTPUT_DIR=output/qwen3-8b-hanabi-grpo
#   NUM_GENERATIONS=16
#   GENERATION_BATCH_SIZE=16
#   NUM_TRAIN_EPOCHS=
#   MAX_STEPS=500
#   VLLM_SERVER_HOST=127.0.0.1
#   VLLM_SERVER_PORT=8000
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
#   HF_TOKEN=hf_xxx
#   HF_REPO_ID=csminion/qwen3-8b-hanabi-grpo
#   HUB_STRATEGY=end
#   HUB_PRIVATE_REPO=false

export HF_TOKEN="${HF_TOKEN:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
DATASET="${DATASET:-data/hanabi.grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hanabi-grpo}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$NUM_GENERATIONS}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-500}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
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
HF_REPO_ID="${HF_REPO_ID:-csminion/qwen3-8b-hanabi-grpo}"
HUB_STRATEGY="${HUB_STRATEGY:-end}"
HUB_PRIVATE_REPO="${HUB_PRIVATE_REPO:-false}"

if [ -n "$WANDB_API_KEY" ] && [ "$WANDB_API_KEY" != "你的key" ]; then
  export WANDB_API_KEY
fi

if [ "$HF_TOKEN" = "hf_xxx" ]; then
  echo "Please set HF_TOKEN to a real Hugging Face token." >&2
  exit 1
fi

if [ "$HF_REPO_ID" = "你的用户名/qwen3-8b-hanabi-grpo" ]; then
  echo "Please set HF_REPO_ID, e.g. <hf_user_or_org>/qwen3-8b-hanabi-grpo." >&2
  exit 1
fi

REPORT_TO="$REPORT_TO" RUN_NAME="$RUN_NAME" WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" WANDB_MODE="$WANDB_MODE" \
WANDB_LOG_MODEL="$WANDB_LOG_MODEL" WANDB_WATCH="$WANDB_WATCH" WANDB_NAME="$WANDB_NAME" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
VLLM_MODE=server \
VLLM_SERVER_HOST="$VLLM_SERVER_HOST" VLLM_SERVER_PORT="$VLLM_SERVER_PORT" \
DATASET="$DATASET" OUTPUT_DIR="$OUTPUT_DIR" \
NUM_GENERATIONS="$NUM_GENERATIONS" GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS" MAX_STEPS="$MAX_STEPS" \
PUSH_TO_HUB=true USE_HF=true HUB_TOKEN="$HF_TOKEN" HUB_MODEL_ID="$HF_REPO_ID" \
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
