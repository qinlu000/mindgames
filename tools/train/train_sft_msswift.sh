#!/usr/bin/env bash
set -euo pipefail

# ms-swift LoRA/QLoRA SFT (Qwen3-8B + Hi-ToM by default).
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   DATASET=Hi-ToM/Hi-ToM_Dataset
#   TRAIN_TYPE=lora   # or qlora
#   OUTPUT_DIR=output/qwen3-8b-hitom
#   CUDA_VISIBLE_DEVICES=0
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1
#
# Example:
#   TRAIN_TYPE=qlora CUDA_VISIBLE_DEVICES=1 \
#     bash tools/train/train_sft_msswift.sh

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-Hi-ToM/Hi-ToM_Dataset}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-8b-hitom}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if command -v uv >/dev/null 2>&1; then
  SWIFT_CMD=(uv run swift)
elif [ -x ".venv/bin/swift" ]; then
  SWIFT_CMD=(.venv/bin/swift)
elif command -v swift >/dev/null 2>&1; then
  SWIFT_CMD=(swift)
else
  echo "swift not found. Install ms-swift or run: uv add \"ms-swift[all]\"" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
"${SWIFT_CMD[@]}" sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --train_type "$TRAIN_TYPE" \
  --output_dir "$OUTPUT_DIR"
