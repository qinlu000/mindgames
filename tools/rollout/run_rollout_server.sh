#!/usr/bin/env bash
set -euo pipefail

# ms-swift rollout server for GRPO (external vLLM engine).
#
# Defaults (override via env vars):
#   MODEL=Qwen/Qwen3-8B
#   HOST=0.0.0.0
#   PORT=8001
#   VLLM_MAX_MODEL_LEN=8192
#   VLLM_MAX_NUM_SEQS=4
#   VLLM_GPU_MEMORY_UTILIZATION=0.80
#   VLLM_ENFORCE_EAGER=true
#   VLLM_ENABLE_LORA=true
#   VLLM_MAX_LORA_RANK=8
#   CUDA_VISIBLE_DEVICES=1
#   NCCL_P2P_DISABLE=1
#   NCCL_IB_DISABLE=1

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
.venv/bin/swift rollout \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_enforce_eager "$VLLM_ENFORCE_EAGER" \
  --vllm_enable_lora "$VLLM_ENABLE_LORA" \
  --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK"
