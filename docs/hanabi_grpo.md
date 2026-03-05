# Hanabi GRPO (single-node, multi-GPU)

This project runs GRPO with ms-swift and a vLLM rollout server. For Hanabi,
the reward is computed by the gym env, so `REWARD_FUNCS` is left empty.

## Prereqs
- Install deps (one-time):
  - `uv sync --extra serve` (or `uv add "ms-swift[all]"`)
- Ensure `data/hanabi.grpo.jsonl` exists (used to pass `env_config` to the gym env).

## Quick Start (auto split)
Run these two commands in two terminals. By default:
- rollout script uses the first half of GPUs
- train script uses the second half of GPUs
- if `/workspace/models/Qwen3-8B` exists, both scripts use it directly

Terminal 1 (rollout server):
```bash
bash tools/rollout/rollout_hanabi_gym.sh
```

Terminal 2 (GRPO training):
```bash
bash tools/train/train_grpo_hanabi_server_wandb.sh
```

## 8x H800 80GB recommended profile
Use a 4+4 split and disable tensor-parallel rollout (`TP=1`) for stability.

Terminal 1 (4 rollout servers, one GPU each):
```bash
for i in 0 1 2 3; do
  port=$((8000 + i))
  CUDA_VISIBLE_DEVICES=$i \
  HOST=127.0.0.1 PORT=$port \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  VLLM_DATA_PARALLEL_SIZE=1 \
  VLLM_MAX_MODEL_LEN=18000 \
  VLLM_MAX_NUM_SEQS=16 \
  NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
  bash tools/rollout/rollout_hanabi_gym_simple.sh &
done
```

Terminal 2:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
DATASET=data/hanabi.grpo.jsonl \
VLLM_SERVER_HOST=127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
VLLM_SERVER_PORT=8000,8001,8002,8003 \
VLLM_SERVER_GROUP_PORT=51216,51217,51218,51219 \
NUM_GENERATIONS=16 \
GENERATION_BATCH_SIZE=64 \
MAX_LENGTH=18000 \
MAX_COMPLETION_LENGTH=18000 \
MAX_STEPS=1000 \
NCCL_P2P_DISABLE=0 \
NCCL_IB_DISABLE=0 \
bash tools/train/train_grpo_hanabi_server_simple.sh
```

Recommended group config on 4 train GPUs:
- `NUM_GENERATIONS=16`
- `GENERATION_BATCH_SIZE=64` (must be divisible by both `NPROC_PER_NODE` and `NUM_GENERATIONS`)

To only inspect auto-resolved settings without launching jobs:
```bash
DRY_RUN=true bash tools/rollout/rollout_hanabi_gym.sh
DRY_RUN=true bash tools/train/train_grpo_hanabi_server_wandb.sh
```

Advanced: if you need full manual arg control, use `tools/train/train_grpo_msswift.sh` with explicit env vars.

The W&B wrapper now does two things by default:
- uploads training metrics/logs to W&B (`REPORT_TO=wandb`)
- uploads training output/checkpoints as a W&B `model` artifact after training

Useful env vars for the wrapper:
- `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE` / `WANDB_NAME`
- `WANDB_LOG_MODEL=checkpoint` (keep checkpoint logging enabled)
- `UPLOAD_CKPT_TO_WANDB=true` (set `false` to skip artifact upload)
- `CKPT_ARTIFACT_NAME` / `CKPT_ARTIFACT_ALIASES` (default aliases: `latest,end`)

## Adjusting the split
- Rollout server: set `CUDA_VISIBLE_DEVICES` to the GPUs it owns.
  For Qwen3-8B on H800, prefer `VLLM_TENSOR_PARALLEL_SIZE=1` and scale with more servers.
- Training: set `CUDA_VISIBLE_DEVICES` to the remaining GPUs and
  `NPROC_PER_NODE` to that count.

## Key environment variables
- `DATASET`: must point at `data/hanabi.grpo.jsonl` for the gym env workflow.
- `VLLM_MODE=server`: tells ms-swift to call the rollout server.
- `VLLM_SERVER_HOST`/`VLLM_SERVER_PORT`: must match the rollout server.
- `REWARD_FUNCS=`: empty for Hanabi gym rewards (do not set a reward model).
- `NUM_TRAIN_EPOCHS`/`MAX_STEPS`: control training length (epochs or optimizer steps).
  If both are set, `MAX_STEPS` wins. Defaults to `MAX_STEPS=500`. With the default Hanabi dataset (1 row), each epoch is ~1 optimizer step.
- `UPLOAD_CKPT_TO_WANDB`: upload `OUTPUT_DIR` to W&B as a `model` artifact after training (default `true` in the W&B wrapper).

## Troubleshooting
- "Address already in use": change `PORT` in `tools/rollout/rollout_hanabi_gym.sh`
  and update `VLLM_SERVER_PORT` for training.
- NCCL issues: for H800, keep `NCCL_P2P_DISABLE=0` and `NCCL_IB_DISABLE=0`.
