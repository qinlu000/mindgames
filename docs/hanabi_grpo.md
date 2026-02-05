# Hanabi GRPO (single-node, multi-GPU)

This project runs GRPO with ms-swift and a vLLM rollout server. For Hanabi,
the reward is computed by the gym env, so `REWARD_FUNCS` is left empty.

## Prereqs
- Install deps (one-time):
  - `uv sync --extra serve` (or `uv add "ms-swift[all]"`)
- Ensure `data/hanabi.grpo.jsonl` exists (used to pass `env_config` to the gym env).

## 4+4 GPU split example (8x H800, group size 16)
Run the rollout server and training in two terminals.

Terminal 1 (rollout server on GPUs 0-3):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_TENSOR_PARALLEL_SIZE=4 \
VLLM_MAX_NUM_SEQS=16 \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
bash tools/rollout/rollout_hanabi_gym.sh
```

Terminal 2 (GRPO training on GPUs 4-7):
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
DATASET=data/hanabi.grpo.jsonl VLLM_MODE=server \
VLLM_SERVER_HOST=127.0.0.1 VLLM_SERVER_PORT=8000 \
NUM_GENERATIONS=16 GENERATION_BATCH_SIZE=16 \
REWARD_FUNCS= EXTERNAL_PLUGINS= \
bash tools/train/train_grpo_msswift.sh
```

Optional W&B wrapper (still requires the rollout server above):
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
bash tools/train/train_grpo_hanabi_server_wandb.sh
```

## Adjusting the split
- Rollout server: set `CUDA_VISIBLE_DEVICES` to the GPUs it owns, and set
  `VLLM_TENSOR_PARALLEL_SIZE` to the same count.
- Training: set `CUDA_VISIBLE_DEVICES` to the remaining GPUs and
  `NPROC_PER_NODE` to that count.

## Key environment variables
- `DATASET`: must point at `data/hanabi.grpo.jsonl` for the gym env workflow.
- `VLLM_MODE=server`: tells ms-swift to call the rollout server.
- `VLLM_SERVER_HOST`/`VLLM_SERVER_PORT`: must match the rollout server.
- `REWARD_FUNCS=`: empty for Hanabi gym rewards (do not set a reward model).
- `NUM_TRAIN_EPOCHS`/`MAX_STEPS`: control training length (epochs or optimizer steps).
  If both are set, `MAX_STEPS` wins. Defaults to `MAX_STEPS=500`. With the default Hanabi dataset (1 row), each epoch is ~1 optimizer step.

## Troubleshooting
- "Address already in use": change `PORT` in `tools/rollout/rollout_hanabi_gym.sh`
  and update `VLLM_SERVER_PORT` for training.
- NCCL issues: for H800, keep `NCCL_P2P_DISABLE=0` and `NCCL_IB_DISABLE=0`.
