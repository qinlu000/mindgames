# Hanabi GRPO (single-node, multi-GPU)

This project runs GRPO with ms-swift and a vLLM rollout server. For Hanabi,
the reward is computed by the gym env, so `REWARD_FUNCS` is left empty.
In the default Hanabi multi-turn rollout path, credit assignment now uses
score-gain rewards: when a move increases the cooperative score, that turn
receives positive reward equal to the score delta instead of waiting for only
the terminal episode score.

## Prereqs
- Install deps (one-time):
  - `uv sync --extra train --extra serve`
  - If you only need the missing package for an existing env, run `uv add deepspeed --optional train`
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
  CONTEXT_MANAGER=hanabi_recent_turns \
  HANABI_CTX_MAX_TURNS=1 \
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
MAX_LENGTH=4096 \
MAX_COMPLETION_LENGTH=64 \
MAX_STEPS=1000 \
NCCL_P2P_DISABLE=0 \
NCCL_IB_DISABLE=0 \
bash tools/train/train_grpo_hanabi_server_simple.sh
```

Recommended group config on 4 train GPUs:
- `NUM_GENERATIONS=16`
- `GENERATION_BATCH_SIZE=64` (must be divisible by both `NPROC_PER_NODE` and `NUM_GENERATIONS`)
- `STEPS_PER_GENERATION` and `GENERATION_BATCH_SIZE` are mutually exclusive
- For Hanabi, start with shorter generations: `MAX_LENGTH=4096`, `MAX_COMPLETION_LENGTH=64`

To only inspect auto-resolved settings without launching jobs:
```bash
DRY_RUN=true bash tools/rollout/rollout_hanabi_gym.sh
DRY_RUN=true bash tools/train/train_grpo_hanabi_server_wandb.sh
```

Advanced: if you need full manual arg control, use `tools/train/train_grpo_base.sh` with explicit env vars.

The W&B wrapper sets W&B logging defaults (`REPORT_TO=wandb`) and focuses on a lean launch flow.

Useful env vars for the wrapper:
- `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE` / `WANDB_NAME`
- `RUN_NAME`
- `DRY_RUN=true` (only print resolved launch args)

## Adjusting the split
- Rollout server: set `CUDA_VISIBLE_DEVICES` to the GPUs it owns.
  For Qwen3-8B on H800, prefer `VLLM_TENSOR_PARALLEL_SIZE=1` and scale with more servers.
- Training: set `CUDA_VISIBLE_DEVICES` to the remaining GPUs and
  `NPROC_PER_NODE` to that count.

## Key environment variables
- `DATASET`: must point at `data/hanabi.grpo.jsonl` for the gym env workflow.
- `VLLM_MODE=server`: tells ms-swift to call the rollout server.
- `VLLM_SERVER_HOST`/`VLLM_SERVER_PORT`: must match the rollout server.
- `VLLM_SERVER_TIMEOUT`: timeout for rollout server RPC (useful on slower networks or heavy rollout loads).
- `CONTEXT_MANAGER`: rollout-side context manager. Default is `hanabi_recent_turns`.
- `HANABI_CTX_MAX_TURNS`: how many recent user turns the rollout server keeps in context (default `1`).
- `REWARD_FUNCS=`: empty for Hanabi gym rewards (do not set a reward model).
- `GENERATION_BATCH_SIZE`: if set, must be divisible by `NUM_GENERATIONS` and `NPROC_PER_NODE`.
- `STEPS_PER_GENERATION`: optional alternative to `GENERATION_BATCH_SIZE` (do not set both).
- `NUM_TRAIN_EPOCHS`/`MAX_STEPS`: control training length (epochs or optimizer steps).
  If both are set, `MAX_STEPS` wins. Defaults to `MAX_STEPS=500`. With the default Hanabi dataset (1 row), each epoch is ~1 optimizer step.

## Troubleshooting
- "Address already in use": change `PORT` in `tools/rollout/rollout_hanabi_gym.sh`
  and update `VLLM_SERVER_PORT` for training.
- NCCL issues: for H800, keep `NCCL_P2P_DISABLE=0` and `NCCL_IB_DISABLE=0`.
- If prompt tokens keep growing / `negative max_tokens` appears:
  set `CONTEXT_MANAGER=hanabi_recent_turns` and keep `HANABI_CTX_MAX_TURNS=1`.

## Field notes: 10x A100-PCIE-40GB mixed-topology host

Observed on March 10, 2026 on the current single-node A100 PCIe machine:

- `2 rollout + 8 train` with `NCCL_P2P_DISABLE=0` and `NCCL_IB_DISABLE=0`
  repeatedly stalled before the first generation.
- The failure signature was:
  - train log printed `Start connecting to vLLM server`
  - train log never printed `Connected to vLLM server`
  - rollout logs only showed `/close_communicator/`, `/get_world_size/`,
    `/init_communicator/`
  - rollout logs never showed `/infer/`
- This points to the external vLLM weight-sync communicator path hanging during
  `init_communicator`, not to dataset quality, reward logic, or missing NVLink.

What worked better on the same host:

- `4 rollout + 4 train`
- `NCCL_P2P_DISABLE=1`
- `NCCL_IB_DISABLE=1`
- one rollout server per GPU (`TP=1`, `DP=1`)

With that profile, the run was able to:

- print `Connected to vLLM server`
- enter `Train: 0/...`
- push repeated `/update_flattened_params/` requests to all rollout servers

Practical recommendation for this machine:

- Start from `4 rollout + 4 train`, not `2 rollout + 8 train`.
- Keep `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` unless a new smoke run
  proves `0/0` is stable on the same topology.
- Treat repeated `/update_flattened_params/` before the first `/infer/` as
  normal first-step LoRA weight sync, not an immediate hang.

If you need to distinguish the two failure modes quickly:

- Communicator-init hang:
  - only `Start connecting to vLLM server`
  - no `Connected to vLLM server`
  - no `/infer/`
- Slow but progressing startup:
  - `Connected to vLLM server` appears
  - rollout logs show many `/update_flattened_params/`
  - train eventually reaches `Train: 0/...`
