# Hanabi DAPO (ms-swift)

This project can run Hanabi with `ms-swift` DAPO by reusing the existing
Hanabi gym rollout server and switching the training loss preset.

In `ms-swift 4.0.x`, DAPO is configured as:

```bash
swift rlhf \
  --rlhf_type grpo \
  --loss_type dapo \
  --beta 0
```

The rollout side is unchanged from Hanabi GRPO. Only the training wrapper is
different.

## Quick start

Terminal 1 (rollout server):

```bash
bash tools/rollout/rollout_hanabi_gym.sh
```

Terminal 2 (DAPO training with auto GPU split and W&B defaults):

```bash
bash tools/train/train_dapo_hanabi_server_wandb.sh
```

If you want explicit control instead of auto GPU selection:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
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
bash tools/train/train_dapo_hanabi_server_simple.sh
```

## DAPO-specific knobs

The new wrappers default to:

- `LOSS_TYPE=dapo`
- `BETA=0`

You can override them through the environment:

```bash
LOSS_TYPE=dapo \
BETA=0 \
MAX_COMPLETION_LENGTH=128 \
bash tools/train/train_dapo_hanabi_server_simple.sh
```

Additional `swift rlhf` parameters can still be passed with
`EXTRA_SWIFT_ARGS`, for example:

```bash
SOFT_MAX_LENGTH=96 \
OVERLONG_FILTER=true \
EXTRA_SWIFT_ARGS="--epsilon_high 0.28 --delta 1.5 --dynamic_sample true" \
bash tools/train/train_dapo_hanabi_server_simple.sh
```

## Version note

On the current host, the installed version is `ms-swift 4.0.0`.
This version supports `loss_type=dapo`, but it does not expose a
`--use_valid_tokens_only` CLI argument. Keep DAPO presets compatible with the
local CLI unless you upgrade `ms-swift`.

## Notes

- Hanabi still uses gym-env rewards from `tools/rollout/hanabi_gym_plugin.py`,
  so `REWARD_FUNCS` remains empty.
- The rollout launchers under `tools/rollout/` do not need any DAPO-specific
  changes.
- For short-action Hanabi training, the existing defaults
  `MAX_LENGTH=4096` and `MAX_COMPLETION_LENGTH=64` are still the right place to
  start. Increase completion length only if you are explicitly training a
  thinking-style policy.
