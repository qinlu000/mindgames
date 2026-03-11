# Hanabi GRPO Launchers

Use the dedicated tmux launchers instead of the older generic scripts.

## No-think

Script: `tools/tmux/launch_hanabi_no_think_grpo_tmux.sh`

Default profile:
- `ENABLE_THINKING=false`
- `MAX_LENGTH=8192`
- `MAX_COMPLETION_LENGTH=256`
- `LEARNING_RATE=5e-6`
- `ASYNC_GENERATE=false`

Example:

```bash
ADAPTER=/path/to/no-think-sft-lora
ADAPTERS="$ADAPTER" \
REF_ADAPTERS="$ADAPTER" \
bash tools/tmux/launch_hanabi_no_think_grpo_tmux.sh
```

## Think

Script: `tools/tmux/launch_hanabi_think_grpo_tmux.sh`

Default profile:
- `ENABLE_THINKING=true`
- `MAX_LENGTH=12288`
- `MAX_COMPLETION_LENGTH=2048`
- `SOFT_MAX_LENGTH=1536`
- `OVERLONG_FILTER=true`
- `LEARNING_RATE=2e-6`
- `ASYNC_GENERATE=false`

Example:

```bash
ADAPTER=/path/to/think-sft-lora
ADAPTERS="$ADAPTER" \
REF_ADAPTERS="$ADAPTER" \
bash tools/tmux/launch_hanabi_think_grpo_tmux.sh
```

## Notes

- Both launchers default to `/workspace/mindgames/.venv-grpo/bin/swift`.
- Both start rollout first, wait for `/health/`, then start training.
- On the current 10x A100-PCIE-40GB host, start from `4 rollout + 4 train`
  with `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`. The older
  `2 rollout + 8 train` layout was observed to stall in the external vLLM
  communicator path before the first `/infer/`.
- For the no-think launcher, `ENABLE_THINKING=false` does not remove the Qwen3
  `<think>` wrapper entirely. In the current ms-swift/Qwen3 template, it uses
  an empty prefix like `<think>\n\n</think>\n\n`. "No-think" here means "no
  reasoning content", not "no think tags at all".
- Before the first rollout request, training may spend noticeable time pushing
  LoRA weights to all rollout servers. During that phase, rollout logs show
  repeated `/update_flattened_params/` and may still have no `/infer/`.
- Override any default with environment variables, for example:

```bash
ADAPTERS="$ADAPTER" \
REF_ADAPTERS="$ADAPTER" \
MAX_COMPLETION_LENGTH=3072 \
NUM_GENERATIONS=6 \
bash tools/tmux/launch_hanabi_think_grpo_tmux.sh
```

Example for the current A100 PCIe host:

```bash
ADAPTER=/root/.cache/modelscope/hub/models/qinlu000/qwen3-8b-hanabi-no-think-lora-16epoch
ADAPTERS="$ADAPTER" \
REF_ADAPTERS="$ADAPTER" \
ROLLOUT_GPU_LIST=0,1,2,3 \
TRAIN_GPU_LIST=5,6,7,8 \
PORTS=8200,8201,8202,8203 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
bash tools/tmux/launch_hanabi_no_think_grpo_tmux.sh
```
