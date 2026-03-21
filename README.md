# mindgames

This branch focuses on three training targets only:
- `MiniHanabi`
- `ColonelBlotto`
- `Negotiation`

The RL training mainline is now pure `VERL`.

## Environment

This branch uses a uv-managed project environment in the current worktree.

Create or refresh `.venv`:

```bash
bash tools/envs/create_verl_env.sh
```

The training stack now lives in `pyproject.toml` + `uv.lock` and is synced with:
- `uv sync --extra train`
- `uv run --extra train ...`

On this Linux x86_64 Python 3.12 host, the `train` extra pins the official prebuilt `flash-attn` wheel that matches the repo's `torch==2.8.0` stack, so `uv sync --extra train` keeps VERL training runnable without a local CUDA build.

Environment notes are in `tools/envs/README.verl-env.md`.

## Training Mainline

The shared entrypoint is:

```bash
bash tools/train/train_mindgames_verl.sh
```

The pure VERL path uses the `MindGamesInteraction` multi-turn interface:
- each episode stays inside one VERL interaction
- each user turn is still a self-contained game-state snapshot from the environment
- `MiniHanabi` uses normalized team score (`score / 9`) as the reward
- `ColonelBlotto` and `Negotiation` read terminal reward from player `0` by default

### Dry run

```bash
bash tools/train/train_mindgames_verl.sh \
  --game mini_hanabi \
  --dry-run \
  --print-config
```

### Formal training

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash tools/train/train_mindgames_verl.sh \
  --game colonel_blotto \
  --model /workspace/models/Qwen3-8B \
  --n-gpus-per-node 2 \
  --train-batch-size 32 \
  --rollout-n 4 \
  --wandb
```

### PPO with MiniHanabi

Use `gae` to enable the standard PPO path with a critic:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash tools/train/train_mindgames_verl.sh \
  --game mini_hanabi \
  --adv-estimator gae \
  --model /workspace/models/Qwen3-8B \
  --n-gpus-per-node 2 \
  --train-batch-size 16 \
  --rollout-n 2 \
  --ppo-micro-batch-size-per-gpu 1 \
  --critic-ppo-micro-batch-size-per-gpu 1
```

Useful game values:
- `--game mini_hanabi`
- `--game colonel_blotto`
- `--game negotiation`

Useful overrides:
- `--env-id`: choose a non-default env id for the same game family
- `--max-steps`: cap the rollout loop manually
- `--reward-player`: override which player reward is read for competitive games
- `--adv-estimator gae`: standard PPO with a critic
- `--adv-estimator grpo`: critic-free grouped rollouts
- `UV_PROJECT_ENVIRONMENT`: point uv at a non-default virtualenv path

## Offline Rollouts

The generic rollout runner is still available for local evaluation:

```bash
uv run --extra agents python tools/run_rollouts.py \
  --env-id MiniHanabi-v0-train \
  --num-players 2 \
  --episodes 1 \
  --agent openai:gpt-4.1-mini \
  --agent openai:gpt-4.1-mini \
  --out data/minihanabi_rollouts.jsonl
```

Swap `MiniHanabi-v0-train` for `ColonelBlotto-v0-train` or `Negotiation-v0-train` as needed.

## W&B

Online logging:

```bash
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=<your_entity_or_team>
export WANDB_PROJECT=mindgames-verl
```

Then add `--wandb` to the training command.
