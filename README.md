# mindgames

This branch focuses on three training targets only:
- `MiniHanabi`
- `ColonelBlotto`
- `Negotiation`

The RL training mainline is now `Agent Lightning + VERL`.

## Environment

This branch uses a uv-managed project environment in the current worktree.

Create or refresh `.venv`:

```bash
bash tools/envs/create_agent_lightning_verl_env.sh
```

The training stack now lives in `pyproject.toml` + `uv.lock` and is synced with:
- `uv sync --extra agents --extra train`
- `uv run --extra agents --extra train ...`

Default sync stays minimal and working on this host: `flash-attn` is not installed by default because the machine exposes CUDA 13.1 while the pinned PyTorch stack is `cu128`.

Environment notes are in `tools/envs/README.agent-lightning-verl-env.md`.

## Training Mainline

The shared entrypoint is:

```bash
bash tools/train/train_agent_lightning_games_verl.sh
```

The rollout shape is intentionally minimal and follows the official docs:
- define one `@rollout`
- use `Trainer.dev()` for smoke testing
- use `agl.VERL(...) + Trainer.fit()` for formal training

Current simplification for this branch:
- every seat is controlled by the same LLM agent
- `MiniHanabi` uses normalized team score (`score / 9`) as the reward
- `ColonelBlotto` and `Negotiation` read terminal reward from player `0` by default

### Dry run

```bash
bash tools/train/train_agent_lightning_games_verl.sh \
  --mode train \
  --game mini_hanabi \
  --dry-run \
  --print-config
```

### Dev smoke test

Start a local OpenAI-compatible endpoint first, then run:

```bash
bash tools/train/train_agent_lightning_games_verl.sh \
  --mode dev \
  --game mini_hanabi \
  --llm-endpoint http://127.0.0.1:8021/v1 \
  --model /workspace/models/Qwen3-8B
```

### Formal training

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash tools/train/train_agent_lightning_games_verl.sh \
  --mode train \
  --game colonel_blotto \
  --model /workspace/models/Qwen3-8B \
  --n-runners 4 \
  --n-gpus-per-node 2 \
  --train-batch-size 32 \
  --rollout-n 4 \
  --wandb
```

Useful game values:
- `--game mini_hanabi`
- `--game colonel_blotto`
- `--game negotiation`

Useful overrides:
- `--env-id`: choose a non-default env id for the same game family
- `--max-steps`: cap the rollout loop manually
- `--reward-player`: override which player reward is read for competitive games
- `--enable-thinking`: pass Qwen thinking mode through the shared LLM agent
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
export WANDB_PROJECT=mindgames-agent-lightning
```

Then add `--wandb` to the training command.
