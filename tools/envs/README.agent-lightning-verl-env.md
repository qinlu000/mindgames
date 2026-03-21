# Agent Lightning VERL Environment

This branch now uses a uv-managed project environment for Agent Lightning training on MiniHanabi, Colonel Blotto, and Negotiation.

Create or refresh the environment:

```bash
bash tools/envs/create_agent_lightning_verl_env.sh
```

Default target env:
- `.venv` in the current worktree root

The environment is fully described by `pyproject.toml` and `uv.lock`.
The setup script now does only two things:
1. `uv sync --extra agents --extra train`
2. verify the main training imports

Note: `flash-attn` is intentionally excluded from the default uv environment on this machine because the detected host CUDA is 13.1 while the pinned PyTorch build is `cu128`, which makes the package fail to build.

Useful override:

```bash
UV_PROJECT_ENVIRONMENT=/workspace/mindgames-agent-lightning-games/.venv-cu128 \
bash tools/envs/create_agent_lightning_verl_env.sh
```

After the env is ready, the shared launcher runs through uv as well:

```bash
bash tools/train/train_agent_lightning_games_verl.sh --mode train --game mini_hanabi --dry-run --print-config
```
