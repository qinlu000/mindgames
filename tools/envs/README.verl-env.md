# VERL Environment

This branch uses a uv-managed project environment for pure VERL training on MiniHanabi, Colonel Blotto, and Negotiation.

Create or refresh the environment:

```bash
bash tools/envs/create_verl_env.sh
```

Default target env:
- `.venv` in the current worktree root

The environment is fully described by `pyproject.toml` and `uv.lock`.
The setup script does two things:
1. `uv sync --extra train`
2. verify the main training imports

For the current Linux x86_64 Python 3.12 setup, the `train` extra includes the official prebuilt `flash-attn` wheel for `torch==2.8.0`, so `uv sync` no longer needs a local FlashAttention build step.

Useful override:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cu128" \
bash tools/envs/create_verl_env.sh
```

After the env is ready, the shared launcher runs through the same environment:

```bash
bash tools/train/train_mindgames_verl.sh --game mini_hanabi --dry-run --print-config
```
