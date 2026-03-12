# Agent Lightning Environment

Create a clean environment for this branch:

```bash
bash tools/envs/create_agent_lightning_env.sh
```

Use a custom target path if needed:

```bash
ENV_DIR=/workspace/mindgames/.venv-agent-lightning-qwen \
bash tools/envs/create_agent_lightning_env.sh
```

This environment installs only the extras required by the new training path:
- `agents`
- `agent-lightning`

`tools/train/train_agent_lightning_hanabi.sh` uses `.venv-agent-lightning` by default.
To point training at a different isolated env, set `AGENT_LIGHTNING_ENV_DIR=/path/to/venv`.
