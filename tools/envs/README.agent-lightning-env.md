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
