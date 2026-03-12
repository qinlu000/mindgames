# Agent Lightning VERL Environment

Create a dedicated environment for Agent Lightning weight training with VERL:

```bash
bash tools/envs/create_agent_lightning_verl_env.sh
```

Default target env:
- `.venv-agent-lightning-verl`

This is intentionally separate from `.venv-agent-lightning`:
- `.venv-agent-lightning`: prompt optimization / APO path
- `.venv-agent-lightning-verl`: weight-training / VERL path

The script follows the official Agent Lightning GPU setup order:
1. install project dependencies for this repo
2. install PyTorch + torchvision
3. install `flash-attn`
4. install `vllm`
5. install `verl`
6. install `agentlightning[verl]`

Useful overrides:

```bash
ENV_DIR=/workspace/mindgames/.venv-agent-lightning-verl-qwen \
TORCH_VERSION=2.8.0 \
TORCHVISION_VERSION=0.23.0 \
VLLM_VERSION=0.10.2 \
VERL_VERSION=0.5.0 \
bash tools/envs/create_agent_lightning_verl_env.sh
```

If your machine needs a different PyTorch wheel index, override `TORCH_INDEX_URL`.

After the env is ready, the Hanabi weight-training wrapper uses it directly:

```bash
bash tools/train/train_agent_lightning_hanabi_verl.sh --dry-run --print-config
```
