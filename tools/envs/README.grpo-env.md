# GRPO Env

This repo's default `.venv` follows `pyproject.toml` and `uv.lock`, which currently
pin the main project to `vllm>=0.15.1,<0.16`.

For GRPO training with `ms-swift`, use a separate environment if you want to test a
more conservative `trl/vllm` combination without changing the main project env.

Default isolated package set used by `tools/envs/create_grpo_env.sh`:

- `ms-swift==4.0.0`
- `trl==0.24.0`
- `vllm==0.10.2`
- `deepspeed==0.18.7`
- `transformers==4.57.6`
- `accelerate==1.12.0`

Create it with:

```bash
bash tools/envs/create_grpo_env.sh
```

Override versions if needed:

```bash
GRPO_VLLM_VERSION=0.13.0 \
GRPO_TRL_VERSION=0.24.0 \
bash tools/envs/create_grpo_env.sh
```
