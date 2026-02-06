#!/usr/bin/env bash
set -euo pipefail

# Run the Hanabi vLLM rollout experiment via expctl (experiment registry).
# The experiment itself starts the vLLM server + parallel workers.
#
# Usage:
#   bash tools/rollout/run_hanabi_qwen3_8b_vllm_4gpu_100_expctl.sh
#
# Defaults (override via env vars):
#   EXP_PATH=experiments/hanabi_qwen3_8b_vllm_no_thinking_100/experiment.yaml
#   RUN_TAG=now
#   NO_WANDB=1

EXP_PATH="${EXP_PATH:-experiments/hanabi_qwen3_8b_vllm_thinking_8192_100/experiment.yaml}"
RUN_TAG="${RUN_TAG:-now}"
NO_WANDB="${NO_WANDB:-1}"
TAIL_LOG="${TAIL_LOG:-1}"

if [[ -x ".venv/bin/python" ]]; then
  PY=(.venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python)
fi

if [[ ! -f "${EXP_PATH}" ]]; then
  echo "EXP_PATH not found: ${EXP_PATH}" >&2
  exit 1
fi

run_dir="$("${PY[@]}" tools/exp/expctl.py prepare "${EXP_PATH}" --run-tag "${RUN_TAG}")"
echo "expctl prepared run_dir: ${run_dir}"

tail_pid=""
cleanup() {
  if [[ -n "${tail_pid}" ]] && kill -0 "${tail_pid}" >/dev/null 2>&1; then
    kill "${tail_pid}" || true
    wait "${tail_pid}" || true
  fi
}
trap cleanup EXIT

if [[ "${TAIL_LOG}" == "1" || "${TAIL_LOG}" == "true" ]]; then
  echo "Tailing: ${run_dir}/run.log"
  tail -n 50 -F "${run_dir}/run.log" &
  tail_pid="$!"
fi

exp_args=(run "${EXP_PATH}" --resume --run-dir "${run_dir}")
if [[ "${NO_WANDB}" == "1" || "${NO_WANDB}" == "true" ]]; then
  exp_args+=(--no-wandb)
fi

"${PY[@]}" tools/exp/expctl.py "${exp_args[@]}" >/dev/null

echo "Done. Results:"
echo "- ${run_dir}/summary.json"
echo "- ${run_dir}/rollouts.jsonl"
echo "- ${run_dir}/run.log"
