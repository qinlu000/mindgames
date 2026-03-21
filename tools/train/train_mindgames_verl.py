#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_PARENT = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(REPO_PARENT))

from mindgames.training import (  # noqa: E402
    launch_verl,
    materialize_run_plan,
    prepare_run_plan,
    print_run_plan,
    resolve_launch_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiniHanabi, Colonel Blotto, or Negotiation with pure VERL.")
    parser.add_argument(
        "--game",
        choices=("mini_hanabi", "colonel_blotto", "negotiation"),
        default="mini_hanabi",
    )
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model", default="/workspace/models/Qwen3-8B")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--val-seed-start", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reward-player", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")

    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--rollout-prompt-length", type=int, default=1024)
    parser.add_argument("--rollout-response-length", type=int, default=5120)
    parser.add_argument("--rollout-n", type=int, default=2)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--rollout-max-model-len", type=int, default=None)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=6144)
    parser.add_argument("--rollout-max-num-seqs", type=int, default=1)
    parser.add_argument("--rollout-update-weights-bucket-megabytes", type=int, default=4096)
    parser.add_argument(
        "--adv-estimator",
        choices=("gae", "grpo"),
        default="grpo",
        help="Use `gae` for standard PPO with a critic, or `grpo` for critic-free grouped rollouts.",
    )
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--critic-ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--ref-log-prob-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-5)
    parser.add_argument("--entropy-coeff", type=float, default=1e-3)
    parser.add_argument("--n-gpus-per-node", type=int, default=4)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--project-name", default="mindgames-verl")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--val-before-train", action="store_true", default=True)
    parser.add_argument("--no-val-before-train", action="store_false", dest="val_before_train")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--param-offload", action="store_true", default=True)
    parser.add_argument("--no-param-offload", action="store_false", dest="param_offload")
    parser.add_argument("--optimizer-offload", action="store_true", default=True)
    parser.add_argument("--no-optimizer-offload", action="store_false", dest="optimizer_offload")
    parser.add_argument("--ref-param-offload", action="store_true", default=True)
    parser.add_argument("--no-ref-param-offload", action="store_false", dest="ref_param_offload")
    return parser


def main() -> None:
    parser = _build_parser()
    config = resolve_launch_config(parser.parse_args())
    plan = prepare_run_plan(config, root_dir=ROOT_DIR, repo_parent=REPO_PARENT)
    materialize_run_plan(plan)

    if config.print_config or config.dry_run:
        print_run_plan(plan)
    if config.dry_run:
        return

    launch_verl(plan, root_dir=ROOT_DIR, repo_parent=REPO_PARENT)


if __name__ == "__main__":
    main()
