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
    QWEN_LORA_TARGET_MODULES,
    get_training_preset,
    list_training_presets,
    launch_verl,
    materialize_run_plan,
    prepare_run_plan,
    print_run_plan,
    resolve_launch_config,
)


def _build_preset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=tuple(p.name for p in list_training_presets()), default=None)
    parser.add_argument("--list-presets", action="store_true")
    return parser


def _build_parser(*, preset_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiniHanabi, Colonel Blotto, or Negotiation with pure VERL.")
    preset_defaults = get_training_preset(preset_name).cli_defaults if preset_name else {}
    parser.add_argument(
        "--preset",
        choices=tuple(p.name for p in list_training_presets()),
        default=preset_name,
        help="Apply a named starter preset, then allow explicit CLI overrides on top.",
    )
    parser.add_argument("--list-presets", action="store_true", help="Print available starter presets and exit.")
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
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM GPU memory utilization override. Omit to use VERL's default rollout setting.",
    )
    parser.add_argument("--rollout-max-model-len", type=int, default=None)
    parser.add_argument("--rollout-max-num-batched-tokens", type=int, default=6144)
    parser.add_argument("--rollout-max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        default=None,
        help=(
            "Disable Qwen thinking mode during chat templating by setting "
            "data.apply_chat_template_kwargs.enable_thinking=false."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        dest="enable_thinking",
        help=(
            "Explicitly enable Qwen thinking mode during chat templating by setting "
            "data.apply_chat_template_kwargs.enable_thinking=true."
        ),
    )
    parser.add_argument(
        "--rollout-sleep-mode",
        action="store_true",
        dest="rollout_enable_sleep_mode",
        default=None,
        help="Force-enable vLLM sleep mode for rollout workers.",
    )
    parser.add_argument(
        "--no-rollout-sleep-mode",
        action="store_false",
        dest="rollout_enable_sleep_mode",
        help="Disable vLLM sleep mode for rollout workers.",
    )
    parser.add_argument(
        "--rollout-enforce-eager",
        action="store_true",
        dest="rollout_enforce_eager",
        default=None,
        help="Force-enable vLLM eager mode for rollout workers to avoid CUDA graph capture.",
    )
    parser.add_argument(
        "--no-rollout-enforce-eager",
        action="store_false",
        dest="rollout_enforce_eager",
        help="Force-disable vLLM eager mode for rollout workers.",
    )
    parser.add_argument(
        "--adv-estimator",
        choices=("gae", "grpo"),
        default="gae",
        help="Use `gae` for step-wise PPO with a critic. `grpo` is unsupported for snapshot-only full-episode training.",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=("full", "lora"),
        default="full",
        help="Use full-parameter updates or LoRA adapters for actor/critic training.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--lora-target-modules",
        default=QWEN_LORA_TARGET_MODULES,
        help=(
            "Use `all-linear` or a comma-separated module list. "
            "The default targets the standard Qwen attention/MLP projections and avoids the critic score head."
        ),
    )
    parser.add_argument(
        "--lora-adapter-path",
        default=None,
        help="Optional path to existing LoRA adapter weights to continue PPO training from.",
    )
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--critic-ppo-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--ref-log-prob-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional actor learning rate override. Omit to use VERL's default actor optimizer config.",
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=None,
        help="Optional critic learning rate override. Omit to use VERL's default critic optimizer config.",
    )
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
    if preset_defaults:
        parser.set_defaults(**preset_defaults)
    return parser


def _print_presets() -> None:
    for preset in list_training_presets():
        print(f"{preset.name}\t{preset.description}")


def main() -> None:
    bootstrap_parser = _build_preset_parser()
    bootstrap_args, _remaining = bootstrap_parser.parse_known_args()
    if bootstrap_args.list_presets:
        _print_presets()
        return

    parser = _build_parser(preset_name=bootstrap_args.preset)
    args = parser.parse_args()
    if args.list_presets:
        _print_presets()
        return

    config = resolve_launch_config(args)
    plan = prepare_run_plan(config, root_dir=ROOT_DIR, repo_parent=REPO_PARENT)
    materialize_run_plan(plan)

    if config.print_config or config.dry_run:
        print_run_plan(plan)
    if config.dry_run:
        return

    launch_verl(plan, root_dir=ROOT_DIR, repo_parent=REPO_PARENT)


if __name__ == "__main__":
    main()
