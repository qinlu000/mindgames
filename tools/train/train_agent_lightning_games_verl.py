#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from agent_lightning_games import (
    DEFAULT_ENV_IDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REWARD_PLAYER,
    GameTask,
    default_max_steps,
    default_reward_player,
    make_rollout,
    resolve_default_qwen3_8b_model,
    resolve_env_id,
)


def _default_experiment_name(game: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{game}-agl-{timestamp}"


def _build_tasks(
    *,
    game: str,
    count: int,
    seed_start: int,
    env_id: str,
    max_steps: int,
    enable_thinking: bool,
    reward_player: int,
) -> list[GameTask]:
    return [
        GameTask(
            game=game,
            seed=seed_start + i,
            env_id=env_id,
            max_steps=max_steps,
            enable_thinking=enable_thinking,
            reward_player=reward_player,
        )
        for i in range(count)
    ]


def _build_verl_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": args.train_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "tensor_model_parallel_size": args.tensor_model_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "n": args.rollout_n,
                "log_prob_micro_batch_size_per_gpu": args.log_prob_micro_batch_size_per_gpu,
            },
            "actor": {
                "ppo_mini_batch_size": args.ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": args.ppo_micro_batch_size_per_gpu,
                "optim": {"lr": args.learning_rate},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": args.entropy_coeff,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": args.param_offload,
                    "optimizer_offload": args.optimizer_offload,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": args.ref_log_prob_micro_batch_size_per_gpu,
                "fsdp_config": {"param_offload": args.ref_param_offload},
            },
            "model": {
                "path": args.model,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": args.gradient_checkpointing,
            },
        },
        "trainer": {
            "n_gpus_per_node": args.n_gpus_per_node,
            "val_before_train": args.val_before_train,
            "critic_warmup": 0,
            "project_name": args.project_name,
            "experiment_name": args.experiment_name,
            "nnodes": 1,
            "test_freq": args.test_freq,
            "save_freq": args.save_freq,
            "total_epochs": args.total_epochs,
            "logger": ["console", "wandb"] if args.wandb else ["console"],
        },
    }


def _print_resolved_plan(args: argparse.Namespace, train_tasks: list[GameTask], val_tasks: list[GameTask]) -> None:
    payload = {
        "mode": args.mode,
        "game": args.game,
        "env_id": resolve_env_id(args.game, args.env_id),
        "reward_player": args.reward_player,
        "max_steps": args.max_steps,
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "model": args.model,
        "dev_endpoint": args.llm_endpoint,
        "same_llm_all_seats": True,
        "verl_config": _build_verl_config(args),
        "train_task_example": train_tasks[0] if train_tasks else None,
        "val_task_example": val_tasks[0] if val_tasks else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_dev(args: argparse.Namespace, train_tasks: list[GameTask], val_tasks: list[GameTask]) -> None:
    import agentlightning as agl

    trainer = agl.Trainer(
        n_workers=args.n_workers,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=args.llm_endpoint,
                api_key=args.api_key,
                model=args.model,
                sampling_parameters={
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                },
            )
        },
    )
    trainer.dev(make_rollout(), train_dataset=train_tasks, val_dataset=val_tasks or None)


def _run_train(args: argparse.Namespace, train_tasks: list[GameTask], val_tasks: list[GameTask]) -> None:
    import agentlightning as agl

    algorithm = agl.VERL(_build_verl_config(args))
    trainer = agl.Trainer(
        n_runners=args.n_runners,
        algorithm=algorithm,
    )
    trainer.fit(make_rollout(), train_dataset=train_tasks, val_dataset=val_tasks or None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MiniHanabi, Colonel Blotto, or Negotiation with Agent Lightning + VERL."
    )
    parser.add_argument("--mode", choices=("dev", "train"), default="dev")
    parser.add_argument(
        "--game",
        choices=("mini_hanabi", "colonel_blotto", "negotiation"),
        default="mini_hanabi",
    )
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model", default=resolve_default_qwen3_8b_model())
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--val-seed-start", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reward-player", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")

    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8021/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)

    parser.add_argument("--n-runners", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--rollout-n", type=int, default=4, help="GRPO group size.")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=32)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--ref-log-prob-micro-batch-size-per-gpu", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--entropy-coeff", type=float, default=1e-3)
    parser.add_argument("--n-gpus-per-node", type=int, default=2)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=32)
    parser.add_argument("--project-name", default="mindgames-agent-lightning")
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
    args = parser.parse_args()
    if args.experiment_name is None:
        args.experiment_name = _default_experiment_name(args.game)
    if args.max_steps is None:
        args.max_steps = default_max_steps(args.game)
    if args.reward_player is None:
        args.reward_player = default_reward_player(args.game)

    train_tasks = _build_tasks(
        game=args.game,
        count=args.train_size,
        seed_start=args.train_seed_start,
        env_id=resolve_env_id(args.game, args.env_id),
        max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        reward_player=args.reward_player,
    )
    val_tasks = _build_tasks(
        game=args.game,
        count=args.val_size,
        seed_start=args.val_seed_start,
        env_id=resolve_env_id(args.game, args.env_id),
        max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        reward_player=args.reward_player,
    )

    if args.print_config or args.dry_run:
        _print_resolved_plan(args, train_tasks, val_tasks)
    if args.dry_run:
        return

    if args.mode == "dev":
        _run_dev(args, train_tasks, val_tasks)
    else:
        _run_train(args, train_tasks, val_tasks)


if __name__ == "__main__":
    main()
