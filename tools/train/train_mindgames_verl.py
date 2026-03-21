#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_PARENT = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(REPO_PARENT))

from mindgames.verl_training import (  # noqa: E402
    DEFAULT_ENV_IDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_REWARD_PLAYER,
    build_dataset_row,
    default_max_steps,
    default_reward_player,
    resolve_env_id,
)


def _default_experiment_name(game: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{game}-verl-{timestamp}"


def _safe_name(value: str) -> str:
    allowed = {"-", "_", "."}
    return "".join(char if char.isalnum() or char in allowed else "-" for char in value)


def _build_rows(
    *,
    game: str,
    count: int,
    seed_start: int,
    env_id: str,
    max_steps: int,
    reward_player: int,
) -> list[dict[str, Any]]:
    return [
        build_dataset_row(
            game=game,  # type: ignore[arg-type]
            seed=seed_start + i,
            index=seed_start + i,
            env_id=env_id,
            max_steps=max_steps,
            reward_player=reward_player,
        )
        for i in range(count)
    ]


def _effective_val_rows(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if val_rows:
        return val_rows
    if not train_rows:
        return []
    placeholder = json.loads(json.dumps(train_rows[0]))
    placeholder["extra_info"]["index"] = -1
    return [placeholder]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _write_interaction_config(path: Path) -> None:
    payload = (
        "interaction:\n"
        "  - name: mindgames\n"
        "    class_name: mindgames.verl_training.MindGamesInteraction\n"
        "    config: {}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _build_overrides(
    args: argparse.Namespace,
    *,
    train_path: Path,
    val_path: Path,
    interaction_config_path: Path,
) -> list[str]:
    logger_list = "[console,wandb]" if args.wandb else "[console]"
    max_model_len = args.rollout_max_model_len or (args.rollout_prompt_length + args.rollout_response_length)

    overrides = [
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        "data.truncation=error",
        "data.shuffle=False",
        f"algorithm.adv_estimator={args.adv_estimator}",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={'True' if args.gradient_checkpointing else 'False'}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.prompt_length={args.rollout_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={args.rollout_response_length}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_seqs={args.rollout_max_num_seqs}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={args.rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.log_prob_micro_batch_size_per_gpu}",
        (
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="
            f"{args.rollout_update_weights_bucket_megabytes}"
        ),
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={args.max_steps}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={args.max_steps}",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config_path}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        f"actor_rollout_ref.actor.entropy_coeff={args.entropy_coeff}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={'True' if args.param_offload else 'False'}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={'True' if args.optimizer_offload else 'False'}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.ref_log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={'True' if args.ref_param_offload else 'False'}",
        f"reward.custom_reward_function.path=pkg://mindgames.verl_training",
        "reward.custom_reward_function.name=compute_score",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.logger={logger_list}",
        f"trainer.nnodes=1",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.val_before_train={'True' if args.val_before_train else 'False'}",
        "trainer.resume_mode=disable",
        "trainer.critic_warmup=0",
        f"++ray_kwargs.ray_init.runtime_env.working_dir={ROOT_DIR}",
        f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={ROOT_DIR}:{REPO_PARENT}",
    ]
    if args.adv_estimator == "gae":
        overrides.extend(
            [
                "critic.enable=True",
                f"critic.model.path={args.model}",
                f"critic.model.tokenizer_path={args.model}",
                (
                    "critic.model.enable_gradient_checkpointing="
                    f"{'True' if args.gradient_checkpointing else 'False'}"
                ),
                f"critic.optim.lr={args.critic_learning_rate}",
                f"critic.ppo_micro_batch_size_per_gpu={args.critic_ppo_micro_batch_size_per_gpu}",
            ]
        )
    else:
        overrides.append("critic.enable=False")
    return overrides


def _print_resolved_plan(
    args: argparse.Namespace,
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_path: Path,
    val_path: Path,
    interaction_config_path: Path,
    overrides: list[str],
) -> None:
    payload = {
        "game": args.game,
        "env_id": args.env_id,
        "adv_estimator": args.adv_estimator,
        "critic_enabled": args.adv_estimator == "gae",
        "reward_player": args.reward_player,
        "max_steps": args.max_steps,
        "model": args.model,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "interaction_config": str(interaction_config_path),
        "overrides": overrides,
        "train_example": train_rows[0] if train_rows else None,
        "val_example": val_rows[0] if val_rows else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


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
    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = _default_experiment_name(args.game)
    if args.env_id is None:
        args.env_id = DEFAULT_ENV_IDS[args.game]
    if args.max_steps is None:
        args.max_steps = default_max_steps(args.game)  # type: ignore[arg-type]
    if args.reward_player is None:
        args.reward_player = default_reward_player(args.game)  # type: ignore[arg-type]

    args.env_id = resolve_env_id(args.game, args.env_id)  # type: ignore[arg-type]

    train_rows = _build_rows(
        game=args.game,
        count=args.train_size,
        seed_start=args.train_seed_start,
        env_id=args.env_id,
        max_steps=args.max_steps,
        reward_player=args.reward_player,
    )
    raw_val_rows = _build_rows(
        game=args.game,
        count=args.val_size,
        seed_start=args.val_seed_start,
        env_id=args.env_id,
        max_steps=args.max_steps,
        reward_player=args.reward_player,
    )
    val_rows = _effective_val_rows(train_rows, raw_val_rows)

    run_dir = ROOT_DIR / "outputs" / "verl_runs" / _safe_name(args.experiment_name)
    train_path = run_dir / "train.jsonl"
    val_path = run_dir / "val.jsonl"
    interaction_config_path = run_dir / "interaction.yaml"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_interaction_config(interaction_config_path)

    overrides = _build_overrides(
        args,
        train_path=train_path,
        val_path=val_path,
        interaction_config_path=interaction_config_path,
    )

    if args.print_config or args.dry_run:
        _print_resolved_plan(
            args,
            train_rows=train_rows,
            val_rows=val_rows,
            train_path=train_path,
            val_path=val_path,
            interaction_config_path=interaction_config_path,
            overrides=overrides,
        )
    if args.dry_run:
        return

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT_DIR), str(REPO_PARENT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    command = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    subprocess.run(command, check=True, cwd=str(ROOT_DIR), env=env)


if __name__ == "__main__":
    main()
