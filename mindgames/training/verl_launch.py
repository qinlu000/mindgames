from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mindgames.training.contracts import GameName
from mindgames.training.dataset import build_dataset_row
from mindgames.training.specs import (
    DEFAULT_ENV_IDS,
    default_max_steps,
    default_reward_player,
    resolve_env_id,
)


INTERACTION_CLASS = "mindgames.training.verl_adapter.MindGamesInteraction"
REWARD_FUNCTION_PATH = "pkg://mindgames.training.verl_adapter"


@dataclass(frozen=True)
class VerlLaunchConfig:
    game: GameName
    env_id: str
    model: str
    train_size: int
    val_size: int
    train_seed_start: int
    val_seed_start: int
    max_steps: int
    reward_player: int
    dry_run: bool
    print_config: bool
    train_batch_size: int
    max_prompt_length: int
    rollout_prompt_length: int
    rollout_response_length: int
    rollout_n: int
    tensor_model_parallel_size: int
    gpu_memory_utilization: float
    rollout_max_model_len: int | None
    rollout_max_num_batched_tokens: int
    rollout_max_num_seqs: int
    rollout_update_weights_bucket_megabytes: int
    adv_estimator: str
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    critic_ppo_micro_batch_size_per_gpu: int
    log_prob_micro_batch_size_per_gpu: int
    ref_log_prob_micro_batch_size_per_gpu: int
    learning_rate: float
    critic_learning_rate: float
    entropy_coeff: float
    n_gpus_per_node: int
    total_epochs: int
    test_freq: int
    save_freq: int
    project_name: str
    experiment_name: str
    wandb: bool
    val_before_train: bool
    gradient_checkpointing: bool
    param_offload: bool
    optimizer_offload: bool
    ref_param_offload: bool

    @classmethod
    def from_namespace(cls, args: Namespace) -> "VerlLaunchConfig":
        game = args.game
        env_id = resolve_env_id(game, args.env_id or DEFAULT_ENV_IDS[game])
        experiment_name = args.experiment_name or default_experiment_name(game)
        max_steps = default_max_steps(game) if args.max_steps is None else int(args.max_steps)
        reward_player = (
            default_reward_player(game) if args.reward_player is None else int(args.reward_player)
        )
        return cls(
            game=game,
            env_id=env_id,
            model=args.model,
            train_size=int(args.train_size),
            val_size=int(args.val_size),
            train_seed_start=int(args.train_seed_start),
            val_seed_start=int(args.val_seed_start),
            max_steps=max_steps,
            reward_player=reward_player,
            dry_run=bool(args.dry_run),
            print_config=bool(args.print_config),
            train_batch_size=int(args.train_batch_size),
            max_prompt_length=int(args.max_prompt_length),
            rollout_prompt_length=int(args.rollout_prompt_length),
            rollout_response_length=int(args.rollout_response_length),
            rollout_n=int(args.rollout_n),
            tensor_model_parallel_size=int(args.tensor_model_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            rollout_max_model_len=(
                None if args.rollout_max_model_len is None else int(args.rollout_max_model_len)
            ),
            rollout_max_num_batched_tokens=int(args.rollout_max_num_batched_tokens),
            rollout_max_num_seqs=int(args.rollout_max_num_seqs),
            rollout_update_weights_bucket_megabytes=int(args.rollout_update_weights_bucket_megabytes),
            adv_estimator=str(args.adv_estimator),
            ppo_mini_batch_size=int(args.ppo_mini_batch_size),
            ppo_micro_batch_size_per_gpu=int(args.ppo_micro_batch_size_per_gpu),
            critic_ppo_micro_batch_size_per_gpu=int(args.critic_ppo_micro_batch_size_per_gpu),
            log_prob_micro_batch_size_per_gpu=int(args.log_prob_micro_batch_size_per_gpu),
            ref_log_prob_micro_batch_size_per_gpu=int(args.ref_log_prob_micro_batch_size_per_gpu),
            learning_rate=float(args.learning_rate),
            critic_learning_rate=float(args.critic_learning_rate),
            entropy_coeff=float(args.entropy_coeff),
            n_gpus_per_node=int(args.n_gpus_per_node),
            total_epochs=int(args.total_epochs),
            test_freq=int(args.test_freq),
            save_freq=int(args.save_freq),
            project_name=str(args.project_name),
            experiment_name=str(experiment_name),
            wandb=bool(args.wandb),
            val_before_train=bool(args.val_before_train),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            param_offload=bool(args.param_offload),
            optimizer_offload=bool(args.optimizer_offload),
            ref_param_offload=bool(args.ref_param_offload),
        )


@dataclass(frozen=True)
class VerlRunFiles:
    run_dir: Path
    train_path: Path
    val_path: Path
    interaction_config_path: Path


@dataclass(frozen=True)
class VerlRunPlan:
    config: VerlLaunchConfig
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    files: VerlRunFiles
    overrides: list[str]
    interaction_class: str = INTERACTION_CLASS
    reward_function_path: str = REWARD_FUNCTION_PATH

    def to_payload(self) -> dict[str, Any]:
        return {
            "game": self.config.game,
            "env_id": self.config.env_id,
            "adv_estimator": self.config.adv_estimator,
            "critic_enabled": self.config.adv_estimator == "gae",
            "reward_player": self.config.reward_player,
            "max_steps": self.config.max_steps,
            "model": self.config.model,
            "train_size": len(self.train_rows),
            "val_size": len(self.val_rows),
            "train_file": str(self.files.train_path),
            "val_file": str(self.files.val_path),
            "interaction_config": str(self.files.interaction_config_path),
            "interaction_class": self.interaction_class,
            "reward_function_path": self.reward_function_path,
            "overrides": self.overrides,
            "train_example": self.train_rows[0] if self.train_rows else None,
            "val_example": self.val_rows[0] if self.val_rows else None,
        }


def default_experiment_name(game: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{game}-verl-{timestamp}"


def safe_name(value: str) -> str:
    allowed = {"-", "_", "."}
    return "".join(char if char.isalnum() or char in allowed else "-" for char in value)


def resolve_launch_config(args: Namespace) -> VerlLaunchConfig:
    return VerlLaunchConfig.from_namespace(args)


def build_rows(
    *,
    game: GameName,
    count: int,
    seed_start: int,
    env_id: str,
    max_steps: int,
    reward_player: int,
) -> list[dict[str, Any]]:
    return [
        build_dataset_row(
            game=game,
            seed=seed_start + i,
            index=seed_start + i,
            env_id=env_id,
            max_steps=max_steps,
            reward_player=reward_player,
        )
        for i in range(count)
    ]


def effective_val_rows(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if val_rows:
        return val_rows
    if not train_rows:
        return []
    placeholder = json.loads(json.dumps(train_rows[0]))
    placeholder["extra_info"]["index"] = -1
    return [placeholder]


def write_dataset_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_interaction_config(path: Path) -> None:
    payload = (
        "interaction:\n"
        "  - name: mindgames\n"
        f"    class_name: {INTERACTION_CLASS}\n"
        "    config: {}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def build_overrides(
    config: VerlLaunchConfig,
    *,
    train_path: Path,
    val_path: Path,
    interaction_config_path: Path,
    root_dir: Path,
    repo_parent: Path,
) -> list[str]:
    logger_list = "[console,wandb]" if config.wandb else "[console]"
    max_model_len = config.rollout_max_model_len or (
        config.rollout_prompt_length + config.rollout_response_length
    )

    overrides = [
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        "data.truncation=error",
        "data.shuffle=False",
        f"algorithm.adv_estimator={config.adv_estimator}",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={config.model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        (
            "actor_rollout_ref.model.enable_gradient_checkpointing="
            f"{'True' if config.gradient_checkpointing else 'False'}"
        ),
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.prompt_length={config.rollout_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={config.rollout_response_length}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_seqs={config.rollout_max_num_seqs}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={config.rollout_max_num_batched_tokens}",
        (
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
            f"{config.log_prob_micro_batch_size_per_gpu}"
        ),
        (
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="
            f"{config.rollout_update_weights_bucket_megabytes}"
        ),
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={config.max_steps}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={config.max_steps}",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path={interaction_config_path}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        f"actor_rollout_ref.actor.entropy_coeff={config.entropy_coeff}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={'True' if config.param_offload else 'False'}",
        (
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload="
            f"{'True' if config.optimizer_offload else 'False'}"
        ),
        (
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
            f"{config.ref_log_prob_micro_batch_size_per_gpu}"
        ),
        f"actor_rollout_ref.ref.fsdp_config.param_offload={'True' if config.ref_param_offload else 'False'}",
        f"reward.custom_reward_function.path={REWARD_FUNCTION_PATH}",
        "reward.custom_reward_function.name=compute_score",
        f"trainer.project_name={config.project_name}",
        f"trainer.experiment_name={config.experiment_name}",
        f"trainer.logger={logger_list}",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        f"trainer.total_epochs={config.total_epochs}",
        f"trainer.test_freq={config.test_freq}",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.val_before_train={'True' if config.val_before_train else 'False'}",
        "trainer.resume_mode=disable",
        "trainer.critic_warmup=0",
        f"++ray_kwargs.ray_init.runtime_env.working_dir={root_dir}",
        f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={root_dir}:{repo_parent}",
    ]
    if config.adv_estimator == "gae":
        overrides.extend(
            [
                "critic.enable=True",
                f"critic.model.path={config.model}",
                f"critic.model.tokenizer_path={config.model}",
                (
                    "critic.model.enable_gradient_checkpointing="
                    f"{'True' if config.gradient_checkpointing else 'False'}"
                ),
                f"critic.optim.lr={config.critic_learning_rate}",
                (
                    "critic.ppo_micro_batch_size_per_gpu="
                    f"{config.critic_ppo_micro_batch_size_per_gpu}"
                ),
            ]
        )
    else:
        overrides.append("critic.enable=False")
    return overrides


def prepare_run_plan(
    config: VerlLaunchConfig,
    *,
    root_dir: Path,
    repo_parent: Path,
) -> VerlRunPlan:
    train_rows = build_rows(
        game=config.game,
        count=config.train_size,
        seed_start=config.train_seed_start,
        env_id=config.env_id,
        max_steps=config.max_steps,
        reward_player=config.reward_player,
    )
    raw_val_rows = build_rows(
        game=config.game,
        count=config.val_size,
        seed_start=config.val_seed_start,
        env_id=config.env_id,
        max_steps=config.max_steps,
        reward_player=config.reward_player,
    )
    val_rows = effective_val_rows(train_rows, raw_val_rows)

    run_dir = root_dir / "outputs" / "verl_runs" / safe_name(config.experiment_name)
    files = VerlRunFiles(
        run_dir=run_dir,
        train_path=run_dir / "train.jsonl",
        val_path=run_dir / "val.jsonl",
        interaction_config_path=run_dir / "interaction.yaml",
    )
    overrides = build_overrides(
        config,
        train_path=files.train_path,
        val_path=files.val_path,
        interaction_config_path=files.interaction_config_path,
        root_dir=root_dir,
        repo_parent=repo_parent,
    )
    return VerlRunPlan(
        config=config,
        train_rows=train_rows,
        val_rows=val_rows,
        files=files,
        overrides=overrides,
    )


def materialize_run_plan(plan: VerlRunPlan) -> None:
    write_dataset_jsonl(plan.files.train_path, plan.train_rows)
    write_dataset_jsonl(plan.files.val_path, plan.val_rows)
    write_interaction_config(plan.files.interaction_config_path)


def print_run_plan(plan: VerlRunPlan) -> None:
    print(json.dumps(plan.to_payload(), indent=2, sort_keys=True))


def launch_verl(plan: VerlRunPlan, *, root_dir: Path, repo_parent: Path) -> None:
    env = os.environ.copy()
    pythonpath_entries = [str(root_dir), str(repo_parent)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    command = [sys.executable, "-m", "verl.trainer.main_ppo", *plan.overrides]
    subprocess.run(command, check=True, cwd=str(root_dir), env=env)
