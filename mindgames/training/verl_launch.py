from __future__ import annotations

import json
import netrc
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
from mindgames.training.verl_snapshot_agent_loop import SNAPSHOT_AGENT_LOOP_NAME


INTERACTION_CLASS = "mindgames.training.verl_adapter.MindGamesInteraction"
REWARD_FUNCTION_PATH = "pkg://mindgames.training.verl_adapter"
SNAPSHOT_AGENT_LOOP_CLASS = (
    "mindgames.training.verl_snapshot_agent_loop.MindGamesSnapshotEpisodeAgentLoop"
)
TRAINER_MAIN_MODULE = "mindgames.training.verl_main_ppo"
LoraTargetModules = str | tuple[str, ...]


def parse_lora_target_modules(raw_value: str) -> LoraTargetModules:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("LoRA target modules cannot be empty.")
    if value == "all-linear":
        return value
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise ValueError("LoRA target modules cannot be empty.")
    return modules


def format_lora_target_modules(value: LoraTargetModules) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(list(value), separators=(",", ":"))


def payload_lora_target_modules(value: LoraTargetModules) -> str | list[str]:
    if isinstance(value, str):
        return value
    return list(value)


@dataclass(frozen=True)
class VerlLaunchConfig:
    preset: str | None
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
    gpu_memory_utilization: float | None
    rollout_max_model_len: int | None
    rollout_max_num_batched_tokens: int
    rollout_max_num_seqs: int
    rollout_enable_sleep_mode: bool | None
    rollout_enforce_eager: bool | None
    adv_estimator: str
    finetune_mode: str
    lora_rank: int
    lora_alpha: int
    lora_target_modules: LoraTargetModules
    lora_adapter_path: str | None
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    critic_ppo_micro_batch_size_per_gpu: int
    log_prob_micro_batch_size_per_gpu: int
    ref_log_prob_micro_batch_size_per_gpu: int
    learning_rate: float | None
    critic_learning_rate: float | None
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
    enable_thinking: bool | None = None

    def __post_init__(self) -> None:
        if self.finetune_mode == "lora" and self.lora_rank <= 0:
            raise ValueError("LoRA mode requires --lora-rank to be a positive integer.")
        if self.adv_estimator == "grpo":
            raise ValueError(
                "MindGames snapshot-only episode training does not support GRPO. "
                "Use `gae` so every visited episode step trains against the final outcome."
            )
        if isinstance(self.lora_target_modules, str):
            object.__setattr__(
                self,
                "lora_target_modules",
                parse_lora_target_modules(self.lora_target_modules),
            )

    @classmethod
    def from_namespace(cls, args: Namespace) -> "VerlLaunchConfig":
        preset = getattr(args, "preset", None)
        game = args.game
        env_id = resolve_env_id(game, args.env_id or DEFAULT_ENV_IDS[game])
        experiment_name = args.experiment_name or default_experiment_name(game)
        max_steps = default_max_steps(game) if args.max_steps is None else int(args.max_steps)
        reward_player = (
            default_reward_player(game) if args.reward_player is None else int(args.reward_player)
        )
        return cls(
            preset=(None if preset in (None, "") else str(preset)),
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
            gpu_memory_utilization=(
                None
                if getattr(args, "gpu_memory_utilization", None) is None
                else float(args.gpu_memory_utilization)
            ),
            rollout_max_model_len=(
                None if args.rollout_max_model_len is None else int(args.rollout_max_model_len)
            ),
            rollout_max_num_batched_tokens=int(args.rollout_max_num_batched_tokens),
            rollout_max_num_seqs=int(args.rollout_max_num_seqs),
            rollout_enable_sleep_mode=(
                None
                if getattr(args, "rollout_enable_sleep_mode", None) is None
                else bool(args.rollout_enable_sleep_mode)
            ),
            rollout_enforce_eager=(
                None
                if getattr(args, "rollout_enforce_eager", None) is None
                else bool(args.rollout_enforce_eager)
            ),
            adv_estimator=str(args.adv_estimator),
            finetune_mode=str(args.finetune_mode),
            lora_rank=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_target_modules=str(args.lora_target_modules),
            lora_adapter_path=(
                None
                if getattr(args, "lora_adapter_path", None) in (None, "")
                else str(args.lora_adapter_path)
            ),
            ppo_mini_batch_size=int(args.ppo_mini_batch_size),
            ppo_micro_batch_size_per_gpu=int(args.ppo_micro_batch_size_per_gpu),
            critic_ppo_micro_batch_size_per_gpu=int(args.critic_ppo_micro_batch_size_per_gpu),
            log_prob_micro_batch_size_per_gpu=int(args.log_prob_micro_batch_size_per_gpu),
            ref_log_prob_micro_batch_size_per_gpu=int(args.ref_log_prob_micro_batch_size_per_gpu),
            learning_rate=(
                None if getattr(args, "learning_rate", None) is None else float(args.learning_rate)
            ),
            critic_learning_rate=(
                None
                if getattr(args, "critic_learning_rate", None) is None
                else float(args.critic_learning_rate)
            ),
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
            enable_thinking=(
                None
                if getattr(args, "enable_thinking", None) is None
                else bool(args.enable_thinking)
            ),
        )


@dataclass(frozen=True)
class VerlRunFiles:
    run_dir: Path
    train_path: Path
    val_path: Path
    agent_loop_config_path: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class VerlRunPlan:
    config: VerlLaunchConfig
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    files: VerlRunFiles
    overrides: list[str]
    agent_loop_name: str = SNAPSHOT_AGENT_LOOP_NAME
    agent_loop_class: str = SNAPSHOT_AGENT_LOOP_CLASS

    def to_payload(self, *, redact_sensitive: bool = False) -> dict[str, Any]:
        overrides = (
            [redact_sensitive_override(override) for override in self.overrides]
            if redact_sensitive
            else self.overrides
        )
        return {
            "preset": self.config.preset,
            "game": self.config.game,
            "env_id": self.config.env_id,
            "adv_estimator": self.config.adv_estimator,
            "finetune_mode": self.config.finetune_mode,
            "critic_enabled": self.config.adv_estimator == "gae",
            "lora_enabled": self.config.finetune_mode == "lora",
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_target_modules": payload_lora_target_modules(self.config.lora_target_modules),
            "lora_adapter_path": self.config.lora_adapter_path,
            "reward_player": self.config.reward_player,
            "max_steps": self.config.max_steps,
            "model": self.config.model,
            "enable_thinking": self.config.enable_thinking,
            "rollout_enable_sleep_mode": self.config.rollout_enable_sleep_mode,
            "rollout_enforce_eager": self.config.rollout_enforce_eager,
            "train_size": len(self.train_rows),
            "val_size": len(self.val_rows),
            "train_file": str(self.files.train_path),
            "val_file": str(self.files.val_path),
            "checkpoint_dir": str(self.files.checkpoint_dir),
            "agent_loop_name": self.agent_loop_name,
            "agent_loop_class": self.agent_loop_class,
            "agent_loop_config": str(self.files.agent_loop_config_path),
            "overrides": overrides,
            "train_example": self.train_rows[0] if self.train_rows else None,
            "val_example": self.val_rows[0] if self.val_rows else None,
        }


def default_experiment_name(game: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{game}-verl-{timestamp}"


def safe_name(value: str) -> str:
    allowed = {"-", "_", "."}
    return "".join(char if char.isalnum() or char in allowed else "-" for char in value)


def default_nccl_env_vars() -> dict[str, str]:
    return {
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "1"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "1"),
    }


def redact_sensitive_override(override: str) -> str:
    marker = "env_vars.WANDB_API_KEY="
    if marker not in override:
        return override
    prefix, _separator, _value = override.partition("=")
    return f'{prefix}="<redacted>"'


def resolve_wandb_api_key() -> str | None:
    env_value = os.environ.get("WANDB_API_KEY")
    if env_value:
        return env_value
    try:
        credentials = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return None
    auth = credentials.authenticators("api.wandb.ai")
    if auth is None:
        return None
    _login, _account, password = auth
    return password or None


def passthrough_runtime_env_vars() -> dict[str, str]:
    env_vars = dict(default_nccl_env_vars())
    wandb_api_key = resolve_wandb_api_key()
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
    for key in ("WANDB_ENTITY", "WANDB_BASE_URL", "WANDB_MODE", "WANDB_DIR", "WANDB_PROJECT"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


def checkpoint_dir_for_run(*, root_dir: Path, project_name: str, experiment_name: str) -> Path:
    return root_dir / "checkpoints" / safe_name(project_name) / safe_name(experiment_name)


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


def write_agent_loop_config(path: Path) -> None:
    payload = (
        f"- name: {SNAPSHOT_AGENT_LOOP_NAME}\n"
        f"  _target_: {SNAPSHOT_AGENT_LOOP_CLASS}\n"
        "  selection_strategy: uniform\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def build_overrides(
    config: VerlLaunchConfig,
    *,
    train_path: Path,
    val_path: Path,
    agent_loop_config_path: Path,
    checkpoint_dir: Path,
    root_dir: Path,
    repo_parent: Path,
) -> list[str]:
    logger_list = "[console,wandb]" if config.wandb else "[console]"
    max_model_len = config.rollout_max_model_len or (
        config.rollout_prompt_length + config.rollout_response_length
    )
    lora_target_modules = format_lora_target_modules(config.lora_target_modules)
    runtime_env_vars = passthrough_runtime_env_vars()

    overrides = [
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        "data.shuffle=False",
        f"actor_rollout_ref.model.path={config.model}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_seqs={config.rollout_max_num_seqs}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={config.rollout_max_num_batched_tokens}",
        (
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
            f"{config.log_prob_micro_batch_size_per_gpu}"
        ),
        "actor_rollout_ref.rollout.multi_turn.enable=False",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_config_path}",
        f"actor_rollout_ref.rollout.agent.default_agent_loop={SNAPSHOT_AGENT_LOOP_NAME}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.entropy_coeff={config.entropy_coeff}",
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
        f"trainer.project_name={config.project_name}",
        f"trainer.experiment_name={config.experiment_name}",
        f"trainer.logger={logger_list}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        f"trainer.total_epochs={config.total_epochs}",
        f"trainer.test_freq={config.test_freq}",
        f"trainer.save_freq={config.save_freq}",
        "trainer.resume_mode=disable",
        f"++ray_kwargs.ray_init.runtime_env.working_dir={root_dir}",
        f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={root_dir}:{repo_parent}",
    ]
    if config.enable_thinking is not None:
        overrides.append(
            "++data.apply_chat_template_kwargs.enable_thinking="
            f"{'true' if config.enable_thinking else 'false'}"
        )
    for key, value in runtime_env_vars.items():
        overrides.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(str(value))}")
    if config.adv_estimator != "gae":
        overrides.append(f"algorithm.adv_estimator={config.adv_estimator}")
    if config.gradient_checkpointing is not True:
        overrides.append(
            "actor_rollout_ref.model.enable_gradient_checkpointing="
            f"{'True' if config.gradient_checkpointing else 'False'}"
        )
    if config.rollout_n != 1:
        overrides.append(f"actor_rollout_ref.rollout.n={config.rollout_n}")
    if config.gpu_memory_utilization is not None:
        overrides.append(
            f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}"
        )
    if config.rollout_enable_sleep_mode is not None:
        overrides.append(
            "+actor_rollout_ref.rollout.enable_sleep_mode="
            f"{'True' if config.rollout_enable_sleep_mode else 'False'}"
        )
    if config.rollout_enforce_eager is not None:
        overrides.append(
            "actor_rollout_ref.rollout.enforce_eager="
            f"{'True' if config.rollout_enforce_eager else 'False'}"
        )
    if config.rollout_prompt_length != config.max_prompt_length:
        overrides.append(f"actor_rollout_ref.rollout.prompt_length={config.rollout_prompt_length}")
    if config.rollout_response_length != 512:
        overrides.append(f"actor_rollout_ref.rollout.response_length={config.rollout_response_length}")
    if config.val_before_train is not True:
        overrides.append(f"trainer.val_before_train={'True' if config.val_before_train else 'False'}")
    if config.finetune_mode == "lora":
        overrides.extend(
            [
                f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
                "actor_rollout_ref.rollout.load_format=safetensors",
            ]
        )
        if config.lora_target_modules != "all-linear":
            overrides.append(f"actor_rollout_ref.model.target_modules={lora_target_modules}")
        if config.lora_adapter_path is not None:
            overrides.append(f"actor_rollout_ref.model.lora_adapter_path={config.lora_adapter_path}")
    if config.learning_rate is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.lr={config.learning_rate}")
    if config.adv_estimator == "gae":
        overrides.extend(
            [
                f"critic.model.path={config.model}",
                (
                    "critic.ppo_micro_batch_size_per_gpu="
                    f"{config.critic_ppo_micro_batch_size_per_gpu}"
                ),
                f"critic.model.fsdp_config.param_offload={'True' if config.param_offload else 'False'}",
                (
                    "critic.model.fsdp_config.optimizer_offload="
                    f"{'True' if config.optimizer_offload else 'False'}"
                ),
            ]
        )
        if config.gradient_checkpointing is not True:
            overrides.append(
                "critic.model.enable_gradient_checkpointing="
                f"{'True' if config.gradient_checkpointing else 'False'}"
            )
        if config.critic_learning_rate is not None:
            overrides.append(f"critic.optim.lr={config.critic_learning_rate}")
        if config.finetune_mode == "lora":
            overrides.extend(
                [
                    f"critic.model.lora_rank={config.lora_rank}",
                    f"critic.model.lora_alpha={config.lora_alpha}",
                ]
            )
            if config.lora_target_modules != "all-linear":
                overrides.append(f"critic.model.target_modules={lora_target_modules}")
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
    checkpoint_dir = checkpoint_dir_for_run(
        root_dir=root_dir,
        project_name=config.project_name,
        experiment_name=config.experiment_name,
    )
    files = VerlRunFiles(
        run_dir=run_dir,
        train_path=run_dir / "train.jsonl",
        val_path=run_dir / "val.jsonl",
        agent_loop_config_path=run_dir / "agent_loop.yaml",
        checkpoint_dir=checkpoint_dir,
    )
    overrides = build_overrides(
        config,
        train_path=files.train_path,
        val_path=files.val_path,
        agent_loop_config_path=files.agent_loop_config_path,
        checkpoint_dir=files.checkpoint_dir,
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
    write_agent_loop_config(plan.files.agent_loop_config_path)
    plan.files.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def print_run_plan(plan: VerlRunPlan) -> None:
    print(json.dumps(plan.to_payload(redact_sensitive=True), indent=2, sort_keys=True))


def launch_verl(plan: VerlRunPlan, *, root_dir: Path, repo_parent: Path) -> None:
    env = os.environ.copy()
    pythonpath_entries = [str(root_dir), str(repo_parent)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    for key, value in passthrough_runtime_env_vars().items():
        env.setdefault(key, value)

    command = [sys.executable, "-m", TRAINER_MAIN_MODULE, *plan.overrides]
    subprocess.run(command, check=True, cwd=str(root_dir), env=env)
