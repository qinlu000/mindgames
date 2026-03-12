#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "mindgames":
            return parent
    raise RuntimeError("Could not locate mindgames project root.")


def _ensure_pkg_importable() -> Path:
    project_root = _find_project_root()
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))
    return project_root


PROJECT_ROOT = _ensure_pkg_importable()

import mindgames as mg  # noqa: E402


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are Player {player_id} in a {num_players}-player cooperative Hanabi game.
Output EXACTLY ONE valid action and nothing else.

Valid formats:
- [Play] X
- [Discard] X
- [Reveal] player N card X color C
- [Reveal] player N card X rank R

Rules:
- Reveal must target exactly one specific card index in another player's hand.
- Reveal must be truthful for that specific card.
- Do not reveal about yourself.
- Use exactly one hint type: color OR rank.
- Fireworks are independent; you may play the next required rank of any color.

Strategy priority:
1) If you know a card is playable, [Play] it.
2) Else if a teammate has a clearly playable card and info_tokens>0, reveal that exact card.
3) Else discard the least useful or most uncertain card.
4) Avoid repeating the same reveal on the same card unless it adds new information.
"""


@dataclass(frozen=True)
class RuntimeConfig:
    agent_kind: str
    system_prompt_template: str
    request_timeout_s: float
    max_retries: int
    retry_delay_s: float
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    enable_thinking: Optional[bool]
    env_kwargs: Dict[str, Any]
    reward_scale: float
    reward_mode: str


RUNTIME_CONFIG: RuntimeConfig | None = None


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw == "":
        return None
    return float(raw)


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean string, received: {raw!r}")


def _parse_logger_list(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["console"]


def _parse_target_modules(raw: str) -> str | List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return "all-linear"
    if len(values) == 1:
        return values[0]
    return values


def _default_experiment_name() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"hanabi_verl_{ts}"


def _default_output_dir(project_name: str, experiment_name: str) -> str:
    return str(PROJECT_ROOT / "checkpoints" / project_name / experiment_name)


def _load_text_file(path: Optional[str], *, fallback: str) -> str:
    if not path:
        return fallback
    return Path(path).read_text(encoding="utf-8")


def _load_task_file(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                raise ValueError(f"Task line {line_no} in {path} is not a JSON object.")
            tasks.append(rec)
    return tasks


def _build_tasks(
    *,
    prefix: str,
    count: int,
    base_seed: int,
    env_id: str,
    num_players: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"{prefix}-{idx:06d}",
            "seed": base_seed + idx,
            "env_id": env_id,
            "num_players": num_players,
        }
        for idx in range(max(count, 0))
    ]


def _load_tasks(
    *,
    task_file: Optional[str],
    prefix: str,
    count: int,
    base_seed: int,
    env_id: str,
    num_players: int,
) -> List[Dict[str, Any]]:
    if task_file:
        return _load_task_file(Path(task_file))
    return _build_tasks(
        prefix=prefix,
        count=count,
        base_seed=base_seed,
        env_id=env_id,
        num_players=num_players,
    )


def _make_env(env_id: str, env_kwargs: Dict[str, Any]) -> mg.Env:
    if env_id not in mg.ENV_REGISTRY:
        raise ValueError(f"env_id={env_id!r} is not registered in mindgames.")
    return mg.make(env_id=env_id, **env_kwargs)


def _sampling_value(name: str, explicit: Any, default: Any, llm: Any) -> Any:
    if explicit is not None:
        return explicit
    sampling_parameters = getattr(llm, "sampling_parameters", None)
    if isinstance(sampling_parameters, dict) and name in sampling_parameters:
        return sampling_parameters[name]
    return default


def _make_agent(*, llm: Any, player_id: int, num_players: int) -> mg.Agent:
    if RUNTIME_CONFIG is None:
        raise RuntimeError("Runtime config has not been initialized.")

    system_prompt = RUNTIME_CONFIG.system_prompt_template.format(
        player_id=player_id,
        num_players=num_players,
    )
    agent_kwargs: Dict[str, Any] = {
        "model_name": getattr(llm, "model"),
        "system_prompt": system_prompt,
        "api_key": getattr(llm, "api_key", None) or "EMPTY",
        "base_url": getattr(llm, "endpoint"),
        "max_retries": RUNTIME_CONFIG.max_retries,
        "retry_delay_s": RUNTIME_CONFIG.retry_delay_s,
        "timeout": RUNTIME_CONFIG.request_timeout_s,
        "temperature": _sampling_value("temperature", RUNTIME_CONFIG.temperature, 0.7, llm),
        "top_p": _sampling_value("top_p", RUNTIME_CONFIG.top_p, 0.95, llm),
        "max_tokens": _sampling_value("max_tokens", RUNTIME_CONFIG.max_tokens, 128, llm),
    }

    if RUNTIME_CONFIG.agent_kind == "qwen":
        agent_kwargs["enable_thinking"] = RUNTIME_CONFIG.enable_thinking
        return mg.agents.QwenAgent(**agent_kwargs)
    if RUNTIME_CONFIG.agent_kind == "openai":
        return mg.agents.OpenAIAgent(**agent_kwargs)
    raise ValueError(f"Unsupported --agent-kind={RUNTIME_CONFIG.agent_kind!r}. Expected one of: qwen, openai.")


def _score_from_rewards(rewards: Any) -> float:
    if isinstance(rewards, dict) and rewards:
        vals = [float(v) for v in rewards.values()]
        return sum(vals) / len(vals)
    if isinstance(rewards, (int, float)):
        return float(rewards)
    return 0.0


def _normalize_reward(score: float) -> float:
    if RUNTIME_CONFIG is None:
        raise RuntimeError("Runtime config has not been initialized.")
    scale = max(float(RUNTIME_CONFIG.reward_scale), 1.0)
    return float(score) / scale


def _select_episode_reward(*, final_score: float, dense_return: float, saw_dense_reward: bool) -> float:
    if RUNTIME_CONFIG is None:
        raise RuntimeError("Runtime config has not been initialized.")

    reward_mode = RUNTIME_CONFIG.reward_mode
    if reward_mode == "score":
        return final_score
    if reward_mode == "episode_return":
        return dense_return if saw_dense_reward else final_score
    if reward_mode == "auto":
        return dense_return if saw_dense_reward else final_score
    raise ValueError(f"Unsupported reward_mode={reward_mode!r}")


def _build_agent(agl: Any):
    @agl.rollout
    def hanabi_weight_rollout(task: Dict[str, Any], llm: Any) -> float:
        env_id = str(task.get("env_id") or "Hanabi-v0-train")
        num_players = int(task.get("num_players") or 2)
        seed = int(task.get("seed") or 0)

        env = _make_env(env_id=env_id, env_kwargs=RUNTIME_CONFIG.env_kwargs if RUNTIME_CONFIG else {})
        env.reset(num_players=num_players, seed=seed)

        agents: Dict[int, mg.Agent] = {
            pid: _make_agent(llm=llm, player_id=pid, num_players=num_players) for pid in range(num_players)
        }

        dense_return = 0.0
        saw_dense_reward = False
        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = agents[player_id](observation)
            done, info = env.step(action=action)
            if isinstance(info, dict) and "step_reward" in info:
                dense_return += float(info["step_reward"])
                saw_dense_reward = True

        rewards, _ = env.close()
        final_score = _score_from_rewards(rewards)
        reward = _select_episode_reward(
            final_score=final_score,
            dense_return=dense_return,
            saw_dense_reward=saw_dense_reward,
        )
        return _normalize_reward(reward)

    return hanabi_weight_rollout


def _build_verl_config(
    args: argparse.Namespace,
    *,
    train_dataset_size: int,
    has_validation: bool,
) -> Dict[str, Any]:
    train_batch_size = min(max(int(args.train_batch_size), 1), max(train_dataset_size, 1))
    max_model_len = args.max_model_len or (int(args.max_prompt_length) + int(args.max_response_length))

    actor_cfg: Dict[str, Any] = {
        "ppo_mini_batch_size": int(args.ppo_mini_batch_size),
        "ppo_micro_batch_size_per_gpu": int(args.ppo_micro_batch_size_per_gpu),
        "optim": {"lr": float(args.lr)},
        "use_kl_loss": _parse_bool(args.use_kl_loss),
        "kl_loss_coef": float(args.kl_loss_coef),
        "entropy_coeff": float(args.entropy_coeff),
        "fsdp_config": {
            "param_offload": _parse_bool(args.param_offload),
            "optimizer_offload": _parse_bool(args.optimizer_offload),
        },
    }
    ref_cfg: Dict[str, Any] = {
        "log_prob_micro_batch_size_per_gpu": int(args.ref_log_prob_micro_batch_size_per_gpu),
        "fsdp_config": {"param_offload": _parse_bool(args.param_offload)},
    }
    critic_cfg: Dict[str, Any] = {
        "optim": {"lr": float(args.critic_lr)},
        "model": {
            "path": args.model,
            "tokenizer_path": args.model,
            "trust_remote_code": _parse_bool(args.trust_remote_code),
            "enable_gradient_checkpointing": True,
            "use_remove_padding": True,
            "fsdp_config": {
                "param_offload": _parse_bool(args.critic_param_offload),
                "optimizer_offload": _parse_bool(args.critic_optimizer_offload),
            },
            "lora_rank": int(args.critic_lora_rank),
            "lora_alpha": int(args.critic_lora_alpha),
            "target_modules": _parse_target_modules(args.critic_target_modules),
        },
        "ppo_mini_batch_size": int(args.ppo_mini_batch_size),
        "ppo_micro_batch_size_per_gpu": int(args.critic_ppo_micro_batch_size_per_gpu),
        "forward_micro_batch_size_per_gpu": int(args.critic_forward_micro_batch_size_per_gpu),
        "ppo_max_token_len_per_gpu": int(args.critic_ppo_max_token_len_per_gpu),
    }
    model_cfg: Dict[str, Any] = {
        "path": args.model,
        "use_remove_padding": True,
        "enable_gradient_checkpointing": True,
        "trust_remote_code": _parse_bool(args.trust_remote_code),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "target_modules": _parse_target_modules(args.target_modules),
    }

    config: Dict[str, Any] = {
        "algorithm": {
            "adv_estimator": args.adv_estimator,
            "gamma": float(args.gamma),
            "lam": float(args.lam),
            "use_kl_in_reward": _parse_bool(args.use_kl_in_reward),
            "kl_penalty": args.kl_penalty,
        },
        "data": {
            "train_batch_size": train_batch_size,
            "val_batch_size": None,
            "max_prompt_length": int(args.max_prompt_length),
            "max_response_length": int(args.max_response_length),
            "filter_overlong_prompts": False,
        },
        "actor_rollout_ref": {
            "model": model_cfg,
            "actor": actor_cfg,
            "ref": ref_cfg,
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "n": int(args.rollout_n),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "prompt_length": int(args.max_prompt_length),
                "response_length": int(args.max_response_length),
                "max_model_len": int(max_model_len),
                "tensor_model_parallel_size": int(args.tensor_model_parallel_size),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "max_num_batched_tokens": int(args.max_num_batched_tokens),
                "max_num_seqs": int(args.max_num_seqs),
                "log_prob_micro_batch_size_per_gpu": int(args.rollout_log_prob_micro_batch_size_per_gpu),
                "val_kwargs": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "n": 1,
                    "do_sample": False,
                },
            },
        },
        "critic": critic_cfg,
        "trainer": {
            "project_name": args.project_name,
            "experiment_name": args.experiment_name,
            "logger": _parse_logger_list(args.logger),
            "nnodes": int(args.nnodes),
            "n_gpus_per_node": int(args.n_gpus_per_node),
            "save_freq": int(args.save_freq),
            "test_freq": int(args.test_freq) if has_validation else -1,
            "total_epochs": int(args.total_epochs),
            "critic_warmup": 0,
            "val_before_train": _parse_bool(args.val_before_train) if has_validation else False,
            "val_only": _parse_bool(args.val_only),
            "resume_mode": args.resume_mode,
            "resume_from_path": args.resume_from_path,
            "default_local_dir": args.output_dir,
        },
    }
    if args.ray_num_cpus is not None:
        config["ray_init"] = {"num_cpus": int(args.ray_num_cpus)}
    return config


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train Hanabi model weights with Agent Lightning VERL on top of mindgames self-play."
    )
    ap.add_argument("--model", default=os.getenv("MODEL", "/workspace/models/Qwen3-8B"))
    ap.add_argument("--agent-kind", choices=("qwen", "openai"), default=os.getenv("AGENT_KIND", "qwen"))
    ap.add_argument("--env-id", default=os.getenv("ENV_ID", "Hanabi-v0-train"))
    ap.add_argument("--num-players", type=int, default=int(os.getenv("NUM_PLAYERS", "2")))
    ap.add_argument("--train-episodes", type=int, default=int(os.getenv("TRAIN_EPISODES", "64")))
    ap.add_argument("--val-episodes", type=int, default=int(os.getenv("VAL_EPISODES", "16")))
    ap.add_argument("--train-seed", type=int, default=int(os.getenv("TRAIN_SEED", "0")))
    ap.add_argument("--val-seed", type=int, default=int(os.getenv("VAL_SEED", "100000")))
    ap.add_argument("--train-task-file", default=os.getenv("TRAIN_TASK_FILE"))
    ap.add_argument("--val-task-file", default=os.getenv("VAL_TASK_FILE"))
    ap.add_argument("--env-kwargs", default=os.getenv("ENV_KWARGS", "{}"))
    ap.add_argument("--system-prompt-file", default=os.getenv("SYSTEM_PROMPT_FILE"))
    ap.add_argument("--temperature", default=os.getenv("TEMPERATURE", "0.7"))
    ap.add_argument("--top-p", default=os.getenv("TOP_P", "0.95"))
    ap.add_argument("--max-tokens", default=os.getenv("MAX_TOKENS", "128"))
    ap.add_argument("--request-timeout-s", type=float, default=float(os.getenv("REQUEST_TIMEOUT_S", "120")))
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    ap.add_argument("--retry-delay-s", type=float, default=float(os.getenv("RETRY_DELAY_S", "0.5")))
    ap.add_argument("--enable-thinking", choices=("true", "false", "auto"), default=os.getenv("ENABLE_THINKING", "false"))
    ap.add_argument("--reward-scale", type=float, default=float(os.getenv("REWARD_SCALE", "25")))
    ap.add_argument("--reward-mode", choices=("auto", "score", "episode_return"), default=os.getenv("REWARD_MODE", "auto"))

    ap.add_argument("--adv-estimator", default=os.getenv("ADV_ESTIMATOR", "grpo"))
    ap.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "1.0")))
    ap.add_argument("--lam", type=float, default=float(os.getenv("LAM", "1.0")))
    ap.add_argument("--use-kl-loss", choices=("true", "false"), default=os.getenv("USE_KL_LOSS", "false"))
    ap.add_argument("--kl-loss-coef", type=float, default=float(os.getenv("KL_LOSS_COEF", "0.0")))
    ap.add_argument("--use-kl-in-reward", choices=("true", "false"), default=os.getenv("USE_KL_IN_REWARD", "false"))
    ap.add_argument("--kl-penalty", default=os.getenv("KL_PENALTY", "kl"))
    ap.add_argument("--entropy-coeff", type=float, default=float(os.getenv("ENTROPY_COEFF", "0.0")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-6")))
    ap.add_argument("--critic-lr", type=float, default=float(os.getenv("CRITIC_LR", "5e-6")))

    ap.add_argument("--rollout-n", type=int, default=int(os.getenv("ROLLOUT_N", "4")))
    ap.add_argument("--train-batch-size", type=int, default=int(os.getenv("TRAIN_BATCH_SIZE", "8")))
    ap.add_argument("--max-prompt-length", type=int, default=int(os.getenv("MAX_PROMPT_LENGTH", "4096")))
    ap.add_argument("--max-response-length", type=int, default=int(os.getenv("MAX_RESPONSE_LENGTH", "128")))
    ap.add_argument("--max-model-len", type=int, default=_parse_optional_int(os.getenv("MAX_MODEL_LEN")))
    ap.add_argument("--max-num-batched-tokens", type=int, default=int(os.getenv("MAX_NUM_BATCHED_TOKENS", "8192")))
    ap.add_argument("--max-num-seqs", type=int, default=int(os.getenv("MAX_NUM_SEQS", "256")))
    ap.add_argument(
        "--rollout-log-prob-micro-batch-size-per-gpu",
        type=int,
        default=int(os.getenv("ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4")),
    )
    ap.add_argument(
        "--ref-log-prob-micro-batch-size-per-gpu",
        type=int,
        default=int(os.getenv("REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "4")),
    )
    ap.add_argument("--ppo-mini-batch-size", type=int, default=int(os.getenv("PPO_MINI_BATCH_SIZE", "32")))
    ap.add_argument(
        "--ppo-micro-batch-size-per-gpu",
        type=int,
        default=int(os.getenv("PPO_MICRO_BATCH_SIZE_PER_GPU", "4")),
    )

    ap.add_argument("--lora-rank", type=int, default=int(os.getenv("LORA_RANK", "0")))
    ap.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "16")))
    ap.add_argument("--target-modules", default=os.getenv("TARGET_MODULES", "all-linear"))
    ap.add_argument("--critic-lora-rank", type=int, default=int(os.getenv("CRITIC_LORA_RANK", "0")))
    ap.add_argument("--critic-lora-alpha", type=int, default=int(os.getenv("CRITIC_LORA_ALPHA", "16")))
    ap.add_argument("--critic-target-modules", default=os.getenv("CRITIC_TARGET_MODULES", "all-linear"))
    ap.add_argument("--param-offload", choices=("true", "false"), default=os.getenv("PARAM_OFFLOAD", "false"))
    ap.add_argument(
        "--optimizer-offload",
        choices=("true", "false"),
        default=os.getenv("OPTIMIZER_OFFLOAD", "false"),
    )
    ap.add_argument(
        "--critic-param-offload",
        choices=("true", "false"),
        default=os.getenv("CRITIC_PARAM_OFFLOAD", os.getenv("PARAM_OFFLOAD", "false")),
    )
    ap.add_argument(
        "--critic-optimizer-offload",
        choices=("true", "false"),
        default=os.getenv("CRITIC_OPTIMIZER_OFFLOAD", os.getenv("OPTIMIZER_OFFLOAD", "false")),
    )
    ap.add_argument("--trust-remote-code", choices=("true", "false"), default=os.getenv("TRUST_REMOTE_CODE", "false"))
    ap.add_argument(
        "--critic-ppo-micro-batch-size-per-gpu",
        type=int,
        default=int(os.getenv("CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU", os.getenv("PPO_MICRO_BATCH_SIZE_PER_GPU", "4"))),
    )
    ap.add_argument(
        "--critic-forward-micro-batch-size-per-gpu",
        type=int,
        default=int(
            os.getenv(
                "CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU",
                os.getenv("CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU", os.getenv("PPO_MICRO_BATCH_SIZE_PER_GPU", "4")),
            )
        ),
    )
    ap.add_argument(
        "--critic-ppo-max-token-len-per-gpu",
        type=int,
        default=int(os.getenv("CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU", "16384")),
    )

    ap.add_argument("--n-runners", type=int, default=int(os.getenv("N_RUNNERS", "1")))
    ap.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "1")))
    ap.add_argument("--n-gpus-per-node", type=int, default=int(os.getenv("N_GPUS_PER_NODE", "1")))
    ap.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=int(os.getenv("TENSOR_MODEL_PARALLEL_SIZE", "1")),
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.6")),
    )
    ap.add_argument("--ray-num-cpus", type=int, default=_parse_optional_int(os.getenv("RAY_NUM_CPUS")))

    ap.add_argument("--total-epochs", type=int, default=int(os.getenv("TOTAL_EPOCHS", "1")))
    ap.add_argument("--save-freq", type=int, default=int(os.getenv("SAVE_FREQ", "-1")))
    ap.add_argument("--test-freq", type=int, default=int(os.getenv("TEST_FREQ", "-1")))
    ap.add_argument("--val-before-train", choices=("true", "false"), default=os.getenv("VAL_BEFORE_TRAIN", "true"))
    ap.add_argument("--val-only", choices=("true", "false"), default=os.getenv("VAL_ONLY", "false"))
    ap.add_argument("--resume-mode", default=os.getenv("RESUME_MODE", "disable"))
    ap.add_argument("--resume-from-path", default=os.getenv("RESUME_FROM_PATH"))

    ap.add_argument("--project-name", default=os.getenv("PROJECT_NAME", "agent_lightning_hanabi"))
    ap.add_argument("--experiment-name", default=os.getenv("EXPERIMENT_NAME", _default_experiment_name()))
    ap.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR"))
    ap.add_argument("--logger", default=os.getenv("LOGGER", "console"))

    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_players < 2 or args.num_players > 5:
        raise SystemExit("--num-players must be between 2 and 5 for Hanabi.")
    if args.rollout_n < 1:
        raise SystemExit("--rollout-n must be >= 1.")
    if args.train_batch_size < 1:
        raise SystemExit("--train-batch-size must be >= 1.")
    if args.n_runners < 1:
        raise SystemExit("--n-runners must be >= 1.")
    if args.n_gpus_per_node < 1:
        raise SystemExit("--n-gpus-per-node must be >= 1.")
    if args.tensor_model_parallel_size < 1:
        raise SystemExit("--tensor-model-parallel-size must be >= 1.")
    if args.total_epochs < 1:
        raise SystemExit("--total-epochs must be >= 1.")
    if args.max_prompt_length < 1 or args.max_response_length < 1:
        raise SystemExit("--max-prompt-length and --max-response-length must be >= 1.")


def _prepare_runtime_config(args: argparse.Namespace) -> None:
    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--env-kwargs must be valid JSON. Received: {args.env_kwargs!r}") from exc
    if not isinstance(env_kwargs, dict):
        raise SystemExit("--env-kwargs must decode to a JSON object.")

    enable_thinking: Optional[bool]
    if args.enable_thinking == "auto":
        enable_thinking = None
    else:
        enable_thinking = args.enable_thinking == "true"

    global RUNTIME_CONFIG
    RUNTIME_CONFIG = RuntimeConfig(
        agent_kind=args.agent_kind,
        system_prompt_template=_load_text_file(
            args.system_prompt_file,
            fallback=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        ),
        request_timeout_s=float(args.request_timeout_s),
        max_retries=int(args.max_retries),
        retry_delay_s=float(args.retry_delay_s),
        temperature=_parse_optional_float(args.temperature),
        top_p=_parse_optional_float(args.top_p),
        max_tokens=_parse_optional_int(args.max_tokens),
        enable_thinking=enable_thinking,
        env_kwargs=env_kwargs,
        reward_scale=float(args.reward_scale),
        reward_mode=args.reward_mode,
    )


def _print_run_summary(
    *,
    train_dataset: Sequence[Dict[str, Any]],
    val_dataset: Optional[Sequence[Dict[str, Any]]],
    config: Dict[str, Any],
) -> None:
    summary = {
        "train_episodes": len(train_dataset),
        "val_episodes": 0 if val_dataset is None else len(val_dataset),
        "output_dir": config["trainer"]["default_local_dir"],
        "project_name": config["trainer"]["project_name"],
        "experiment_name": config["trainer"]["experiment_name"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    _prepare_runtime_config(args)

    if not args.output_dir:
        args.output_dir = _default_output_dir(args.project_name, args.experiment_name)

    train_dataset = _load_tasks(
        task_file=args.train_task_file,
        prefix="train",
        count=int(args.train_episodes),
        base_seed=int(args.train_seed),
        env_id=args.env_id,
        num_players=int(args.num_players),
    )
    val_dataset = _load_tasks(
        task_file=args.val_task_file,
        prefix="val",
        count=int(args.val_episodes),
        base_seed=int(args.val_seed),
        env_id=args.env_id,
        num_players=int(args.num_players),
    )

    if not train_dataset:
        raise SystemExit("Train dataset is empty.")
    if not val_dataset:
        val_dataset = None
        args.val_before_train = "false"
        args.val_only = "false"
        args.test_freq = -1

    try:
        import agentlightning as agl
    except ImportError as exc:
        raise SystemExit(
            "agentlightning is not installed in the selected env. Create `.venv-agent-lightning-verl` first."
        ) from exc

    config = _build_verl_config(
        args,
        train_dataset_size=len(train_dataset),
        has_validation=val_dataset is not None,
    )
    agent = _build_agent(agl)
    algorithm = agl.VERL(config=config)
    trainer = agl.Trainer(
        n_runners=int(args.n_runners),
        algorithm=algorithm,
        adapter=agl.LlmProxyTraceToTriplet(),
    )

    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
    _print_run_summary(train_dataset=train_dataset, val_dataset=val_dataset, config=config)

    if args.dry_run:
        return 0

    trainer.fit(agent, train_dataset, val_dataset=val_dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
