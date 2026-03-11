#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


DEFAULT_PROMPT_TEMPLATE = """You are Player {player_id} in a {num_players}-player cooperative Hanabi game.
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
    model: str
    agent_kind: str
    api_key: Optional[str]
    base_url: Optional[str]
    request_timeout_s: float
    max_retries: int
    retry_delay_s: float
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    enable_thinking: Optional[bool]
    env_kwargs: Dict[str, Any]
    reward_scale: float


RUNTIME_CONFIG: RuntimeConfig | None = None


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw == "":
        return None
    return float(raw)


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    return int(raw)


def _load_prompt_template(path: Optional[str]) -> str:
    if not path:
        return DEFAULT_PROMPT_TEMPLATE
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
        for idx in range(count)
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


def _make_agent(*, prompt: str) -> mg.Agent:
    if RUNTIME_CONFIG is None:
        raise RuntimeError("Runtime config has not been initialized.")

    agent_kwargs: Dict[str, Any] = {
        "model_name": RUNTIME_CONFIG.model,
        "system_prompt": prompt,
        "api_key": RUNTIME_CONFIG.api_key,
        "base_url": RUNTIME_CONFIG.base_url,
        "max_retries": RUNTIME_CONFIG.max_retries,
        "retry_delay_s": RUNTIME_CONFIG.retry_delay_s,
        "timeout": RUNTIME_CONFIG.request_timeout_s,
    }
    if RUNTIME_CONFIG.temperature is not None:
        agent_kwargs["temperature"] = RUNTIME_CONFIG.temperature
    if RUNTIME_CONFIG.top_p is not None:
        agent_kwargs["top_p"] = RUNTIME_CONFIG.top_p
    if RUNTIME_CONFIG.max_tokens is not None:
        agent_kwargs["max_tokens"] = RUNTIME_CONFIG.max_tokens

    kind = RUNTIME_CONFIG.agent_kind
    if kind == "qwen":
        agent_kwargs["enable_thinking"] = bool(RUNTIME_CONFIG.enable_thinking)
        return mg.agents.QwenAgent(**agent_kwargs)
    if kind == "openai":
        return mg.agents.OpenAIAgent(**agent_kwargs)
    raise ValueError(f"Unsupported --agent-kind={kind!r}. Expected one of: qwen, openai.")


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
    return score / scale


def _build_rollout_fn(agl: Any):
    @agl.rollout
    def hanabi_episode_rollout(task: Dict[str, Any], prompt_template: Any) -> float:
        env_id = str(task.get("env_id") or "Hanabi-v0-train")
        num_players = int(task.get("num_players") or 2)
        seed = int(task.get("seed") or 0)

        env = _make_env(env_id=env_id, env_kwargs=RUNTIME_CONFIG.env_kwargs if RUNTIME_CONFIG else {})
        env.reset(num_players=num_players, seed=seed)

        agents: Dict[int, mg.Agent] = {}
        for pid in range(num_players):
            prompt = prompt_template.format(player_id=pid, num_players=num_players)
            agents[pid] = _make_agent(prompt=prompt)

        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = agents[player_id](observation)
            done, _ = env.step(action=action)

        rewards, _ = env.close()
        return _normalize_reward(_score_from_rewards(rewards))

    return hanabi_episode_rollout


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the Hanabi agent prompt with Agent Lightning APO on top of mindgames self-play."
    )
    ap.add_argument("--model", default=os.getenv("MODEL", "/workspace/models/Qwen3-8B"))
    ap.add_argument("--agent-kind", choices=("qwen", "openai"), default=os.getenv("AGENT_KIND", "qwen"))
    ap.add_argument("--env-id", default=os.getenv("ENV_ID", "Hanabi-v0-train"))
    ap.add_argument("--num-players", type=int, default=int(os.getenv("NUM_PLAYERS", "2")))
    ap.add_argument("--train-episodes", type=int, default=int(os.getenv("TRAIN_EPISODES", "128")))
    ap.add_argument("--val-episodes", type=int, default=int(os.getenv("VAL_EPISODES", "32")))
    ap.add_argument("--train-seed", type=int, default=int(os.getenv("TRAIN_SEED", "0")))
    ap.add_argument("--val-seed", type=int, default=int(os.getenv("VAL_SEED", "100000")))
    ap.add_argument("--train-task-file", default=os.getenv("TRAIN_TASK_FILE"))
    ap.add_argument("--val-task-file", default=os.getenv("VAL_TASK_FILE"))
    ap.add_argument("--env-kwargs", default=os.getenv("ENV_KWARGS", "{}"))
    ap.add_argument("--prompt-template-file", default=os.getenv("PROMPT_TEMPLATE_FILE"))
    ap.add_argument("--max-trials", type=int, default=_parse_optional_int(os.getenv("MAX_TRIALS")))
    ap.add_argument("--temperature", default=os.getenv("TEMPERATURE", "0"))
    ap.add_argument("--top-p", default=os.getenv("TOP_P"))
    ap.add_argument("--max-tokens", default=os.getenv("MAX_TOKENS", "256"))
    ap.add_argument("--request-timeout-s", type=float, default=float(os.getenv("REQUEST_TIMEOUT_S", "60")))
    ap.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    ap.add_argument("--retry-delay-s", type=float, default=float(os.getenv("RETRY_DELAY_S", "0.5")))
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--enable-thinking", choices=("true", "false", "auto"), default=os.getenv("ENABLE_THINKING", "false"))
    ap.add_argument("--reward-scale", type=float, default=float(os.getenv("REWARD_SCALE", "25")))
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        import agentlightning as agl
    except ImportError as exc:
        raise SystemExit(
            "agentlightning is not installed. Run `uv sync --extra agent-lightning --extra agents` first."
        ) from exc

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
        model=args.model,
        agent_kind=args.agent_kind,
        api_key=args.api_key,
        base_url=args.base_url,
        request_timeout_s=float(args.request_timeout_s),
        max_retries=int(args.max_retries),
        retry_delay_s=float(args.retry_delay_s),
        temperature=_parse_optional_float(args.temperature),
        top_p=_parse_optional_float(args.top_p),
        max_tokens=_parse_optional_int(args.max_tokens),
        enable_thinking=enable_thinking,
        env_kwargs=env_kwargs,
        reward_scale=float(args.reward_scale),
    )

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
        raise SystemExit("Validation dataset is empty.")

    rollout_fn = _build_rollout_fn(agl)
    trainer = agl.Trainer(
        rollout_fn=rollout_fn,
        adapter=agl.TraceToMessages(),
        algorithm=agl.APO(),
        initial_resources={
            "prompt_template": agl.PromptTemplate(_load_prompt_template(args.prompt_template_file))
        },
    )

    fit_kwargs: Dict[str, Any] = {}
    if args.max_trials is not None:
        fit_kwargs["max_trials"] = int(args.max_trials)

    trainer.fit(train_dataset, val_dataset, **fit_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
