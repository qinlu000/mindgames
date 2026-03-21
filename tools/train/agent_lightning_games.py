#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, cast


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name in {"mindgames-agent-lightning-games", "mindgames"}:
            return parent
    raise RuntimeError("Could not locate mindgames project root.")


def _ensure_pkg_importable() -> None:
    project_root = _find_project_root()
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()

import mindgames as mg  # noqa: E402
from mindgames.prompting import normalize_action_for_env  # noqa: E402

GameName = Literal["mini_hanabi", "colonel_blotto", "negotiation"]

DEFAULT_ENV_IDS: dict[GameName, str] = {
    "mini_hanabi": "MiniHanabi-v0-train",
    "colonel_blotto": "ColonelBlotto-v0-train",
    "negotiation": "Negotiation-v0-train",
}
DEFAULT_MAX_STEPS: dict[GameName, int] = {
    "mini_hanabi": 64,
    "colonel_blotto": 32,
    "negotiation": 20,
}
DEFAULT_REWARD_PLAYER: dict[GameName, int] = {
    "mini_hanabi": -1,
    "colonel_blotto": 0,
    "negotiation": 0,
}


class GameTask(TypedDict, total=False):
    game: GameName
    seed: int
    env_id: str
    max_steps: int
    enable_thinking: bool
    reward_player: int


def resolve_default_qwen3_8b_model() -> str:
    local_model = Path("/workspace/models/Qwen3-8B")
    if local_model.is_dir():
        return str(local_model)
    return "Qwen/Qwen3-8B"


def resolve_env_id(game: GameName, env_id: Optional[str]) -> str:
    return env_id or DEFAULT_ENV_IDS[game]


def default_max_steps(game: GameName) -> int:
    return DEFAULT_MAX_STEPS[game]


def default_reward_player(game: GameName) -> int:
    return DEFAULT_REWARD_PLAYER[game]


def _unwrap_env(env: object) -> object:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def _base_env_id(env_id: str) -> str:
    return env_id[:-6] if env_id.endswith("-train") else env_id


def _build_trainable_agent(llm: Any, *, system_prompt: Optional[str], enable_thinking: bool) -> mg.Agent:
    sampling_parameters = dict(getattr(llm, "sampling_parameters", {}) or {})
    return mg.agents.QwenAgent(
        model_name=llm.model,
        system_prompt=system_prompt,
        base_url=llm.endpoint,
        api_key=llm.api_key or "dummy",
        max_retries=2,
        timeout=120,
        max_tokens=int(sampling_parameters.get("max_tokens", 128)),
        temperature=float(sampling_parameters.get("temperature", 0.0)),
        top_p=float(sampling_parameters.get("top_p", 1.0)),
        enable_thinking=enable_thinking,
    )


def _extract_reward(core_env: object, *, game: GameName, reward_player: int) -> float:
    rewards = getattr(getattr(core_env, "state"), "rewards", None)
    if not isinstance(rewards, dict):
        raise ValueError(f"{game} env did not expose terminal rewards.")
    if game == "mini_hanabi" or reward_player < 0:
        return float(rewards.get(0, 0.0)) / 9.0
    return float(rewards.get(reward_player, 0.0))


def rollout_single_episode(task: GameTask, llm: Any) -> float:
    game = cast(GameName, task.get("game", "mini_hanabi"))
    env_id = str(task.get("env_id", DEFAULT_ENV_IDS[game]))
    seed = int(task["seed"])
    max_steps = int(task.get("max_steps", DEFAULT_MAX_STEPS[game]))
    enable_thinking = bool(task.get("enable_thinking", False))
    reward_player = int(task.get("reward_player", DEFAULT_REWARD_PLAYER[game]))

    env = mg.make(env_id)
    env.reset(num_players=2, seed=seed)

    env_spec = mg.get_env_spec(_base_env_id(env_id))
    prompt_profile = env_spec.prompt_profile
    agent = _build_trainable_agent(
        llm,
        system_prompt=(prompt_profile.system_prompt if prompt_profile is not None else None),
        enable_thinking=enable_thinking,
    )

    for _ in range(max_steps):
        _player_id, observation = env.get_observation()
        raw_action = agent(observation)
        action = normalize_action_for_env(env, observation, raw_action)
        done, _info = env.step(action)
        if done:
            break

    core_env = _unwrap_env(env)
    return _extract_reward(core_env, game=game, reward_player=reward_player)


def make_rollout():
    import agentlightning as agl

    @agl.rollout
    def games_rollout(task: GameTask, llm) -> float:
        return rollout_single_episode(task, llm)

    return games_rollout


__all__ = [
    "DEFAULT_ENV_IDS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_REWARD_PLAYER",
    "GameTask",
    "default_max_steps",
    "default_reward_player",
    "make_rollout",
    "resolve_default_qwen3_8b_model",
    "resolve_env_id",
    "rollout_single_episode",
]
