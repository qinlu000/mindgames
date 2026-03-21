from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict, cast
from uuid import uuid4

import mindgames as mg
from mindgames.envs.registration import get_prompt_profile
from mindgames.prompting import normalize_action_for_env

try:
    from verl.interactions.base import BaseInteraction
except ModuleNotFoundError:
    class BaseInteraction:  # type: ignore[no-redef]
        def __init__(self, config: dict[str, Any]):
            self.config = config

GameName = Literal["mini_hanabi", "colonel_blotto", "negotiation"]

DEFAULT_ENV_IDS: dict[GameName, str] = {
    "mini_hanabi": "MiniHanabi-v0-train",
    "colonel_blotto": "ColonelBlotto-v0-train",
    "negotiation": "Negotiation-v0-train",
}
DEFAULT_MAX_STEPS: dict[GameName, int] = {
    "mini_hanabi": 12,
    "colonel_blotto": 32,
    "negotiation": 20,
}
DEFAULT_REWARD_PLAYER: dict[GameName, int] = {
    "mini_hanabi": -1,
    "colonel_blotto": 0,
    "negotiation": 0,
}


class InteractionKwargs(TypedDict, total=False):
    name: str
    game: GameName
    seed: int
    env_id: str
    max_steps: int
    reward_player: int


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


def _split_observation(observation: str) -> tuple[Optional[str], str]:
    if "\n\n" not in observation:
        return None, observation.strip()
    prompt_block, remainder = observation.split("\n\n", 1)
    return prompt_block.strip(), remainder.strip()


def _strip_dynamic_role_line(prompt_block: Optional[str]) -> str:
    if not prompt_block:
        return ""
    lines = prompt_block.splitlines()
    if lines and (lines[0].startswith("You are Player ") or lines[0].startswith("You are player ")):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _build_system_message(env_id: str, prompt_block: Optional[str]) -> Optional[str]:
    prompt_profile = get_prompt_profile(env_id)
    parts: list[str] = []

    if prompt_profile is not None and prompt_profile.system_prompt:
        parts.append(prompt_profile.system_prompt.strip())

    parts.append(
        "At each turn, act for the current player shown in the latest game state. "
        "The current player may change between turns."
    )
    parts.append(
        "Each new user message is a self-contained snapshot of the latest state in the same episode. "
        "Use the latest snapshot when choosing the next action."
    )

    static_rules = _strip_dynamic_role_line(prompt_block)
    if static_rules:
        parts.append(static_rules)

    merged = "\n\n".join(part for part in parts if part)
    return merged or None


def _format_state_message(observation: str) -> str:
    _prompt_block, state_text = _split_observation(observation)
    if not state_text:
        return "Current game state:"
    return f"Current game state:\n{state_text}"


def _normalize_score(game: GameName, raw_reward: float) -> float:
    if game == "mini_hanabi":
        return float(raw_reward) / 9.0
    return float(raw_reward)


def _extract_reward(core_env: object, *, game: GameName, reward_player: int) -> float:
    rewards = getattr(getattr(core_env, "state"), "rewards", None)
    if not isinstance(rewards, dict):
        if game == "mini_hanabi":
            game_state = getattr(getattr(core_env, "state"), "game_state", None)
            fireworks = game_state.get("fireworks") if isinstance(game_state, dict) else None
            if isinstance(fireworks, dict):
                score = sum(int(value) for value in fireworks.values())
                return _normalize_score(game, float(score))
        raise ValueError(f"{game} env did not expose terminal rewards.")

    if game == "mini_hanabi" or reward_player < 0:
        reward = float(rewards.get(0, 0.0))
    else:
        reward = float(rewards.get(reward_player, 0.0))
    return _normalize_score(game, reward)


def _build_terminal_message(env: Any, *, game: GameName, reward: float) -> str:
    core_env = _unwrap_env(env)
    game_state = getattr(getattr(core_env, "state"), "game_state", None)
    if game == "mini_hanabi" and isinstance(game_state, dict):
        fireworks = game_state.get("fireworks")
        if isinstance(fireworks, dict):
            score = sum(int(value) for value in fireworks.values())
            return f"Episode finished.\nFinal score: {score}/9\nNormalized reward: {reward:.4f}"
    return f"Episode finished.\nNormalized reward: {reward:.4f}"


def build_initial_prompt_messages(
    *,
    game: GameName,
    seed: int,
    env_id: Optional[str] = None,
) -> list[dict[str, str]]:
    resolved_env_id = resolve_env_id(game, env_id)
    env = mg.make(resolved_env_id)
    env.reset(num_players=2, seed=seed)
    _player_id, observation = env.get_observation()
    prompt_block, _state_text = _split_observation(observation)
    system_message = _build_system_message(resolved_env_id, prompt_block)

    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": _format_state_message(observation)})
    return messages


def build_interaction_kwargs(
    *,
    game: GameName,
    seed: int,
    env_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    reward_player: Optional[int] = None,
) -> InteractionKwargs:
    return {
        "name": "mindgames",
        "game": game,
        "seed": int(seed),
        "env_id": resolve_env_id(game, env_id),
        "max_steps": default_max_steps(game) if max_steps is None else int(max_steps),
        "reward_player": default_reward_player(game) if reward_player is None else int(reward_player),
    }


def build_dataset_row(
    *,
    game: GameName,
    seed: int,
    index: int,
    env_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    reward_player: Optional[int] = None,
) -> dict[str, Any]:
    interaction_kwargs = build_interaction_kwargs(
        game=game,
        seed=seed,
        env_id=env_id,
        max_steps=max_steps,
        reward_player=reward_player,
    )
    return {
        "prompt": build_initial_prompt_messages(game=game, seed=seed, env_id=env_id),
        "data_source": f"mindgames/{game}",
        "reward_model": {"ground_truth": ""},
        "extra_info": {
            "index": int(index),
            "interaction_kwargs": interaction_kwargs,
        },
    }


def _coerce_score_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    del data_source, solution_str, ground_truth
    payload = extra_info or {}
    turn_scores = _coerce_score_list(payload.get("turn_scores"))
    tool_rewards = _coerce_score_list(payload.get("tool_rewards"))
    score = (turn_scores[-1] if turn_scores else 0.0) + sum(tool_rewards)
    return {
        "score": float(score),
        "terminal_reward": float(turn_scores[-1] if turn_scores else 0.0),
        "tool_reward": float(sum(tool_rewards)),
    }


class MindGamesInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instances: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        game: GameName = "mini_hanabi",
        seed: int = 0,
        env_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        reward_player: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if instance_id is None:
            instance_id = str(uuid4())

        resolved_env_id = resolve_env_id(game, env_id)
        env = mg.make(resolved_env_id)
        env.reset(num_players=2, seed=seed)
        _player_id, observation = env.get_observation()

        self._instances[instance_id] = {
            "env": env,
            "game": game,
            "reward_player": default_reward_player(game) if reward_player is None else int(reward_player),
            "current_observation": observation,
            "max_steps": default_max_steps(game) if max_steps is None else int(max_steps),
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del kwargs
        state = self._instances[instance_id]
        env = state["env"]
        game = cast(GameName, state["game"])
        reward_player = int(state["reward_player"])
        current_observation = str(state["current_observation"])

        assistant_message = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                assistant_message = str(message.get("content", ""))
                break

        normalized_action = normalize_action_for_env(env, current_observation, assistant_message)
        done, info = env.step(normalized_action)
        del info

        if done:
            reward = _extract_reward(_unwrap_env(env), game=game, reward_player=reward_player)
            response = _build_terminal_message(env, game=game, reward=reward)
            return True, response, reward, {"normalized_action": normalized_action}

        _player_id, next_observation = env.get_observation()
        state["current_observation"] = next_observation
        compact_observation = _format_state_message(next_observation)
        return False, compact_observation, 0.0, {"normalized_action": normalized_action}

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._instances.pop(instance_id, None)


__all__ = [
    "DEFAULT_ENV_IDS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_REWARD_PLAYER",
    "MindGamesInteraction",
    "build_dataset_row",
    "build_initial_prompt_messages",
    "build_interaction_kwargs",
    "compute_score",
    "default_max_steps",
    "default_reward_player",
    "resolve_env_id",
]
