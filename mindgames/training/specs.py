from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from mindgames.training.contracts import GameName


def _unwrap_env(env: object) -> object:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def _identity_reward(value: float) -> float:
    return float(value)


def _normalize_hanabi_reward(value: float) -> float:
    return float(value) / 9.0


def _extract_hanabi_score(game_state: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(game_state, dict):
        return None
    fireworks = game_state.get("fireworks")
    if not isinstance(fireworks, dict):
        return None
    return float(sum(int(v) for v in fireworks.values()))


def _default_terminal_message(game_state: Optional[dict[str, Any]], reward: float) -> str:
    del game_state
    return f"Episode finished.\nNormalized reward: {reward:.4f}"


def _hanabi_terminal_message(game_state: Optional[dict[str, Any]], reward: float) -> str:
    score = _extract_hanabi_score(game_state)
    if score is None:
        return _default_terminal_message(game_state, reward)
    return f"Episode finished.\nFinal score: {int(score)}/9\nNormalized reward: {reward:.4f}"


@dataclass(frozen=True)
class GameSpec:
    name: GameName
    env_id_prefixes: tuple[str, ...]
    default_env_id: str
    default_max_steps: int
    default_reward_player: int
    turn_instruction: str
    snapshot_instruction: str
    reward_normalizer: Callable[[float], float] = _identity_reward
    fallback_terminal_score: Optional[Callable[[Optional[dict[str, Any]]], Optional[float]]] = None
    terminal_message_builder: Callable[[Optional[dict[str, Any]], float], str] = _default_terminal_message

    def extract_terminal_reward(self, env: object, reward_player: int) -> float:
        core_env = _unwrap_env(env)
        state = getattr(core_env, "state", None)
        rewards = getattr(state, "rewards", None)
        game_state = getattr(state, "game_state", None)

        if isinstance(rewards, dict):
            if self.name == "mini_hanabi" or reward_player < 0:
                raw_reward = float(rewards.get(0, 0.0))
            else:
                raw_reward = float(rewards.get(reward_player, 0.0))
            return self.reward_normalizer(raw_reward)

        if self.fallback_terminal_score is not None:
            raw_score = self.fallback_terminal_score(game_state if isinstance(game_state, dict) else None)
            if raw_score is not None:
                return self.reward_normalizer(raw_score)

        raise ValueError(f"{self.name} env did not expose terminal rewards.")

    def build_terminal_message(self, env: object, reward: float) -> str:
        core_env = _unwrap_env(env)
        state = getattr(core_env, "state", None)
        game_state = getattr(state, "game_state", None)
        payload = game_state if isinstance(game_state, dict) else None
        return self.terminal_message_builder(payload, reward)


COMMON_TURN_INSTRUCTION = (
    "At each turn, act for the current player shown in the latest game state. "
    "The current player may change between turns."
)
COMMON_SNAPSHOT_INSTRUCTION = (
    "Each new user message is a self-contained snapshot of the latest state in the same episode. "
    "Use the latest snapshot when choosing the next action."
)


GAME_SPECS: dict[GameName, GameSpec] = {
    "mini_hanabi": GameSpec(
        name="mini_hanabi",
        env_id_prefixes=("MiniHanabi-v0",),
        default_env_id="MiniHanabi-v0-train",
        default_max_steps=12,
        default_reward_player=-1,
        turn_instruction=COMMON_TURN_INSTRUCTION,
        snapshot_instruction=COMMON_SNAPSHOT_INSTRUCTION,
        reward_normalizer=_normalize_hanabi_reward,
        fallback_terminal_score=_extract_hanabi_score,
        terminal_message_builder=_hanabi_terminal_message,
    ),
    "colonel_blotto": GameSpec(
        name="colonel_blotto",
        env_id_prefixes=("ColonelBlotto-v0",),
        default_env_id="ColonelBlotto-v0-train",
        default_max_steps=32,
        default_reward_player=0,
        turn_instruction=COMMON_TURN_INSTRUCTION,
        snapshot_instruction=COMMON_SNAPSHOT_INSTRUCTION,
    ),
    "negotiation": GameSpec(
        name="negotiation",
        env_id_prefixes=("Negotiation-v0",),
        default_env_id="Negotiation-v0-train",
        default_max_steps=20,
        default_reward_player=0,
        turn_instruction=COMMON_TURN_INSTRUCTION,
        snapshot_instruction=COMMON_SNAPSHOT_INSTRUCTION,
    ),
}

DEFAULT_ENV_IDS: dict[GameName, str] = {
    game: spec.default_env_id for game, spec in GAME_SPECS.items()
}
DEFAULT_MAX_STEPS: dict[GameName, int] = {
    game: spec.default_max_steps for game, spec in GAME_SPECS.items()
}
DEFAULT_REWARD_PLAYER: dict[GameName, int] = {
    game: spec.default_reward_player for game, spec in GAME_SPECS.items()
}


def get_game_spec(game: GameName) -> GameSpec:
    return GAME_SPECS[game]


def resolve_env_id(game: GameName, env_id: Optional[str]) -> str:
    return env_id or get_game_spec(game).default_env_id


def default_max_steps(game: GameName) -> int:
    return get_game_spec(game).default_max_steps


def default_reward_player(game: GameName) -> int:
    return get_game_spec(game).default_reward_player


def infer_game_name(env_id: str) -> Optional[GameName]:
    for name, spec in GAME_SPECS.items():
        if any(env_id.startswith(prefix) for prefix in spec.env_id_prefixes):
            return name
    return None
