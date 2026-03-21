from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict


GameName = Literal["mini_hanabi", "colonel_blotto", "negotiation"]


class InteractionKwargs(TypedDict, total=False):
    name: str
    game: GameName
    seed: int
    env_id: str
    max_steps: int
    reward_player: int


@dataclass(frozen=True)
class GameStep:
    game: GameName
    env_id: str
    episode_id: str
    turn_index: int
    actor_id: int
    observation: str
    legal_actions: Optional[list[str]]
    action_mode: str
    obs_mode: str
    reward_mode: str


@dataclass(frozen=True)
class EpisodeStepResult:
    step: GameStep
    raw_action: str
    normalized_action: str
    done: bool
    step_info: dict[str, Any]
    next_step: Optional[GameStep]
    reward_delta: float
    terminal_reward: Optional[float]
    terminal_message: Optional[str]
