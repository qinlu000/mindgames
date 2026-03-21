from __future__ import annotations

from typing import Any, Optional

try:
    from verl.interactions.base import BaseInteraction
except ModuleNotFoundError:
    class BaseInteraction:  # type: ignore[no-redef]
        def __init__(self, config: dict[str, Any]):
            self.config = config

from mindgames.training.contracts import GameName
from mindgames.training.dataset import (
    build_dataset_row,
    build_initial_prompt_messages,
    build_interaction_kwargs,
)
from mindgames.training.episode import MindGamesEpisode, format_state_message


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
        self._episodes: dict[str, MindGamesEpisode] = {}

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
        episode = MindGamesEpisode.create(
            game=game,
            seed=seed,
            env_id=env_id,
            max_steps=max_steps,
            reward_player=reward_player,
            episode_id=instance_id,
        )
        self._episodes[episode.episode_id] = episode
        return episode.episode_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del kwargs
        episode = self._episodes[instance_id]

        assistant_message = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                assistant_message = str(message.get("content", ""))
                break

        transition = episode.step(assistant_message)
        metrics = {"normalized_action": transition.normalized_action}
        if transition.done:
            return (
                True,
                transition.terminal_message or "Episode finished.",
                float(transition.terminal_reward or 0.0),
                metrics,
            )

        if transition.next_step is None:
            raise RuntimeError("Non-terminal transition must carry the next step.")
        return (
            False,
            format_state_message(transition.next_step.observation),
            0.0,
            metrics,
        )

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        episode = self._episodes.pop(instance_id, None)
        if episode is not None:
            episode.close()


__all__ = [
    "MindGamesInteraction",
    "build_dataset_row",
    "build_initial_prompt_messages",
    "build_interaction_kwargs",
    "compute_score",
]
