from __future__ import annotations

from typing import Any, Optional

from mindgames.training.contracts import GameName, InteractionKwargs
from mindgames.training.episode import MindGamesEpisode
from mindgames.training.specs import default_max_steps, default_reward_player, resolve_env_id


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


def build_initial_prompt_messages(
    *,
    game: GameName,
    seed: int,
    env_id: Optional[str] = None,
) -> list[dict[str, str]]:
    episode = MindGamesEpisode.create(
        game=game,
        seed=seed,
        env_id=env_id,
        episode_id=f"{game}:{seed}",
    )
    try:
        return episode.build_initial_prompt_messages()
    finally:
        episode.close()


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
