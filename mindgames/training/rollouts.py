from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from mindgames.training.episode import MindGamesEpisode


@dataclass(frozen=True)
class EpisodeTrace:
    step_records: list[dict[str, Any]]
    end_record: dict[str, Any]


def jsonl_write(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def build_step_record(
    *,
    env: Any,
    env_id: str,
    episode_id: int,
    seed: Optional[int],
    step_index: int,
    player_id: int,
    observation: str,
    action: str,
    raw_reasoning: Optional[str],
    normalized_action: str,
    infer_ms: int,
    done: bool,
    step_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "step",
        "env_id": env_id,
        "episode_id": episode_id,
        "seed": seed,
        "step": step_index,
        "player_id": player_id,
        "role": getattr(env.state, "role_mapping", {}).get(player_id, f"Player {player_id}"),
        "observation": observation,
        "action": action,
        "raw_action": action,
        "reasoning": raw_reasoning,
        "raw_reasoning": raw_reasoning,
        "normalized_action": normalized_action,
        "infer_ms": infer_ms,
        "done": done,
        "step_info": step_info,
    }


def build_end_record(
    *,
    env_id: str,
    episode_id: int,
    seed: Optional[int],
    rewards: Any,
    game_info: Any,
) -> dict[str, Any]:
    return {
        "type": "episode_end",
        "env_id": env_id,
        "episode_id": episode_id,
        "seed": seed,
        "rewards": rewards,
        "game_info": game_info,
    }


def write_episode_json(
    *,
    out_dir: Path,
    env_id: str,
    episode_id: int,
    seed: Optional[int],
    step_records: list[dict[str, Any]],
    end_record: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode_{episode_id:06d}.json"
    out_path.write_text(
        json.dumps(
            {
                "env_id": env_id,
                "episode_id": episode_id,
                "seed": seed,
                "steps": step_records,
                "episode_end": end_record,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_mindgames_episode(
    *,
    episode: MindGamesEpisode,
    agents: Mapping[int, Any],
    seed: Optional[int],
    numeric_episode_id: int,
) -> EpisodeTrace:
    step_records: list[dict[str, Any]] = []
    step_index = 0

    while episode.has_active_step():
        step = episode.current_step()
        t0 = time.time()
        action = agents[step.actor_id](step.observation)
        infer_ms = int((time.time() - t0) * 1000)
        _, raw_reasoning = agents[step.actor_id].get_last_content_reasoning()
        transition = episode.step(action)
        step_records.append(
            build_step_record(
                env=episode.env,
                env_id=episode.env_id,
                episode_id=numeric_episode_id,
                seed=seed,
                step_index=step_index,
                player_id=step.actor_id,
                observation=step.observation,
                action=action,
                raw_reasoning=raw_reasoning,
                normalized_action=transition.normalized_action,
                infer_ms=infer_ms,
                done=transition.done,
                step_info=transition.step_info,
            )
        )
        step_index += 1

    rewards, game_info = episode.close()
    end_record = build_end_record(
        env_id=episode.env_id,
        episode_id=numeric_episode_id,
        seed=seed,
        rewards=rewards,
        game_info=game_info,
    )
    return EpisodeTrace(step_records=step_records, end_record=end_record)
