#!/usr/bin/env python3
"""
Summarize mindgames rollout JSONL (from tools/run_rollouts.py).

Focuses on practical eval metrics:
- win/loss/draw rates (when rewards are competitive)
- avg reward / score
- invalid_move rate
- avg turn_count

Examples:
  python tools/summarize_textarena_rollouts.py data/tad.rollouts.jsonl
  python tools/summarize_textarena_rollouts.py data/hanabi.rollouts.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get(d: Dict[str, Any], key: int) -> Any:
    return d.get(key, d.get(str(key)))


def _is_competitive_rewards(rewards: Dict[str, Any]) -> bool:
    # Heuristic: if not all rewards equal, treat as competitive.
    vals = list(rewards.values())
    if not vals:
        return False
    return any(v != vals[0] for v in vals[1:])


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def summarize(paths: List[Path]) -> Dict[str, Any]:
    per_env: Dict[str, Any] = {}

    for path in paths:
        for rec in _iter_jsonl(path):
            if rec.get("type") != "episode_end":
                continue

            env_id = rec.get("env_id", "UNKNOWN")
            rewards = rec.get("rewards") or {}
            game_info = rec.get("game_info") or {}

            env = per_env.setdefault(
                env_id,
                {
                    "episodes": 0,
                    "by_player": defaultdict(lambda: {"reward": [], "turn_count": [], "invalid_move": 0, "wins": 0, "losses": 0, "draws": 0}),
                    "score": [],
                    "reasons": defaultdict(int),
                },
            )

            env["episodes"] += 1
            env["reasons"][(str((_get(game_info, 0) or {}).get("reason", "")) or "").strip()] += 1

            # player-level aggregates
            # Note: JSON keys may be strings.
            player_ids: List[int] = []
            for k in rewards.keys():
                try:
                    player_ids.append(int(k))
                except Exception:
                    pass
            if not player_ids:
                for k in game_info.keys():
                    try:
                        player_ids.append(int(k))
                    except Exception:
                        pass
            player_ids = sorted(set(player_ids))

            # cooperative score (e.g., Hanabi): rewards are identical across players
            if rewards and not _is_competitive_rewards(rewards):
                try:
                    env["score"].append(float(next(iter(rewards.values()))))
                except Exception:
                    pass

            for pid in player_ids:
                r = _get(rewards, pid)
                info = _get(game_info, pid) or {}
                invalid = bool(info.get("invalid_move", False))
                turns = info.get("turn_count")

                bp = env["by_player"][pid]
                if r is not None:
                    try:
                        bp["reward"].append(float(r))
                    except Exception:
                        pass
                if turns is not None:
                    try:
                        bp["turn_count"].append(float(turns))
                    except Exception:
                        pass
                if invalid:
                    bp["invalid_move"] += 1

            # competitive W/L/D for 2-player (general)
            if len(player_ids) == 2 and rewards:
                r0 = _get(rewards, player_ids[0])
                r1 = _get(rewards, player_ids[1])
                try:
                    r0f = float(r0)
                    r1f = float(r1)
                except Exception:
                    r0f = r1f = 0.0

                if r0f > r1f:
                    env["by_player"][player_ids[0]]["wins"] += 1
                    env["by_player"][player_ids[1]]["losses"] += 1
                elif r0f < r1f:
                    env["by_player"][player_ids[1]]["wins"] += 1
                    env["by_player"][player_ids[0]]["losses"] += 1
                else:
                    env["by_player"][player_ids[0]]["draws"] += 1
                    env["by_player"][player_ids[1]]["draws"] += 1

    # finalize
    out: Dict[str, Any] = {"envs": {}}
    for env_id, env in per_env.items():
        episodes = env["episodes"]
        by_player_out: Dict[str, Any] = {}
        for pid, bp in sorted(env["by_player"].items(), key=lambda kv: kv[0]):
            by_player_out[str(pid)] = {
                "avg_reward": _mean(bp["reward"]),
                "avg_turn_count": _mean(bp["turn_count"]),
                "invalid_rate": (bp["invalid_move"] / episodes) if episodes else None,
                "win_rate": (bp["wins"] / episodes) if episodes else None,
                "loss_rate": (bp["losses"] / episodes) if episodes else None,
                "draw_rate": (bp["draws"] / episodes) if episodes else None,
            }

        score = env["score"]
        reasons = dict(sorted(env["reasons"].items(), key=lambda kv: kv[1], reverse=True))

        out["envs"][env_id] = {
            "episodes": episodes,
            "avg_score_if_coop": _mean(score),
            "perfect_score_rate_if_coop": (sum(1 for s in score if s == 25.0) / len(score)) if score else None,
            "by_player": by_player_out,
            "end_reasons": reasons,
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="One or more rollout JSONL files")
    ap.add_argument("--json", action="store_true", help="Output JSON (default).")
    args = ap.parse_args()

    result = summarize([Path(p) for p in args.paths])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
