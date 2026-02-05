#!/usr/bin/env python3
"""
Pretty-print mindgames JSONL logs for human reading.

Examples:
  python tools/data/view_jsonl.py data/rollouts.jsonl | less -R
  python tools/data/view_jsonl.py data/rollouts.jsonl --episode-id 0
  python tools/data/view_jsonl.py data/rollouts.jsonl --tail 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                continue
            yield obj


def _truncate(s: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def _format_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _episode_filter(episode_ids: Optional[set[int]], obj: Dict[str, Any]) -> bool:
    if not episode_ids:
        return True
    eid = obj.get("episode_id", None)
    return isinstance(eid, int) and eid in episode_ids


def _print_step(obj: Dict[str, Any], *, max_obs_chars: Optional[int], max_action_chars: Optional[int]) -> None:
    eid = obj.get("episode_id", "?")
    step = obj.get("step", "?")
    pid = obj.get("player_id", "?")
    infer_ms = obj.get("infer_ms", None)
    done = obj.get("done", None)
    role = obj.get("role", None)
    action = obj.get("action", "")
    obs = obj.get("observation", "")

    header = f"EP {eid} STEP {step} P{pid}"
    if role:
        header += f" ({role})"
    if infer_ms is not None:
        header += f" infer_ms={infer_ms}"
    if done is not None:
        header += f" done={done}"

    print("=" * 80)
    print(header)
    if isinstance(action, str) and action:
        print("\nAction:")
        print(_truncate(action, max_action_chars))
    if isinstance(obs, str) and obs:
        print("\nObservation:")
        print(_truncate(obs, max_obs_chars))

    step_info = obj.get("step_info", None)
    if step_info:
        print("\nStep info:")
        print(_format_json(step_info))


def _print_episode_end(obj: Dict[str, Any]) -> None:
    eid = obj.get("episode_id", "?")
    rewards = obj.get("rewards", None)
    game_info = obj.get("game_info", None)
    print("=" * 80)
    print(f"EP {eid} END")
    if rewards is not None:
        print("\nRewards:")
        print(_format_json(rewards))
    if game_info is not None:
        print("\nGame info:")
        print(_format_json(game_info))


def _tail(seq: Sequence[Dict[str, Any]], n: Optional[int]) -> Sequence[Dict[str, Any]]:
    if n is None:
        return seq
    if n <= 0:
        return []
    return seq[-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="JSONL path (from run_rollouts/probe scripts)")
    ap.add_argument("--episode-id", type=int, action="append", default=None, help="Filter to episode_id (repeatable)")
    ap.add_argument("--tail", type=int, default=None, help="Show only last N records")
    ap.add_argument("--max-obs-chars", type=int, default=2000, help="Truncate long observations (0 = no truncation)")
    ap.add_argument("--max-action-chars", type=int, default=400, help="Truncate long actions (0 = no truncation)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    episode_ids = set(args.episode_id) if args.episode_id else None
    records = [r for r in _iter_jsonl(path) if _episode_filter(episode_ids, r)]
    records = list(_tail(records, args.tail))

    max_obs_chars = None if args.max_obs_chars == 0 else int(args.max_obs_chars)
    max_action_chars = None if args.max_action_chars == 0 else int(args.max_action_chars)

    for obj in records:
        t = obj.get("type", None)
        if t == "step":
            _print_step(obj, max_obs_chars=max_obs_chars, max_action_chars=max_action_chars)
        elif t == "episode_end":
            _print_episode_end(obj)
        else:
            print("=" * 80)
            print(_format_json(obj))

    if records:
        print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
