#!/usr/bin/env python3
"""
Split a rollouts JSONL file (from tools/run_rollouts.py) into one JSON per episode.

Example:
  cd mindgames
  python tools/split_rollouts_jsonl.py experiments/hanabi_eval/runs/<run_id>/rollouts.jsonl --out-dir experiments/hanabi_eval/runs/<run_id>/episodes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _maybe_parse_hanabi_state(observation: str) -> Optional[Dict[str, Any]]:
    # Best-effort parser for the built-in Hanabi text observation format.
    # For non-Hanabi envs this returns None.
    if "Hanabi" not in observation and "Fireworks" not in observation:
        return None

    def _grab_int(prefix: str) -> Optional[int]:
        needle = prefix + " there are "
        i = observation.find(needle)
        if i < 0:
            return None
        j = i + len(needle)
        k = j
        while k < len(observation) and observation[k].isdigit():
            k += 1
        if k == j:
            return None
        try:
            return int(observation[j:k])
        except Exception:
            return None

    info_tokens = _grab_int("Info tokens:")
    fuse_tokens = _grab_int("Fuse tokens:")

    deck_size: Optional[int] = None
    deck_needle = "Deck size:"
    i = observation.find(deck_needle)
    if i >= 0:
        j = i + len(deck_needle)
        while j < len(observation) and observation[j] == " ":
            j += 1
        k = j
        while k < len(observation) and observation[k].isdigit():
            k += 1
        if k > j:
            try:
                deck_size = int(observation[j:k])
            except Exception:
                deck_size = None

    fireworks: Dict[str, int] = {}
    for color in ("white", "yellow", "green", "blue", "red"):
        needle = f"{color}:"
        idx = observation.find(needle)
        if idx < 0:
            needle = f"{color}     :"
            idx = observation.find(needle)
        if idx < 0:
            continue
        j = idx + len(needle)
        while j < len(observation) and observation[j] == " ":
            j += 1
        if j < len(observation) and observation[j].isdigit():
            fireworks[color] = int(observation[j])

    state: Dict[str, Any] = {"fireworks": fireworks}
    if deck_size is not None:
        state["deck_size"] = deck_size
    if info_tokens is not None:
        state["info_tokens"] = info_tokens
    if fuse_tokens is not None:
        state["fuse_tokens"] = fuse_tokens
    return state


def _compact_step_rec(step_rec: Dict[str, Any], *, max_obs_chars: Optional[int]) -> Dict[str, Any]:
    obs = step_rec.get("observation") or ""

    out: Dict[str, Any] = {
        "step": step_rec.get("step"),
        "player_id": step_rec.get("player_id"),
        "role": step_rec.get("role"),
        "action": step_rec.get("action"),
        "infer_ms": step_rec.get("infer_ms"),
        "done": step_rec.get("done"),
    }

    if isinstance(obs, str) and max_obs_chars is not None and max_obs_chars > 0 and len(obs) > max_obs_chars:
        out["observation"] = obs[: max(0, max_obs_chars - 3)] + "..."
    else:
        out["observation"] = obs

    state = _maybe_parse_hanabi_state(obs) if isinstance(obs, str) else None
    if state:
        out["state"] = state

    step_info = step_rec.get("step_info")
    if step_info:
        out["step_info"] = step_info

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Input rollouts.jsonl")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for per-episode JSON files")
    ap.add_argument("--indent", type=int, default=2, help="Pretty-print indent (0 = minified)")
    ap.add_argument(
        "--max-obs-chars",
        type=int,
        default=0,
        help="Truncate observation to this many chars (0 = no truncation).",
    )
    args = ap.parse_args()

    in_path = Path(args.path)
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else int(args.indent)

    steps_by_ep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    end_by_ep: Dict[int, Dict[str, Any]] = {}
    meta_by_ep: Dict[int, Dict[str, Any]] = {}

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {line_no} in {in_path}: {e}") from e
            if not isinstance(obj, dict):
                continue
            ep = obj.get("episode_id", None)
            if not isinstance(ep, int):
                continue

            t = obj.get("type", None)
            if t == "step":
                steps_by_ep[ep].append(_compact_step_rec(obj, max_obs_chars=(None if int(args.max_obs_chars) == 0 else int(args.max_obs_chars))))
                meta_by_ep.setdefault(
                    ep,
                    {"env_id": obj.get("env_id", None), "seed": obj.get("seed", None)},
                )
            elif t == "episode_end":
                end_by_ep[ep] = obj
                meta_by_ep.setdefault(
                    ep,
                    {"env_id": obj.get("env_id", None), "seed": obj.get("seed", None)},
                )

    for ep, steps in sorted(steps_by_ep.items(), key=lambda kv: kv[0]):
        meta = meta_by_ep.get(ep, {})
        out_path = out_dir / f"episode_{ep:06d}.json"
        out_path.write_text(
            json.dumps(
                {
                    "env_id": meta.get("env_id", None),
                    "episode_id": ep,
                    "seed": meta.get("seed", None),
                    "steps": steps,
                    "episode_end": end_by_ep.get(ep, None),
                },
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
