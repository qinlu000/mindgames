#!/usr/bin/env python3
"""
Convert mindgames rollout JSONL (from tools/rollout/run_rollouts.py) into
a generic chat-style SFT JSONL dataset.

Output schema (per line):
{
  "messages": [
    {"role":"system","content": "..."},
    {"role":"user","content": "..."},
    {"role":"assistant","content": "..."}
  ],
  "meta": {...}
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _pick_reasoning(step_rec: Dict[str, Any], reasoning_field: str) -> str:
    if reasoning_field and reasoning_field != "auto":
        val = step_rec.get(reasoning_field)
        return "" if val is None else str(val).strip()
    for key in ("raw_reasoning", "reasoning"):
        val = step_rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _ensure_think_wrapped(reasoning: str, think_open: str, think_close: str) -> str:
    reasoning = reasoning.strip()
    if not reasoning:
        return ""
    if think_open in reasoning and think_close in reasoning:
        return reasoning
    return f"{think_open}\n{reasoning}\n{think_close}"


def _assistant_content(step_rec: Dict[str, Any], args: argparse.Namespace) -> str:
    action = str(step_rec.get("action", "")).strip()
    if args.assistant_format == "action":
        return action
    reasoning = _pick_reasoning(step_rec, args.reasoning_field)
    if not reasoning:
        return action
    think_block = _ensure_think_wrapped(reasoning, args.think_open, args.think_close)
    if not action:
        return think_block
    return f"{think_block}\n{action}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Rollouts JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="SFT JSONL")
    ap.add_argument("--system", default="You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.")
    ap.add_argument("--env-id", default="", help="Optional filter: only keep this env_id")
    ap.add_argument("--skip-invalid", action="store_true", help="Drop all steps from players flagged invalid_move at episode end")
    ap.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional: only keep episodes whose (cooperative) episode score >= this value (uses the first reward value).",
    )
    ap.add_argument(
        "--assistant-format",
        choices=["action", "think_action"],
        default="action",
        help="Assistant target format: action only, or <think>reasoning</think> + action.",
    )
    ap.add_argument(
        "--reasoning-field",
        default="auto",
        help="Reasoning field to use when --assistant-format=think_action. "
             "Use 'auto' to prefer raw_reasoning then reasoning.",
    )
    ap.add_argument(
        "--think-open",
        default="<think>",
        help="Opening think tag when wrapping reasoning.",
    )
    ap.add_argument(
        "--think-close",
        default="</think>",
        help="Closing think tag when wrapping reasoning.",
    )
    ap.add_argument(
        "--require-action-pattern",
        action="store_true",
        help="Drop samples whose action does not match the standard Hanabi action format.",
    )
    ap.add_argument(
        "--action-pattern",
        default=r"^\[(Play|Discard|Reveal)\]\s+",
        help="Regex used with --require-action-pattern, matched against raw action.",
    )
    args = ap.parse_args()
    action_re = re.compile(args.action_pattern, re.IGNORECASE)

    records = _read_jsonl(Path(args.in_path))
    out_file = Path(args.out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # accumulate one episode at a time
    episode_steps: List[Dict[str, Any]] = []
    episode_env_id: Optional[str] = None
    episode_id: Optional[int] = None

    def flush_episode(end_rec: Dict[str, Any], fp) -> None:
        nonlocal episode_steps, episode_env_id, episode_id
        if episode_env_id is None:
            return

        rewards = end_rec.get("rewards") or {}
        game_info = end_rec.get("game_info") or {}
        # Cooperative envs like Hanabi use identical rewards across players (final score).
        episode_score = None
        if rewards:
            try:
                episode_score = float(next(iter(rewards.values())))
            except Exception:
                episode_score = None
        if args.min_score is not None and episode_score is not None and episode_score < args.min_score:
            episode_steps = []
            episode_env_id = None
            episode_id = None
            return

        for s in episode_steps:
            pid = s.get("player_id")
            pid_key = str(pid)
            pid_info = game_info.get(pid) or game_info.get(pid_key) or {}
            invalid = bool(pid_info.get("invalid_move", False))
            if args.skip_invalid and invalid:
                continue
            action_text = str(s.get("action", "")).strip()
            if args.require_action_pattern and not action_re.search(action_text):
                continue

            sample = {
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": s.get("observation", "")},
                    {"role": "assistant", "content": _assistant_content(s, args)},
                ],
                "meta": {
                    "env_id": episode_env_id,
                    "episode_id": episode_id,
                    "seed": s.get("seed"),
                    "step": s.get("step"),
                    "player_id": pid,
                    "role": s.get("role"),
                    "infer_ms": s.get("infer_ms"),
                    "episode_score": episode_score,
                    "reward": rewards.get(pid, rewards.get(pid_key)),
                    "invalid_move": invalid,
                },
            }
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")

        episode_steps = []
        episode_env_id = None
        episode_id = None

    with out_file.open("w", encoding="utf-8") as fp:
        for rec in records:
            rtype = rec.get("type")
            if rtype == "step":
                if args.env_id and rec.get("env_id") != args.env_id:
                    continue
                if episode_env_id is None:
                    episode_env_id = rec.get("env_id")
                    episode_id = rec.get("episode_id")
                episode_steps.append(rec)
                continue

            if rtype == "episode_end":
                if args.env_id and rec.get("env_id") != args.env_id:
                    continue
                flush_episode(rec, fp)
                continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
