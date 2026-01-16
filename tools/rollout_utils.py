#!/usr/bin/env python3
"""
Shared helpers for compact rollout/episode formats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


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
    raw_action = step_rec.get("raw_action", step_rec.get("action"))
    normalized_action = step_rec.get("normalized_action", step_rec.get("action"))

    out: Dict[str, Any] = {
        "step": step_rec.get("step"),
        "player_id": step_rec.get("player_id"),
        "role": step_rec.get("role"),
        "action": step_rec.get("action"),
        "raw_action": raw_action,
        "normalized_action": normalized_action,
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
