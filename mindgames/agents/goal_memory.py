from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mindgames.core import Agent, AgentWrapper

__all__ = ["GoalMemoryConfig", "GoalMemoryAgentWrapper"]

_ALLOWED_GOAL_STATUSES = {"active", "completed", "invalidated", "expired", "removed"}
_ALLOWED_OPS = {"set", "remove"}
_PRIORITY_ORDER = {"high": 2, "medium": 1, "low": 0}

_PLAY_RE = re.compile(r"\[\s*play\s*\]\s*(\d+)\b", re.IGNORECASE)
_DISCARD_RE = re.compile(r"\[\s*discard\s*\]\s*(\d+)\b", re.IGNORECASE)
_REVEAL_RE = re.compile(
    r"\[\s*reveal\s*\]\s*player\s+(\d+)\s+card\s+(\d+)\s+(?:color\s+([a-z]+)|rank\s+(\d+))",
    re.IGNORECASE,
)
_ACTION_LINE_RE = re.compile(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_PLAYER_ID_RE = re.compile(r"You are player\s+(\d+)\b|You are Player\s+(\d+)\b")
_SLOT_TARGET_RE = re.compile(r"^(?:player\s*)?(\d+)[_\s]*slot\s*(\d+)$", re.IGNORECASE)
_SELF_SLOT_TARGET_RE = re.compile(r"^self[_\s]*slot\s*(\d+)$", re.IGNORECASE)
_TOKEN_TARGET_RE = re.compile(r"^(info|fuse)[_\s]*token$", re.IGNORECASE)
_FIREWORK_TARGET_RE = re.compile(r"^firework[_\s]*([a-z]+)$", re.IGNORECASE)


@dataclass
class GoalMemoryConfig:
    max_active_goals: int = 3
    render_topk: int = 2
    default_ttl: int = 2
    max_ops_per_turn: int = 4


@dataclass
class GoalRecord:
    goal_id: str
    goal: str
    target: str = "none"
    priority: str = "medium"
    ttl: int = 2
    status: str = "active"
    created_turn: int = 0
    last_updated_turn: int = 0

    def ttl_left(self, current_turn: int) -> int:
        return int(self.ttl) - int(current_turn - self.last_updated_turn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "target": self.target,
            "priority": self.priority,
            "ttl": self.ttl,
            "status": self.status,
            "created_turn": self.created_turn,
            "last_updated_turn": self.last_updated_turn,
        }


class GoalMemoryState:
    def __init__(self, config: GoalMemoryConfig):
        self.config = config
        self.reset()

    def reset(self, *, episode_id: Optional[int] = None, agent_player_id: Optional[int] = None) -> None:
        self.episode_id = episode_id
        self.agent_player_id = agent_player_id
        self.current_turn = 0
        self.goals: Dict[str, GoalRecord] = {}
        self.events: List[Dict[str, Any]] = []

    def set_turn_context(self, *, turn_id: int, player_id: Optional[int] = None) -> None:
        self.current_turn = int(turn_id)
        if player_id is not None:
            self.agent_player_id = int(player_id)
        self._expire_stale_goals()

    def active_goals(self) -> List[GoalRecord]:
        goals = [goal for goal in self.goals.values() if goal.status == "active"]
        goals.sort(key=lambda g: (-_priority_rank(g.priority), -g.last_updated_turn, g.goal_id))
        return goals

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": "goal_memory.v2_minimal",
            "episode_id": self.episode_id,
            "agent_player_id": self.agent_player_id,
            "current_turn": self.current_turn,
            "max_active_goals": self.config.max_active_goals,
            "default_ttl": self.config.default_ttl,
            "goals": [goal.to_dict() for goal in sorted(self.goals.values(), key=lambda g: (g.created_turn, g.goal_id))],
        }

    def render(self) -> str:
        active = self.active_goals()
        if not active:
            return (
                "Your active goals from previous turns (after system validation and state updates):\n"
                "- (none)"
            )

        labels = ["PRIMARY", "SUPPORT", "FALLBACK"]
        lines = [
            "Your active goals from previous turns (after system validation and state updates):",
            "Treat them as your current working commitments.",
            "If you want to change or remove them, do it explicitly through goal_ops.",
        ]
        for idx, goal in enumerate(active[: self.config.render_topk]):
            label = labels[idx] if idx < len(labels) else f"GOAL-{idx + 1}"
            age = max(0, self.current_turn - goal.created_turn)
            ttl_left = goal.ttl_left(self.current_turn)
            timing = "new" if age == 0 else "carry-over"
            if ttl_left <= 0:
                timing = f"{timing}, expiring"
            lines.append(
                f"- {label} [{timing}, ttl_left={ttl_left}]: "
                f"{goal.goal_id} | {goal.goal} | target={goal.target} | priority={goal.priority}"
            )
        return "\n".join(lines)

    def apply_goal_ops(self, goal_ops: List[Dict[str, Any]], *, source: str = "llm") -> List[Dict[str, Any]]:
        applied: List[Dict[str, Any]] = []
        for raw_op in list(goal_ops or [])[: self.config.max_ops_per_turn]:
            op_name = str(raw_op.get("op") or "").strip().lower()
            if op_name not in _ALLOWED_OPS:
                continue
            goal_id = str(raw_op.get("goal_id") or "").strip()
            if not goal_id:
                continue
            if op_name == "set":
                event = self._set_goal(goal_id, raw_op, source=source)
            else:
                event = self._remove_goal(goal_id, source=source)
            if event is not None:
                applied.append(event)
        self._trim_active_goals()
        return applied

    def record_action_result(self, action: str, *, selected_goal_id: Optional[str]) -> List[Dict[str, Any]]:
        parsed = _parse_hanabi_action(action)
        if not parsed:
            return []

        events: List[Dict[str, Any]] = []
        matched_goal_id: Optional[str] = None
        selected = self.goals.get(selected_goal_id) if selected_goal_id else None
        if selected and selected.status == "active" and _selected_goal_matches_action(selected, parsed, self.agent_player_id):
            matched_goal_id = selected.goal_id
            event = self._set_goal_status(selected, "completed", reason=f"selected goal matched action via {parsed['kind']}", source="system")
            if event is not None:
                events.append(event)

        if parsed["kind"] in {"play", "discard"} and self.agent_player_id is not None:
            affected_slot = int(parsed["slot"])
            for goal in self.active_goals():
                if goal.goal_id == matched_goal_id:
                    continue
                target = _parse_target_ref(goal.target, self.agent_player_id)
                if not target or target["kind"] != "slot":
                    continue
                if target["player"] != self.agent_player_id or target["slot"] is None:
                    continue
                if target["slot"] == affected_slot:
                    event = self._set_goal_status(
                        goal,
                        "invalidated",
                        reason=f"target {goal.target} was consumed by {parsed['kind']}",
                        source="system",
                    )
                    if event is not None:
                        events.append(event)
                elif target["slot"] > affected_slot:
                    old_target = goal.target
                    goal.target = _slot_target_ref(target["player"], target["slot"] - 1, self.agent_player_id)
                    event = self._record_event(
                        goal.goal_id,
                        "rebase_goal",
                        "system",
                        before_status=goal.status,
                        after_status=goal.status,
                        reason=f"shifted target from {old_target} to {goal.target} after {parsed['kind']} {affected_slot}",
                    )
                    events.append(event)

        return events

    def _expire_stale_goals(self) -> None:
        for goal in list(self.goals.values()):
            if goal.status != "active":
                continue
            if goal.ttl_left(self.current_turn) < 0:
                self._set_goal_status(goal, "expired", reason="ttl expired", source="system")

    def _set_goal(self, goal_id: str, raw_op: Dict[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
        existing = self.goals.get(goal_id)
        goal_text = _coerce_optional_str(raw_op.get("goal"))
        if existing is None:
            if not goal_text:
                return None
            goal = GoalRecord(
                goal_id=goal_id,
                goal=goal_text,
                target=_coerce_target_ref(raw_op.get("target")),
                priority=_normalize_priority(raw_op.get("priority"), default="medium"),
                ttl=_coerce_positive_int(raw_op.get("ttl"), default=self.config.default_ttl),
                status="active",
                created_turn=self.current_turn,
                last_updated_turn=self.current_turn,
            )
            self.goals[goal_id] = goal
            return self._record_event(goal_id, "set", source, before_status=None, after_status="active", reason=goal.goal)

        before_status = existing.status
        existing.goal = goal_text or existing.goal
        if raw_op.get("target") is not None:
            existing.target = _coerce_target_ref(raw_op.get("target"))
        existing.priority = _normalize_priority(raw_op.get("priority"), default=existing.priority)
        existing.ttl = _coerce_positive_int(raw_op.get("ttl"), default=existing.ttl)
        existing.status = "active"
        existing.last_updated_turn = self.current_turn
        return self._record_event(goal_id, "set", source, before_status=before_status, after_status=existing.status, reason=existing.goal)

    def _remove_goal(self, goal_id: str, *, source: str) -> Optional[Dict[str, Any]]:
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        return self._set_goal_status(goal, "removed", reason="removed", source=source)

    def _set_goal_status(self, goal: GoalRecord, status: str, *, reason: str, source: str) -> Optional[Dict[str, Any]]:
        if status not in _ALLOWED_GOAL_STATUSES:
            return None
        before_status = goal.status
        goal.status = status
        goal.last_updated_turn = self.current_turn
        return self._record_event(goal.goal_id, status, source, before_status=before_status, after_status=status, reason=reason)

    def _trim_active_goals(self) -> None:
        active = self.active_goals()
        keep = set(goal.goal_id for goal in active[: self.config.max_active_goals])
        for goal in active[self.config.max_active_goals :]:
            if goal.goal_id in keep:
                continue
            self._set_goal_status(goal, "removed", reason="trimmed to active-goal budget", source="system")

    def _record_event(
        self,
        goal_id: str,
        op: str,
        actor: str,
        *,
        before_status: Optional[str],
        after_status: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        event = {
            "event_id": f"goal_evt_{len(self.events):05d}",
            "episode_id": self.episode_id,
            "turn_id": self.current_turn,
            "goal_id": goal_id,
            "op": op,
            "actor": actor,
            "before_status": before_status,
            "after_status": after_status,
            "reason": reason,
        }
        self.events.append(event)
        return event


class GoalMemoryAgentWrapper(AgentWrapper):
    def __init__(self, agent: Agent, config: Optional[GoalMemoryConfig] = None):
        super().__init__(agent)
        self.config = config or GoalMemoryConfig()
        self.state = GoalMemoryState(self.config)
        self.last_goal_turn_output: Optional[Dict[str, Any]] = None
        self.last_goal_events: List[Dict[str, Any]] = []
        self.last_goal_prompt: Optional[str] = None
        self._last_pre_turn_active_ids: List[str] = []

    def reset_episode(self, *, episode_id: Optional[int] = None, player_id: Optional[int] = None) -> None:
        self.state.reset(episode_id=episode_id, agent_player_id=player_id)
        self.last_goal_turn_output = None
        self.last_goal_events = []
        self.last_goal_prompt = None
        self._last_pre_turn_active_ids = []

    def set_turn_context(self, *, episode_id: Optional[int] = None, turn_id: int, player_id: Optional[int] = None) -> None:
        if episode_id is not None:
            self.state.episode_id = episode_id
        self.state.set_turn_context(turn_id=turn_id, player_id=player_id)

    def get_goal_memory_snapshot(self) -> Dict[str, Any]:
        return self.state.snapshot()

    def get_last_goal_turn_output(self) -> Optional[Dict[str, Any]]:
        return dict(self.last_goal_turn_output) if isinstance(self.last_goal_turn_output, dict) else None

    def get_last_goal_events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self.last_goal_events]

    def get_last_goal_prompt(self) -> Optional[str]:
        return self.last_goal_prompt

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")

        if self.state.agent_player_id is None:
            player_id = _infer_player_id(observation)
            if player_id is not None:
                self.state.agent_player_id = player_id

        self._last_pre_turn_active_ids = [goal.goal_id for goal in self.state.active_goals()]
        prompt = self._build_prompt(observation)
        self.last_goal_prompt = prompt

        raw_output = self._call_wrapped_agent(prompt)
        turn_output = _parse_turn_output(raw_output)
        if not turn_output.get("action"):
            fallback_action = _extract_action_from_text(raw_output)
            if fallback_action:
                turn_output["action"] = fallback_action
        if not turn_output.get("action"):
            raise ValueError(f"GoalMemoryAgentWrapper could not extract an action from: {raw_output!r}")

        events = self.state.apply_goal_ops(turn_output.get("goal_ops") or [], source="llm")
        self.last_goal_events = list(events)
        turn_output["active_goal_ids_before_turn"] = list(self._last_pre_turn_active_ids)
        turn_output["active_goal_ids_after_ops"] = [goal.goal_id for goal in self.state.active_goals()]
        turn_output["goal_memory"] = self.state.snapshot()
        self.last_goal_turn_output = turn_output
        return str(turn_output["action"])

    def record_step_result(
        self,
        *,
        action: str,
        normalized_action: Optional[str] = None,
        step_info: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> None:
        del step_info
        applied = self.state.record_action_result(normalized_action or action, selected_goal_id=(self.last_goal_turn_output or {}).get("selected_goal_id"))
        if applied:
            self.last_goal_events.extend(applied)
        if self.last_goal_turn_output is not None:
            self.last_goal_turn_output["goal_memory"] = self.state.snapshot()
            self.last_goal_turn_output["goal_events"] = self.get_last_goal_events()
            self.last_goal_turn_output["done"] = bool(done)

    def _build_prompt(self, observation: str) -> str:
        observation = _adapt_text_to_goal_memory_mode(observation, kind="observation")
        return (
            "Goal memory is enabled. Keep only a tiny cross-turn goal list.\n"
            "The rendered goals below are goals you explicitly set in previous turns and that the system carried into this turn after validation and state updates.\n"
            "Treat them as your current working commitments unless you explicitly change or remove them through goal_ops.\n"
            f"Keep at most {self.config.max_active_goals} active goals and at most {self.config.max_ops_per_turn} goal_ops this turn.\n"
            "Return EXACTLY ONE JSON object with this schema:\n"
            "{\n"
            '  "selected_goal_id": "string or null",\n'
            '  "goal_ops": [\n'
            "    {\n"
            '      "op": "set|remove",\n'
            '      "goal_id": "string",\n'
            '      "goal": "short goal text",\n'
            '      "target": "short target ref like self_slot2 or player1_slot4",\n'
            '      "priority": "high|medium|low",\n'
            '      "ttl": 2\n'
            "    }\n"
            "  ],\n"
            '  "action": "[Play] X | [Discard] X | [Reveal] player N card X color C | [Reveal] player N card X rank R"\n'
            "}\n"
            "For `remove`, provide only `op` and `goal_id`.\n"
            "Return JSON only; do not add any prose.\n\n"
            f"{self.state.render()}\n\n"
            "Game observation:\n"
            f"{observation}"
        )

    def _call_wrapped_agent(self, prompt: str) -> str:
        original_system_prompt = getattr(self.agent, "system_prompt", None)
        if original_system_prompt is not None:
            self.agent.system_prompt = _adapt_text_to_goal_memory_mode(original_system_prompt, kind="system")
        try:
            return self.agent(prompt)
        finally:
            if original_system_prompt is not None:
                self.agent.system_prompt = original_system_prompt


def _parse_turn_output(text: str) -> Dict[str, Any]:
    payload = _extract_json_object(text)
    if payload is None:
        return {"selected_goal_id": None, "goal_ops": [], "action": None, "raw_response": text, "parse_error": "json_not_found"}

    goal_ops = payload.get("goal_ops")
    if not isinstance(goal_ops, list):
        goal_ops = []
    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        action = None
    return {
        "selected_goal_id": _coerce_optional_str(payload.get("selected_goal_id")),
        "goal_ops": [dict(item) for item in goal_ops if isinstance(item, dict)],
        "action": action.strip() if isinstance(action, str) else None,
        "raw_response": text,
        "parse_error": None,
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    for candidate in (text, _extract_fenced_json(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


def _extract_fenced_json(text: str) -> Optional[str]:
    match = _FENCED_JSON_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _adapt_text_to_goal_memory_mode(text: Optional[str], *, kind: str) -> str:
    content = (text or "").strip()
    if not content:
        return _goal_memory_contract_prefix(kind)

    replacements = [
        (
            "Output EXACTLY ONE valid action and nothing else (no reasoning).",
            "Return EXACTLY ONE JSON object and nothing else. The `action` field must contain exactly one valid Hanabi action. Do not include free-form reasoning outside the JSON object.",
        ),
        (
            "Output EXACTLY ONE action, nothing else.",
            "Return EXACTLY ONE JSON object, nothing else. The `action` field must contain exactly one legal Hanabi action.",
        ),
    ]
    updated = content
    replaced = False
    for before, after in replacements:
        if before in updated:
            updated = updated.replace(before, after)
            replaced = True
    if replaced:
        return updated

    return _goal_memory_contract_prefix(kind) + "\n\n" + content


def _goal_memory_contract_prefix(kind: str) -> str:
    if kind == "system":
        return (
            "You are an expert Hanabi teammate with goal memory. "
            "Return EXACTLY ONE JSON object and nothing else. "
            "The JSON must contain selected_goal_id, goal_ops, and action."
        )
    return (
        "Goal-memory response mode: return EXACTLY ONE JSON object and nothing else. "
        "The `action` field must contain exactly one legal Hanabi action."
    )


def _extract_action_from_text(text: str) -> Optional[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    bracket_lines = [line for line in lines if "[" in line and "]" in line]
    if bracket_lines:
        line = bracket_lines[-1]
        return line[line.index("[") :].strip()
    for line in reversed(lines):
        if _ACTION_LINE_RE.match(line):
            return line.strip()
    return None


def _parse_hanabi_action(action: str) -> Optional[Dict[str, Any]]:
    text = (action or "").strip()
    if not text:
        return None
    match = _PLAY_RE.search(text)
    if match:
        return {"kind": "play", "slot": int(match.group(1))}
    match = _DISCARD_RE.search(text)
    if match:
        return {"kind": "discard", "slot": int(match.group(1))}
    match = _REVEAL_RE.search(text)
    if match:
        return {
            "kind": "reveal",
            "player": int(match.group(1)),
            "slot": int(match.group(2)),
            "color": _coerce_optional_str(match.group(3)),
            "rank": _coerce_optional_int(match.group(4)),
        }
    return None


def _selected_goal_matches_action(goal: GoalRecord, parsed_action: Dict[str, Any], agent_player_id: Optional[int]) -> bool:
    if goal.status != "active":
        return False
    target = _parse_target_ref(goal.target, agent_player_id)
    if not target:
        return False
    if target["kind"] == "slot":
        if target["player"] == agent_player_id:
            return parsed_action.get("kind") in {"play", "discard"} and target["slot"] == parsed_action.get("slot")
        return (
            parsed_action.get("kind") == "reveal"
            and target["player"] == parsed_action.get("player")
            and target["slot"] == parsed_action.get("slot")
        )
    if target["kind"] == "token":
        return target.get("token_type") == "info" and parsed_action.get("kind") == "discard"
    return False


def _parse_target_ref(target: str, agent_player_id: Optional[int]) -> Optional[Dict[str, Any]]:
    text = (target or "").strip()
    if not text or text.lower() == "none":
        return {"kind": "none"}

    match = _SELF_SLOT_TARGET_RE.match(text)
    if match:
        return {
            "kind": "slot",
            "player": agent_player_id,
            "slot": int(match.group(1)),
        }

    match = _SLOT_TARGET_RE.match(text)
    if match:
        return {
            "kind": "slot",
            "player": int(match.group(1)),
            "slot": int(match.group(2)),
        }

    match = _TOKEN_TARGET_RE.match(text)
    if match:
        return {"kind": "token", "token_type": match.group(1).lower()}

    match = _FIREWORK_TARGET_RE.match(text)
    if match:
        return {"kind": "firework", "color": match.group(1).lower()}

    return None


def _slot_target_ref(player: Optional[int], slot: Optional[int], agent_player_id: Optional[int]) -> str:
    if player is None or slot is None:
        return "none"
    if agent_player_id is not None and player == agent_player_id:
        return f"self_slot{slot}"
    return f"player{player}_slot{slot}"


def _coerce_target_ref(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        text = " ".join(value.strip().split())
        return text or "none"
    if isinstance(value, dict):
        player = _coerce_optional_int(value.get("player"))
        slot = _coerce_optional_int(value.get("slot"))
        if player is not None and slot is not None:
            return f"player{player}_slot{slot}"
        entity_type = str(value.get("entity_type") or "").strip().lower()
        if entity_type == "token":
            token_type = _coerce_optional_str(value.get("token_type")) or "token"
            return f"{token_type.lower()}_token"
        if entity_type == "firework":
            color = _coerce_optional_str(value.get("firework_color")) or _coerce_optional_str(value.get("color"))
            return f"firework_{color.lower()}" if color else "firework"
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    text = _coerce_optional_str(value)
    return text or "none"


def _priority_rank(priority: str) -> int:
    return _PRIORITY_ORDER.get(_normalize_priority(priority, default="medium"), _PRIORITY_ORDER["medium"])


def _normalize_priority(value: Any, *, default: str) -> str:
    if isinstance(value, (int, float)):
        if float(value) >= 0.67:
            return "high"
        if float(value) <= 0.33:
            return "low"
        return "medium"
    text = str(value or "").strip().lower()
    if text in _PRIORITY_ORDER:
        return text
    return default


def _infer_player_id(observation: str) -> Optional[int]:
    match = _PLAYER_ID_RE.search(observation or "")
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return _coerce_optional_int(value)


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_positive_int(value: Any, *, default: int) -> int:
    coerced = _coerce_optional_int(value)
    if coerced is None:
        return int(default)
    return max(0, coerced)
