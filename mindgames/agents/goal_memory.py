from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mindgames.core import Agent, AgentWrapper

__all__ = ["GoalMemoryConfig", "GoalMemoryAgentWrapper"]

_ALLOWED_GOAL_TYPES = {
    "play_self_slot",
    "save_partner_card",
    "set_up_partner_play",
    "recover_info",
    "safe_discard_fallback",
}
_ALLOWED_GOAL_STATUSES = {"active", "completed", "invalidated", "expired", "deleted"}
_ALLOWED_ENTITY_TYPES = {"card_slot", "token", "firework", "none"}
_ALLOWED_OPS = {"upsert_goal", "reprioritize_goal", "complete_goal", "invalidate_goal", "delete_goal"}

_PLAY_RE = re.compile(r"\[\s*play\s*\]\s*(\d+)\b", re.IGNORECASE)
_DISCARD_RE = re.compile(r"\[\s*discard\s*\]\s*(\d+)\b", re.IGNORECASE)
_REVEAL_RE = re.compile(
    r"\[\s*reveal\s*\]\s*player\s+(\d+)\s+card\s+(\d+)\s+(?:color\s+([a-z]+)|rank\s+(\d+))",
    re.IGNORECASE,
)
_ACTION_LINE_RE = re.compile(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_PLAYER_ID_RE = re.compile(r"You are player\s+(\d+)\b|You are Player\s+(\d+)\b")


@dataclass
class GoalMemoryConfig:
    max_active_goals: int = 3
    render_topk: int = 2
    default_ttl: int = 2
    max_ops_per_turn: int = 4


@dataclass
class GoalTarget:
    entity_type: str = "none"
    player: Optional[int] = None
    slot: Optional[int] = None
    color: Optional[str] = None
    rank: Optional[int] = None
    token_type: Optional[str] = None
    firework_color: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalTarget":
        payload = dict(data or {})
        entity_type = str(payload.get("entity_type") or "none").strip().lower()
        if entity_type not in _ALLOWED_ENTITY_TYPES:
            entity_type = "none"
        return cls(
            entity_type=entity_type,
            player=_coerce_optional_int(payload.get("player")),
            slot=_coerce_optional_int(payload.get("slot")),
            color=_coerce_optional_str(payload.get("color")),
            rank=_coerce_optional_int(payload.get("rank")),
            token_type=_coerce_optional_str(payload.get("token_type")),
            firework_color=_coerce_optional_str(payload.get("firework_color")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "player": self.player,
            "slot": self.slot,
            "color": self.color,
            "rank": self.rank,
            "token_type": self.token_type,
            "firework_color": self.firework_color,
        }


@dataclass
class GoalRecord:
    goal_id: str
    goal_type: str
    target: GoalTarget
    status: str = "active"
    priority: float = 0.5
    confidence: float = 0.5
    ttl: int = 2
    created_turn: int = 0
    last_updated_turn: int = 0
    reason: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    belief_refs: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    abort_conditions: List[str] = field(default_factory=list)
    source: str = "llm"
    parent_goal_id: Optional[str] = None
    supersedes_goal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {"tags": [], "notes": None})

    def ttl_left(self, current_turn: int) -> int:
        return int(self.ttl) - int(current_turn - self.last_updated_turn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "target": self.target.to_dict(),
            "status": self.status,
            "priority": self.priority,
            "confidence": self.confidence,
            "ttl": self.ttl,
            "created_turn": self.created_turn,
            "last_updated_turn": self.last_updated_turn,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "belief_refs": list(self.belief_refs),
            "preconditions": list(self.preconditions),
            "success_conditions": list(self.success_conditions),
            "abort_conditions": list(self.abort_conditions),
            "source": self.source,
            "parent_goal_id": self.parent_goal_id,
            "supersedes_goal_id": self.supersedes_goal_id,
            "metadata": dict(self.metadata),
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
        goals.sort(key=lambda g: (-g.priority, -g.last_updated_turn, g.goal_id))
        return goals

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": "goal_memory.v1",
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
            return "Active goals:\n- (none)"

        labels = ["PRIMARY", "SUPPORT", "FALLBACK"]
        lines = ["Active goals:"]
        for idx, goal in enumerate(active[: self.config.render_topk]):
            label = labels[idx] if idx < len(labels) else f"GOAL-{idx + 1}"
            age = max(0, self.current_turn - goal.created_turn)
            ttl_left = goal.ttl_left(self.current_turn)
            timing = "new" if age == 0 else "carry-over"
            if ttl_left <= 0:
                timing = f"{timing}, expiring"
            target = _render_target(goal.target)
            lines.append(f"- {label} [{timing}, ttl_left={ttl_left}]: {goal.goal_type} -> {target}")
            if goal.reason:
                lines.append(f"  why: {goal.reason}")
            abort_preview = next((item for item in goal.abort_conditions if item), None)
            if abort_preview:
                lines.append(f"  abort if: {abort_preview}")
        return "\n".join(lines)

    def apply_goal_ops(self, goal_ops: List[Dict[str, Any]], *, source: str = "llm") -> List[Dict[str, Any]]:
        applied: List[Dict[str, Any]] = []
        for raw_op in list(goal_ops or [])[: self.config.max_ops_per_turn]:
            op_name = str(raw_op.get("op") or "").strip()
            if op_name not in _ALLOWED_OPS:
                continue
            goal_id = str(raw_op.get("goal_id") or "").strip()
            if not goal_id:
                continue
            if op_name == "upsert_goal":
                event = self._upsert_goal(goal_id, raw_op, source=source)
            elif op_name == "reprioritize_goal":
                event = self._reprioritize_goal(goal_id, raw_op, source=source)
            elif op_name == "complete_goal":
                event = self._transition_goal(goal_id, "completed", raw_op, source=source)
            elif op_name == "invalidate_goal":
                event = self._transition_goal(goal_id, "invalidated", raw_op, source=source)
            else:
                event = self._transition_goal(goal_id, "deleted", raw_op, source=source)
            if event is not None:
                applied.append(event)
        self._trim_active_goals()
        return applied

    def record_action_result(self, action: str, *, selected_goal_id: Optional[str]) -> List[Dict[str, Any]]:
        parsed = _parse_hanabi_action(action)
        if not parsed:
            return []

        events: List[Dict[str, Any]] = []
        matches: set[str] = set()
        for goal in self.active_goals():
            if _goal_matches_action(goal, parsed, self.agent_player_id):
                matches.add(goal.goal_id)
                reason = f"action matched goal via {parsed['kind']}"
                event = self._set_goal_status(goal, "completed", reason=reason, source="system")
                if event is not None:
                    events.append(event)

        if parsed["kind"] in {"play", "discard"} and self.agent_player_id is not None:
            affected_slot = int(parsed["slot"])
            for goal in self.active_goals():
                target = goal.target
                if target.entity_type != "card_slot" or target.player != self.agent_player_id or target.slot is None:
                    continue
                if goal.goal_id in matches:
                    continue
                if target.slot == affected_slot:
                    event = self._set_goal_status(
                        goal,
                        "invalidated",
                        reason=f"target slot {affected_slot} was consumed by {parsed['kind']}",
                        source="system",
                    )
                    if event is not None:
                        events.append(event)
                elif target.slot > affected_slot:
                    old_slot = target.slot
                    target.slot -= 1
                    goal.metadata["last_rebased_turn"] = self.current_turn
                    event = self._record_event(
                        goal.goal_id,
                        "rebase_goal",
                        "system",
                        before_status=goal.status,
                        after_status=goal.status,
                        reason=f"shifted slot from {old_slot} to {target.slot} after {parsed['kind']} {affected_slot}",
                    )
                    events.append(event)

        if selected_goal_id:
            selected = self.goals.get(selected_goal_id)
            if selected and selected.status == "active" and selected.goal_type == "recover_info":
                if parsed["kind"] == "discard":
                    event = self._set_goal_status(
                        selected,
                        "completed",
                        reason="discard used to recover information token",
                        source="system",
                    )
                    if event is not None:
                        events.append(event)

        return events

    def _expire_stale_goals(self) -> None:
        for goal in list(self.goals.values()):
            if goal.status != "active":
                continue
            if goal.ttl_left(self.current_turn) < 0:
                self._set_goal_status(goal, "expired", reason="ttl expired", source="system")

    def _upsert_goal(self, goal_id: str, raw_op: Dict[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
        payload = dict(raw_op.get("goal") or {})
        existing = self.goals.get(goal_id)
        if existing is None:
            goal_type = str(payload.get("goal_type") or "").strip()
            if goal_type not in _ALLOWED_GOAL_TYPES:
                return None
            target = GoalTarget.from_dict(payload.get("target"))
            if not _target_is_valid(goal_type, target, self.agent_player_id):
                return None
            goal = GoalRecord(
                goal_id=goal_id,
                goal_type=goal_type,
                target=target,
                status="active",
                priority=_clamp01(payload.get("priority"), default=0.5),
                confidence=_clamp01(payload.get("confidence"), default=0.5),
                ttl=_coerce_positive_int(payload.get("ttl"), default=self.config.default_ttl),
                created_turn=self.current_turn,
                last_updated_turn=self.current_turn,
                reason=_coerce_optional_str(payload.get("reason")) or _coerce_optional_str(raw_op.get("reason")) or "",
                evidence_refs=_normalize_str_list(payload.get("evidence_refs") or raw_op.get("evidence_refs")),
                belief_refs=_normalize_str_list(payload.get("belief_refs")),
                preconditions=_normalize_str_list(payload.get("preconditions")),
                success_conditions=_normalize_str_list(payload.get("success_conditions")),
                abort_conditions=_normalize_str_list(payload.get("abort_conditions")),
                source=source,
                parent_goal_id=_coerce_optional_str(payload.get("parent_goal_id")),
                supersedes_goal_id=_coerce_optional_str(payload.get("supersedes_goal_id")),
                metadata=_normalize_metadata(payload.get("metadata")),
            )
            self.goals[goal_id] = goal
            return self._record_event(goal_id, "upsert_goal", source, before_status=None, after_status="active", reason=goal.reason)

        goal_payload_type = _coerce_optional_str(payload.get("goal_type")) or existing.goal_type
        target = GoalTarget.from_dict(payload.get("target")) if payload.get("target") is not None else existing.target
        if goal_payload_type not in _ALLOWED_GOAL_TYPES or not _target_is_valid(goal_payload_type, target, self.agent_player_id):
            return None
        before_status = existing.status
        existing.goal_type = goal_payload_type
        existing.target = target
        existing.status = "active"
        existing.priority = _clamp01(payload.get("priority"), default=existing.priority)
        existing.confidence = _clamp01(payload.get("confidence"), default=existing.confidence)
        existing.ttl = _coerce_positive_int(payload.get("ttl"), default=existing.ttl)
        existing.last_updated_turn = self.current_turn
        existing.reason = _coerce_optional_str(payload.get("reason")) or _coerce_optional_str(raw_op.get("reason")) or existing.reason
        existing.evidence_refs = _normalize_str_list(payload.get("evidence_refs") or raw_op.get("evidence_refs")) or existing.evidence_refs
        existing.belief_refs = _normalize_str_list(payload.get("belief_refs")) or existing.belief_refs
        existing.preconditions = _normalize_str_list(payload.get("preconditions")) or existing.preconditions
        existing.success_conditions = _normalize_str_list(payload.get("success_conditions")) or existing.success_conditions
        existing.abort_conditions = _normalize_str_list(payload.get("abort_conditions")) or existing.abort_conditions
        existing.source = source
        existing.parent_goal_id = _coerce_optional_str(payload.get("parent_goal_id")) or existing.parent_goal_id
        existing.supersedes_goal_id = _coerce_optional_str(payload.get("supersedes_goal_id")) or existing.supersedes_goal_id
        if payload.get("metadata") is not None:
            existing.metadata = _normalize_metadata(payload.get("metadata"))
        return self._record_event(goal_id, "upsert_goal", source, before_status=before_status, after_status=existing.status, reason=existing.reason)

    def _reprioritize_goal(self, goal_id: str, raw_op: Dict[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        goal.priority = _clamp01(raw_op.get("new_priority"), default=goal.priority)
        goal.last_updated_turn = self.current_turn
        reason = _coerce_optional_str(raw_op.get("reason")) or "reprioritized"
        return self._record_event(goal_id, "reprioritize_goal", source, before_status=goal.status, after_status=goal.status, reason=reason)

    def _transition_goal(self, goal_id: str, status: str, raw_op: Dict[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        reason = _coerce_optional_str(raw_op.get("reason")) or status
        return self._set_goal_status(goal, status, reason=reason, source=source)

    def _set_goal_status(self, goal: GoalRecord, status: str, *, reason: str, source: str) -> Optional[Dict[str, Any]]:
        if status not in _ALLOWED_GOAL_STATUSES:
            return None
        before_status = goal.status
        goal.status = status
        if status in {"completed", "invalidated", "deleted"}:
            goal.last_updated_turn = self.current_turn
        return self._record_event(goal.goal_id, status, source, before_status=before_status, after_status=status, reason=reason)

    def _trim_active_goals(self) -> None:
        active = self.active_goals()
        keep = set(goal.goal_id for goal in active[: self.config.max_active_goals])
        for goal in active[self.config.max_active_goals :]:
            if goal.goal_id in keep:
                continue
            self._set_goal_status(goal, "deleted", reason="trimmed to active-goal budget", source="system")

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
        return (
            "Goal memory is enabled. Maintain a small typed working agenda across your turns.\n"
            "If you change, complete, invalidate, or reprioritize a goal, do it through goal_ops explicitly.\n"
            f"Keep at most {self.config.max_active_goals} active goals and at most {self.config.max_ops_per_turn} goal_ops this turn.\n"
            "Return EXACTLY ONE JSON object with this schema:\n"
            "{\n"
            '  "selected_goal_id": "string or null",\n'
            '  "goal_ops": [\n'
            "    {\n"
            '      "op": "upsert_goal|reprioritize_goal|complete_goal|invalidate_goal|delete_goal",\n'
            '      "goal_id": "string",\n'
            '      "reason": "string",\n'
            '      "goal": {\n'
            '        "goal_type": "play_self_slot|save_partner_card|set_up_partner_play|recover_info|safe_discard_fallback",\n'
            '        "target": {"entity_type": "card_slot|token|firework|none", "player": 0, "slot": 0},\n'
            '        "priority": 0.0,\n'
            '        "confidence": 0.0,\n'
            '        "ttl": 1,\n'
            '        "reason": "short reason",\n'
            '        "belief_refs": [],\n'
            '        "preconditions": [],\n'
            '        "success_conditions": [],\n'
            '        "abort_conditions": []\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "action": "[Play] X | [Discard] X | [Reveal] player N card X color C | [Reveal] player N card X rank R"\n'
            "}\n"
            "Return JSON only; do not add any prose.\n\n"
            f"{self.state.render()}\n\n"
            "Game observation:\n"
            f"{observation}"
        )

    def _call_wrapped_agent(self, prompt: str) -> str:
        original_system_prompt = getattr(self.agent, "system_prompt", None)
        if original_system_prompt is not None:
            self.agent.system_prompt = (
                f"{original_system_prompt}\n\n"
                "When goal memory is enabled, do not return free-form text. Return exactly one JSON object."
            )
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


def _goal_matches_action(goal: GoalRecord, parsed_action: Dict[str, Any], agent_player_id: Optional[int]) -> bool:
    target = goal.target
    if goal.status != "active":
        return False
    if goal.goal_type == "play_self_slot":
        return (
            parsed_action.get("kind") == "play"
            and target.entity_type == "card_slot"
            and target.player == agent_player_id
            and target.slot == parsed_action.get("slot")
        )
    if goal.goal_type == "safe_discard_fallback":
        return (
            parsed_action.get("kind") == "discard"
            and target.entity_type == "card_slot"
            and target.player == agent_player_id
            and target.slot == parsed_action.get("slot")
        )
    if goal.goal_type in {"save_partner_card", "set_up_partner_play"}:
        return (
            parsed_action.get("kind") == "reveal"
            and target.entity_type == "card_slot"
            and target.player == parsed_action.get("player")
            and target.slot == parsed_action.get("slot")
        )
    return False


def _target_is_valid(goal_type: str, target: GoalTarget, agent_player_id: Optional[int]) -> bool:
    if goal_type not in _ALLOWED_GOAL_TYPES:
        return False
    if goal_type == "recover_info":
        return target.entity_type == "token" and target.token_type in {None, "info"}
    if goal_type in {"play_self_slot", "safe_discard_fallback"}:
        return target.entity_type == "card_slot" and target.player == agent_player_id and target.slot is not None
    if goal_type in {"save_partner_card", "set_up_partner_play"}:
        return (
            target.entity_type == "card_slot"
            and target.player is not None
            and target.player != agent_player_id
            and target.slot is not None
        )
    return True


def _render_target(target: GoalTarget) -> str:
    if target.entity_type == "card_slot":
        return f"player {target.player} slot {target.slot}"
    if target.entity_type == "token":
        return f"{target.token_type or 'token'} token"
    if target.entity_type == "firework":
        return f"firework {target.firework_color or '?'}"
    return "none"


def _infer_player_id(observation: str) -> Optional[int]:
    match = _PLAYER_ID_RE.search(observation or "")
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return _coerce_optional_int(value)


def _normalize_metadata(value: Any) -> Dict[str, Any]:
    payload = dict(value or {}) if isinstance(value, dict) else {}
    tags = payload.get("tags")
    notes = payload.get("notes")
    return {
        "tags": _normalize_str_list(tags),
        "notes": _coerce_optional_str(notes),
    }


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        text = _coerce_optional_str(item)
        if text:
            items.append(text)
    return items


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


def _clamp01(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, numeric))
