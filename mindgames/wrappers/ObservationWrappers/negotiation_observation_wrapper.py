from typing import Dict, List, Optional, Tuple

from mindgames.core import GAME_ID, ObservationType, ObservationWrapper, Observations


class NegotiationObservationWrapper(ObservationWrapper):
    MAX_PUBLIC_EVENTS = 12
    MAX_PUBLIC_EVENT_CHARS = 240
    MAX_PUBLIC_HISTORY_CHARS = 1_800
    MAX_ADMIN_MESSAGES = 2
    MAX_ADMIN_MESSAGE_CHARS = 240

    def __init__(self, env):
        super().__init__(env)
        self.full_observations: Dict[int, List[Tuple[int, str, ObservationType]]] = {}
        self.max_public_events = int(
            getattr(env, "observation_max_public_events", self.MAX_PUBLIC_EVENTS)
        )
        self.max_public_event_chars = int(
            getattr(env, "observation_max_public_event_chars", self.MAX_PUBLIC_EVENT_CHARS)
        )
        self.max_public_history_chars = int(
            getattr(env, "observation_max_public_history_chars", self.MAX_PUBLIC_HISTORY_CHARS)
        )
        self.max_admin_messages = int(
            getattr(env, "observation_max_admin_messages", self.MAX_ADMIN_MESSAGES)
        )
        self.max_admin_message_chars = int(
            getattr(env, "observation_max_admin_message_chars", self.MAX_ADMIN_MESSAGE_CHARS)
        )

    def _sender_name(self, sender_id: int) -> str:
        if sender_id == GAME_ID:
            return "GAME"
        return self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")

    def _clip_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        suffix = " ... [truncated]"
        if max_chars <= len(suffix):
            return text[:max_chars]
        return text[: max_chars - len(suffix)].rstrip() + suffix

    def _trim_history(self, entries: List[str], max_chars: int) -> List[str]:
        if not entries:
            return entries

        trimmed: List[str] = []
        total_chars = 0
        for entry in reversed(entries):
            entry_len = len(entry)
            if trimmed and total_chars + entry_len > max_chars:
                break
            if not trimmed and entry_len > max_chars:
                clipped_entry = self._clip_text(entry, max_chars)
                trimmed.append(clipped_entry)
                break
            trimmed.append(entry)
            total_chars += entry_len
        return list(reversed(trimmed))

    def _convert_obs_to_str(self, player_id: int) -> str:
        prompt = None
        latest_state = None
        public_events: List[str] = []
        admin_messages: List[str] = []

        for sender_id, message, obs_type in self.full_observations.get(player_id, []):
            sender_name = self._sender_name(sender_id)
            if obs_type == ObservationType.PROMPT:
                prompt = message
            elif obs_type == ObservationType.GAME_BOARD:
                latest_state = message
            elif obs_type == ObservationType.GAME_ADMIN:
                admin_messages.append(
                    self._clip_text(f"[GAME] {message}", self.max_admin_message_chars)
                )
            elif obs_type in {
                ObservationType.PLAYER_ACTION,
                ObservationType.GAME_ACTION_DESCRIPTION,
                ObservationType.GAME_MESSAGE,
            }:
                public_events.append(
                    self._clip_text(
                        f"[{sender_name}] {message}",
                        self.max_public_event_chars,
                    )
                )

        if prompt is None:
            raise ValueError("NegotiationObservationWrapper requires a PROMPT observation.")

        parts: List[str] = [prompt]
        if public_events:
            if len(public_events) > self.max_public_events:
                public_events = public_events[-self.max_public_events :]
            public_events = self._trim_history(public_events, self.max_public_history_chars)
            parts.append("Recent public history (oldest -> newest):\n" + "\n".join(public_events))
        if latest_state is not None:
            parts.append(latest_state)
        if admin_messages:
            admin_messages = admin_messages[-self.max_admin_messages :]
            parts.append("Admin notes:\n" + "\n".join(admin_messages))
        return "\n\n".join(parts)

    def observation(self, player_id: int, observation: Optional[Observations]):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        if player_id not in self.full_observations:
            self.full_observations[player_id] = []
        self.full_observations[player_id].extend(observation)
        return self._convert_obs_to_str(player_id=player_id)
