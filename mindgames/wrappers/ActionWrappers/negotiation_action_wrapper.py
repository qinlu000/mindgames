import re

from mindgames.core import ActionWrapper, Env


class NegotiationActionClipWrapper(ActionWrapper):
    OFFER_RE = re.compile(r"\[offer\s*:\s*(.*?)\s*->\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    ACCEPT_RE = re.compile(r"\[accept\]", re.IGNORECASE)
    DENY_RE = re.compile(r"\[deny\]", re.IGNORECASE)

    def __init__(self, env: Env, max_num_characters: int = 1_000):
        super().__init__(env)
        self.max_num_characters = max_num_characters

    def _extract_prefixed_control_segment(self, action: str) -> tuple[str, str]:
        cursor = 0
        while cursor < len(action) and action[cursor].isspace():
            cursor += 1

        first_control_start = cursor
        prefix_end = None

        while cursor < len(action):
            while cursor < len(action) and action[cursor].isspace():
                cursor += 1
            if cursor >= len(action):
                break

            tail = action[cursor:]
            match = None
            for pattern in (self.ACCEPT_RE, self.DENY_RE, self.OFFER_RE):
                match = pattern.match(tail)
                if match is not None:
                    break

            if match is None:
                break

            cursor += match.end()
            prefix_end = cursor

        if prefix_end is None:
            return "", action

        prefix = action[first_control_start:prefix_end].strip()
        plain_text = action[prefix_end:].strip()
        return prefix, plain_text

    def action(self, action: str) -> str:
        action = str(action)
        if len(action) <= self.max_num_characters:
            return action

        prefix, plain_text = self._extract_prefixed_control_segment(action)
        if not prefix:
            return action[: self.max_num_characters]

        if len(prefix) >= self.max_num_characters:
            return prefix[: self.max_num_characters]
        if not plain_text:
            return prefix

        remaining = self.max_num_characters - len(prefix) - 1
        if remaining <= 0:
            return prefix[: self.max_num_characters]
        return f"{prefix} {plain_text[:remaining].rstrip()}"
