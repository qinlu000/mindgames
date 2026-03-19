import re

from mindgames.core import ActionWrapper, Env


class NegotiationActionClipWrapper(ActionWrapper):
    OFFER_RE = re.compile(r"\[offer\s*:\s*(.*?)\s*->\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    ACCEPT_RE = re.compile(r"\[accept\]", re.IGNORECASE)
    DENY_RE = re.compile(r"\[deny\]", re.IGNORECASE)

    def __init__(self, env: Env, max_num_characters: int = 1_000):
        super().__init__(env)
        self.max_num_characters = max_num_characters

    def action(self, action: str) -> str:
        action = str(action)
        if len(action) <= self.max_num_characters:
            return action

        control_parts = []
        protected_spans = []

        for pattern, token in ((self.ACCEPT_RE, "[Accept]"), (self.DENY_RE, "[Deny]")):
            for match in pattern.finditer(action):
                control_parts.append(token)
                protected_spans.append(match.span())

        offer_match = self.OFFER_RE.search(action)
        if offer_match is not None:
            control_parts.append(
                f"[Offer: {offer_match.group(1).strip()} -> {offer_match.group(2).strip()}]"
            )
            protected_spans.append(offer_match.span())

        if not control_parts:
            return action[: self.max_num_characters]

        plain_segments = []
        last_idx = 0
        for start, end in sorted(protected_spans):
            if start > last_idx:
                plain_segments.append(action[last_idx:start])
            last_idx = max(last_idx, end)
        if last_idx < len(action):
            plain_segments.append(action[last_idx:])
        plain_text = " ".join("".join(plain_segments).split())

        prefix = " ".join(control_parts)
        if len(prefix) >= self.max_num_characters:
            return prefix[: self.max_num_characters]
        if not plain_text:
            return prefix

        remaining = self.max_num_characters - len(prefix) - 1
        if remaining <= 0:
            return prefix[: self.max_num_characters]
        return f"{prefix} {plain_text[:remaining].rstrip()}"
