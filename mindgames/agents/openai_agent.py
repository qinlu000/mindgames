import os
import re
import time
from json import JSONDecodeError
from typing import Optional

from mindgames.core import Agent

STANDARD_GAME_PROMPT = (
    "You are an expert Hanabi teammate.\n"
    "Output EXACTLY ONE valid action and nothing else (no reasoning).\n\n"
    "Valid formats:\n"
    "- [Play] X\n"
    "- [Discard] X\n"
    "- [Reveal] player N card X color C\n"
    "- [Reveal] player N card X rank R\n\n"
    "Rules (non-standard Hanabi here):\n"
    "- Reveal must target exactly ONE specific card index in another player's hand.\n"
    "- Reveal must be truthful for that specific card.\n"
    "- Do not reveal about yourself.\n"
    "- Use exactly one hint type: color OR rank.\n"
    "- Fireworks are independent; you may play the next required rank of any color.\n\n"
    "Strategy priority:\n"
    "1) If you know a card is playable, [Play] it.\n"
    "2) Else if a teammate has a clearly playable card and info_tokens>0, reveal that exact card.\n"
    "3) Else discard the least useful / most uncertain card.\n"
    "4) Avoid repeating the same Reveal on the same card unless it adds new info."
)
_RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRY_MSG_TOKENS = (
    "empty model response",
    "empty completion choices",
    "timed out",
    "timeout",
    "connection refused",
    "connection error",
)

_ACTION_LINE_RE = re.compile(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", flags=re.IGNORECASE)


def _extract_action_from_text(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    bracket_lines = [ln for ln in lines if "[" in ln and "]" in ln]
    if bracket_lines:
        ln = bracket_lines[-1]
        if "[" in ln:
            return ln[ln.index("[") :].strip()
        return ln.strip()
    for ln in reversed(lines):
        if _ACTION_LINE_RE.match(ln):
            return ln.strip()
    return None


class OpenAIAgent(Agent):
    """Agent class using an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        verbose: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__()
        max_retries = int(kwargs.pop("max_retries", 10))
        retry_delay_s = float(kwargs.pop("retry_delay_s", kwargs.pop("retry_initial_delay_s", 0.0)))
        kwargs.pop("retry_max_delay_s", None)
        kwargs.pop("retry_jitter_s", None)

        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if retry_delay_s < 0:
            raise ValueError(f"retry_delay_s must be >= 0, got {retry_delay_s}")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.kwargs = kwargs
        self.last_message = None
        self.last_usage = None

        try:
            from openai import OpenAI
            from openai import APIConnectionError, APITimeoutError, RateLimitError
            from openai import APIStatusError, BadRequestError, AuthenticationError, NotFoundError, PermissionDeniedError
        except ImportError as exc:
            raise ImportError("OpenAI package is required for OpenAIAgent. Install it with: pip install openai") from exc

        # Use httpx.Timeout to avoid huge connect timeouts when a proxy is down.
        try:
            import httpx  # type: ignore
        except Exception:  # pragma: no cover
            httpx = None

        self._retryable_exceptions = (APIConnectionError, APITimeoutError, RateLimitError, JSONDecodeError, TypeError)
        self._non_retryable_exceptions = (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            BadRequestError,
        )
        self._status_error = APIStatusError

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        connect_timeout_s = float(kwargs.pop("connect_timeout_s", 10.0))
        write_timeout_s = float(kwargs.pop("write_timeout_s", 10.0))
        pool_timeout_s = float(kwargs.pop("pool_timeout_s", 10.0))

        timeout = kwargs.get("timeout", None)
        if httpx is not None and isinstance(timeout, (int, float)) and timeout > 0:
            read_timeout_s = float(timeout)
            kwargs["timeout"] = httpx.Timeout(
                timeout=read_timeout_s,
                connect=min(connect_timeout_s, read_timeout_s),
                read=read_timeout_s,
                write=min(write_timeout_s, read_timeout_s),
                pool=min(pool_timeout_s, read_timeout_s),
            )

        # Disable the OpenAI SDK's internal retries so our retry loop is the single source of truth.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def _is_retryable_error(self, e: Exception) -> bool:
        if isinstance(e, self._non_retryable_exceptions):
            return False
        if isinstance(e, self._retryable_exceptions):
            return True
        if isinstance(e, self._status_error):
            status_code = getattr(e, "status_code", None)
            return status_code in _RETRY_STATUS_CODES if status_code is not None else True
        msg = str(e).lower()
        return any(token in msg for token in _RETRY_MSG_TOKENS)

    def _make_request(self, observation: str) -> str:
        messages = [{"role": "user", "content": observation}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            **self.kwargs,
        )

        choices = getattr(completion, "choices", None)
        if not choices:
            try:
                payload = completion.model_dump()
            except Exception:
                payload = {"type": type(completion).__name__}
            raise RuntimeError(f"Empty completion choices: {payload}")

        msg = choices[0].message
        try:
            self.last_message = msg.model_dump()
        except Exception:
            self.last_message = {"role": getattr(msg, "role", "assistant"), "content": getattr(msg, "content", None)}
        try:
            self.last_usage = completion.usage.model_dump() if completion.usage is not None else None
        except Exception:
            self.last_usage = None

        content = (msg.content or "").strip()
        if not content:
            last = self.last_message or {"role": getattr(msg, "role", "assistant"), "content": getattr(msg, "content", None)}
            reasoning = last.get("reasoning") or last.get("reasoning_content") if isinstance(last, dict) else None
            if isinstance(reasoning, str):
                fallback = _extract_action_from_text(reasoning)
                if fallback:
                    return fallback
            raise RuntimeError(f"Empty model response (no assistant content). last_message={last}")
        return content

    def _retry_request(self, observation: str) -> str:
        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                if not self._is_retryable_error(e) or attempt >= self.max_retries:
                    raise

                print(f"Attempt {attempt} failed with error: {e}")
                if self.retry_delay_s:
                    time.sleep(self.retry_delay_s)

        raise last_exception

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)
