import os
import random
import time
from typing import Optional

from mindgames.core import Agent

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."


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
        retry_initial_delay_s = float(kwargs.pop("retry_initial_delay_s", 2.0))
        retry_max_delay_s = float(kwargs.pop("retry_max_delay_s", 60.0))
        retry_jitter_s = float(kwargs.pop("retry_jitter_s", 0.25))

        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        if retry_initial_delay_s < 0:
            raise ValueError(f"retry_initial_delay_s must be >= 0, got {retry_initial_delay_s}")
        if retry_max_delay_s < 0:
            raise ValueError(f"retry_max_delay_s must be >= 0, got {retry_max_delay_s}")
        if retry_max_delay_s and retry_max_delay_s < retry_initial_delay_s:
            raise ValueError("retry_max_delay_s must be >= retry_initial_delay_s")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_initial_delay_s = retry_initial_delay_s
        self.retry_max_delay_s = retry_max_delay_s
        self.retry_jitter_s = retry_jitter_s
        self.kwargs = kwargs
        self.last_message = None
        self.last_usage = None

        try:
            from openai import OpenAI
            from openai import APIConnectionError, APITimeoutError, RateLimitError
            from openai import APIStatusError, BadRequestError, AuthenticationError, NotFoundError, PermissionDeniedError
        except ImportError:
            raise ImportError("OpenAI package is required for OpenAIAgent. Install it with: pip install openai")

        self._openai_exceptions = {
            "APIConnectionError": APIConnectionError,
            "APITimeoutError": APITimeoutError,
            "RateLimitError": RateLimitError,
            "APIStatusError": APIStatusError,
            "BadRequestError": BadRequestError,
            "AuthenticationError": AuthenticationError,
            "NotFoundError": NotFoundError,
            "PermissionDeniedError": PermissionDeniedError,
        }

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _is_retryable_error(self, e: Exception) -> bool:
        exc = self._openai_exceptions
        if isinstance(e, (exc["AuthenticationError"], exc["PermissionDeniedError"], exc["NotFoundError"], exc["BadRequestError"])):
            return False
        if isinstance(e, (exc["APIConnectionError"], exc["APITimeoutError"], exc["RateLimitError"])):
            return True
        if isinstance(e, exc["APIStatusError"]):
            status_code = getattr(e, "status_code", None)
            # Retry transient HTTP statuses.
            return status_code in {408, 409, 429, 500, 502, 503, 504} if status_code is not None else True
        # Fallback: network-ish exceptions without OpenAI typed wrappers.
        msg = str(e).lower()
        if "timed out" in msg or "timeout" in msg:
            return True
        if "connection refused" in msg or "connection error" in msg:
            return True
        return False

    def _make_request(self, observation: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": observation},
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            stop=None,
            **self.kwargs,
        )

        msg = completion.choices[0].message
        try:
            self.last_message = msg.model_dump()
        except Exception:
            self.last_message = {"role": getattr(msg, "role", "assistant"), "content": getattr(msg, "content", None)}
        try:
            self.last_usage = completion.usage.model_dump() if completion.usage is not None else None
        except Exception:
            self.last_usage = None

        return (msg.content or "").strip()

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

                delay = self.retry_initial_delay_s * (2 ** (attempt - 1))
                if self.retry_max_delay_s:
                    delay = min(delay, self.retry_max_delay_s)
                delay += random.random() * (self.retry_jitter_s * delay)
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(delay)

        raise last_exception

    def __call__(self, observation: str) -> str:
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)
