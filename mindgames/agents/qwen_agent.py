from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from mindgames.agents.openai_agent import OpenAIAgent, STANDARD_GAME_PROMPT


class QwenAgent(OpenAIAgent):
    """A small OpenAI-compatible agent tuned for Qwen/Qwen-VL via vLLM."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        *,
        enable_thinking: Optional[bool] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        eb = dict(kwargs.pop("extra_body", None) or {})
        if extra_body:
            eb.update(extra_body)
        if enable_thinking is not None or chat_template_kwargs is not None:
            ctk = dict(eb.get("chat_template_kwargs") or {})
            if chat_template_kwargs:
                ctk.update(chat_template_kwargs)
            if enable_thinking is not None:
                ctk["enable_thinking"] = bool(enable_thinking)
            eb["chat_template_kwargs"] = ctk
        if eb:
            kwargs["extra_body"] = eb

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            verbose=verbose,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        self.last_response = None
        self.last_content = None
        self.last_reasoning = None

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

        try:
            self.last_response = completion.model_dump()
        except Exception:
            self.last_response = None

        msg = completion.choices[0].message
        try:
            self.last_message = msg.model_dump()
        except Exception:
            self.last_message = {"role": getattr(msg, "role", "assistant"), "content": getattr(msg, "content", None)}

        self.last_content = getattr(msg, "content", None)
        self.last_reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

        try:
            self.last_usage = completion.usage.model_dump() if completion.usage is not None else None
        except Exception:
            self.last_usage = None

        return (self.last_content or "").strip()

    def get_last_content_reasoning(self) -> Tuple[Optional[str], Optional[str]]:
        return (
            self.last_content if isinstance(self.last_content, str) else None,
            self.last_reasoning if isinstance(self.last_reasoning, str) else None,
        )
