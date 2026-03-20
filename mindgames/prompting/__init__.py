from mindgames.prompting.action_extraction import (
    extract_boxed_answer,
    extract_raw_action,
    normalize_chat_action_text,
    normalize_action_text,
    normalize_model_action,
    select_legal_action,
)
from mindgames.prompting.action_parsers import (
    colonel_blotto_parse_available_actions,
    mini_hanabi_parse_available_actions,
    negotiation_parse_available_actions,
)
from mindgames.prompting.templates import (
    COLONEL_BLOTTO_SYSTEM_PROMPT,
    DEFAULT_COMPETITIVE_SYSTEM_PROMPT,
    DEFAULT_COOPERATIVE_SYSTEM_PROMPT,
    MINI_HANABI_SYSTEM_PROMPT,
    NEGOTIATION_SYSTEM_PROMPT,
    PromptProfile,
    TEMPLATE_FACTORY,
    render_prompt,
)

__all__ = [
    "DEFAULT_COMPETITIVE_SYSTEM_PROMPT",
    "DEFAULT_COOPERATIVE_SYSTEM_PROMPT",
    "COLONEL_BLOTTO_SYSTEM_PROMPT",
    "MINI_HANABI_SYSTEM_PROMPT",
    "NEGOTIATION_SYSTEM_PROMPT",
    "PromptProfile",
    "TEMPLATE_FACTORY",
    "render_prompt",
    "extract_boxed_answer",
    "extract_raw_action",
    "normalize_chat_action_text",
    "normalize_action_text",
    "normalize_model_action",
    "select_legal_action",
    "colonel_blotto_parse_available_actions",
    "mini_hanabi_parse_available_actions",
    "negotiation_parse_available_actions",
    "apply_action_wrappers",
    "get_legal_actions_for_env",
    "normalize_action_for_env",
]


def apply_action_wrappers(env, action):
    from mindgames.prompting.normalization import apply_action_wrappers as _impl

    return _impl(env, action)


def get_legal_actions_for_env(env, observation):
    from mindgames.prompting.normalization import get_legal_actions_for_env as _impl

    return _impl(env, observation)


def normalize_action_for_env(env, observation, raw_action):
    from mindgames.prompting.normalization import normalize_action_for_env as _impl

    return _impl(env, observation, raw_action)
