from __future__ import annotations

from typing import Any, Optional

from mindgames.envs.registration import get_action_parser, get_prompt_profile
from mindgames.prompting.action_extraction import normalize_model_action


def get_legal_actions_for_env(env: Any, observation: str) -> Optional[list[str]]:
    action_parser = get_action_parser(env)
    if action_parser is None:
        return None

    try:
        actions = action_parser(observation, env=env)
    except TypeError:
        actions = action_parser(observation)

    if actions is None:
        return None
    return [str(action) for action in actions if isinstance(action, str) and action.strip()]

def normalize_action_for_env(env: Any, observation: str, raw_action: str) -> str:
    prompt_profile = get_prompt_profile(env)
    legal_actions = get_legal_actions_for_env(env, observation)
    return normalize_model_action(
        raw_action,
        prompt_profile=prompt_profile,
        legal_actions=legal_actions,
    )
