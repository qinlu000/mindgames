from mindgames.core import ActionWrapper, Env
from mindgames.envs.registration import get_prompt_profile
from mindgames.prompting.action_extraction import normalize_model_action

__all__ = ["ActionFormattingWrapper"]


class ActionFormattingWrapper(ActionWrapper):
    """
    Normalize model output into a single action string.

    This wrapper now does more than just add brackets: it strips common
    reasoning/template artifacts and applies prompt-profile-aware action
    normalization before the action reaches the underlying environment.
    """

    def __init__(self, env: Env):
        super().__init__(env)

    def action(self, action: str) -> str:
        try:
            prompt_profile = get_prompt_profile(self.env)
        except Exception:
            prompt_profile = None
        return normalize_model_action(action, prompt_profile=prompt_profile)
