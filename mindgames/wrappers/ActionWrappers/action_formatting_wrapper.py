from mindgames.core import ActionWrapper, Env
from mindgames.envs.registration import get_prompt_profile
from mindgames.prompting.action_extraction import normalize_model_action

__all__ = ["ActionFormattingWrapper"]


class ActionFormattingWrapper(ActionWrapper):
    """
    A wrapper that formats actions by adding brackets if they're missing.
    
    This wrapper ensures that all actions follow a consistent format by wrapping
    them in square brackets if they don't already contain brackets. This is useful
    for environments that require actions to be enclosed in brackets but where
    agents might not always follow this convention.
    
    Example:
        - Input: "move north"
        - Output: "[move north]"
        
        - Input: "[trade wheat]"
        - Output: "[trade wheat]" (unchanged)
    """

    def __init__(self, env: Env):
        """
        Initialize the ActionFormattingWrapper.
        
        Args:
            env (Env): The environment to wrap.
        """
        super().__init__(env)

    def action(self, action: str) -> str:
        try:
            prompt_profile = get_prompt_profile(self.env)
        except Exception:
            prompt_profile = None
        return normalize_model_action(action, prompt_profile=prompt_profile)
