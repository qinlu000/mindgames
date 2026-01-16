from mindgames.core import ActionWrapper, Env

__all__ = ["ActionFormattingWrapper"]

import re


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
        """
        Format the action by adding brackets if they're missing.
        
        This method checks if the action already contains square brackets.
        If not, it wraps the entire action string in square brackets.
        
        Args:
            action (str): The action to format.
            
        Returns:
            str: The formatted action, with brackets added if necessary.
        """
        if not isinstance(action, str):
            action = str(action)

        # Keep only the first non-empty line (LLMs sometimes append explanations).
        for line in action.splitlines():
            if line.strip():
                action = line.strip()
                break
        else:
            action = ""

        if "[" in action and "]" in action:
            return action

        # Hanabi-style formats: "[Play] X", "[Discard] X", "[Reveal] ...".
        # If the model outputs "Play 0" / "Discard: 0" / "Reveal player ...", normalize it.
        m = re.match(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", action, flags=re.IGNORECASE)
        if m:
            verb = m.group(1).capitalize()
            rest = m.group(2).strip()
            return f"[{verb}] {rest}"

        return f"[{action}]"
