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

        # Extract a single action line from potentially multi-line model output.
        # Prefer the *last* action-looking line (many models place reasoning first, then the final action).
        lines = [ln.strip() for ln in action.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Prefer bracketed actions; if the action isn't at the start of the line (e.g. "Final Answer: [Play] 0"),
        # slice from the first "[" to reduce leading chatter.
        bracket_lines: list[str] = []
        for ln in lines:
            if "[" in ln and "]" in ln:
                bracket_lines.append(ln)
        if bracket_lines:
            ln = bracket_lines[-1]
            if "[" in ln:
                action = ln[ln.index("[") :].strip()
            else:
                action = ln
        else:
            # Hanabi-style formats: "[Play] X", "[Discard] X", "[Reveal] ...".
            # If the model outputs "Play 0" / "Discard: 0" / "Reveal player ...", normalize it.
            # Otherwise, fall back to the last non-empty line.
            verb_lines: list[str] = []
            verb_re = re.compile(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", flags=re.IGNORECASE)
            for ln in lines:
                if verb_re.match(ln):
                    verb_lines.append(ln)
            action = (verb_lines[-1] if verb_lines else lines[-1]).strip()

        if "[" in action and "]" in action:
            return action

        m = re.match(r"^\s*(play|discard|reveal)\s*[:\-]?\s+(.+?)\s*$", action, flags=re.IGNORECASE)
        if m:
            verb = m.group(1).capitalize()
            rest = m.group(2).strip()
            return f"[{verb}] {rest}"

        return f"[{action}]"
