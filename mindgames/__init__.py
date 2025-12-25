"""Minimal TextArena-like API for a small set of mindgames."""

from mindgames.__about__ import __version__

from mindgames.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    RenderWrapper,
    ActionWrapper,
    Agent,
    AgentWrapper,
    State,
    Message,
    Observations,
    Rewards,
    Info,
    GAME_ID,
    ObservationType,
)
from mindgames.state import (
    SinglePlayerState,
    TwoPlayerState,
    FFAMultiPlayerState,
    TeamMultiPlayerState,
    MinimalMultiPlayerState,
)
from mindgames.envs.registration import ENV_REGISTRY, make, register, register_with_versions, check_env_exists

# Register selected environments
import mindgames.envs  # noqa: F401

from mindgames import agents, wrappers  # noqa: F401

__all__ = [
    "__version__",
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "RenderWrapper",
    "ActionWrapper",
    "Agent",
    "AgentWrapper",
    "State",
    "Message",
    "Observations",
    "Rewards",
    "Info",
    "GAME_ID",
    "ObservationType",
    "SinglePlayerState",
    "TwoPlayerState",
    "FFAMultiPlayerState",
    "TeamMultiPlayerState",
    "MinimalMultiPlayerState",
    "ENV_REGISTRY",
    "make",
    "register",
    "register_with_versions",
    "check_env_exists",
    "agents",
    "wrappers",
]
