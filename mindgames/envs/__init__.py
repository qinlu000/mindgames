"""Register selected environments (mirrors spiral-rl/spiral layout style)."""

from mindgames.envs.registration import register_with_versions
from mindgames.wrappers import (
    LLMObservationWrapper,
    ActionFormattingWrapper,
    GameMessagesAndCurrentBoardObservationWrapper,
    ClipCharactersActionWrapper,
)

DEFAULT_WRAPPERS = [LLMObservationWrapper, ActionFormattingWrapper]
BOARDGAME_WRAPPERS = [GameMessagesAndCurrentBoardObservationWrapper, ActionFormattingWrapper]
CONVERSATIONAL_WRAPPERS = [LLMObservationWrapper, ClipCharactersActionWrapper]

# Hanabi (co-op)
register_with_versions(
    id="Hanabi-v0",
    entry_point="mindgames.envs.Hanabi.env:HanabiEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
    info_tokens=8,
    fuse_tokens=4,
)

# Truth & Deception (2-player)
register_with_versions(
    id="TruthAndDeception-v0",
    entry_point="mindgames.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    wrappers={"default": [LLMObservationWrapper], "-train": CONVERSATIONAL_WRAPPERS},
    max_turns=6,
)
register_with_versions(
    id="TruthAndDeception-v0-long",
    entry_point="mindgames.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    wrappers={"default": [LLMObservationWrapper], "-train": CONVERSATIONAL_WRAPPERS},
    max_turns=12,
)
