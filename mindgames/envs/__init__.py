"""Register selected environments (mirrors spiral-rl/spiral layout style)."""

from mindgames.envs.registration import register_with_versions
from mindgames.wrappers import (
    LLMObservationWrapper,
    ActionFormattingWrapper,
    GameMessagesAndCurrentBoardObservationWrapper,
    GameMessagesObservationWrapper,
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

# Hanabi (standard hinting)
register_with_versions(
    id="HanabiStandard-v0",
    entry_point="mindgames.envs.Hanabi.env_standard:HanabiStandardEnv",
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

# Truth & Deception with ToM scenario-backed facts (no world knowledge).
register_with_versions(
    id="TruthAndDeceptionToM-v0",
    entry_point="mindgames.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    wrappers={"default": [LLMObservationWrapper], "-train": CONVERSATIONAL_WRAPPERS},
    max_turns=6,
    data_path="mindgames/envs/TruthAndDeception/facts_tom.json",
    reveal_context_to_guesser=True,
)
register_with_versions(
    id="TruthAndDeceptionToM-v0-private",
    entry_point="mindgames.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    wrappers={"default": [LLMObservationWrapper], "-train": CONVERSATIONAL_WRAPPERS},
    max_turns=6,
    data_path="mindgames/envs/TruthAndDeception/facts_tom.json",
    reveal_context_to_guesser=False,
)

# Liar's Dice (imperfect information + bluffing)
register_with_versions(
    id="LiarsDice-v0-small",
    entry_point="mindgames.envs.LiarsDice.env:LiarsDiceEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": DEFAULT_WRAPPERS},
    num_dice=3,
)
register_with_versions(
    id="LiarsDice-v0",
    entry_point="mindgames.envs.LiarsDice.env:LiarsDiceEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": DEFAULT_WRAPPERS},
    num_dice=5,
)
register_with_versions(
    id="LiarsDice-v0-large",
    entry_point="mindgames.envs.LiarsDice.env:LiarsDiceEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": DEFAULT_WRAPPERS},
    num_dice=12,
)

# Iterated Two-Thirds Average (2-player)
register_with_versions(
    id="IteratedTwoThirdsAverage-v0",
    entry_point="mindgames.envs.IteratedTwoThirdsAverage.env:IteratedTwoThirdsAverageEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": [GameMessagesObservationWrapper, ActionFormattingWrapper]},
    num_rounds=10,
    min_guess=0.0,
    max_guess=100.0,
)

# Iterated Two-Thirds Average (3-player)
register_with_versions(
    id="IteratedTwoThirdsAverage3P-v0",
    entry_point="mindgames.envs.IteratedTwoThirdsAverage.env_3p:IteratedTwoThirdsAverage3PEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": [GameMessagesObservationWrapper, ActionFormattingWrapper]},
    num_rounds=10,
    min_guess=0.0,
    max_guess=100.0,
)
