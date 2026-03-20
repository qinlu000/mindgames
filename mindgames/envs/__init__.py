"""Register selected environments (mirrors spiral-rl/spiral layout style)."""

from mindgames.envs.registration import register_with_versions
from mindgames.prompting import (
    COLONEL_BLOTTO_SYSTEM_PROMPT,
    MINI_HANABI_SYSTEM_PROMPT,
    NEGOTIATION_SYSTEM_PROMPT,
    PromptProfile,
)
from mindgames.prompting.action_parsers import (
    colonel_blotto_parse_available_actions,
    mini_hanabi_parse_available_actions,
    negotiation_parse_available_actions,
)
from mindgames.wrappers import (
    LLMObservationWrapper,
    ActionFormattingWrapper,
    GameMessagesAndCurrentBoardObservationWrapper,
    GameMessagesObservationWrapper,
    ClipCharactersActionWrapper,
    NegotiationObservationWrapper,
    NegotiationActionClipWrapper,
)

DEFAULT_WRAPPERS = [LLMObservationWrapper, ActionFormattingWrapper]
BOARDGAME_WRAPPERS = [GameMessagesAndCurrentBoardObservationWrapper, ActionFormattingWrapper]
CONVERSATIONAL_WRAPPERS = [LLMObservationWrapper, ClipCharactersActionWrapper]
NEGOTIATION_WRAPPERS = [NegotiationObservationWrapper]
NEGOTIATION_TRAIN_WRAPPERS = [NegotiationObservationWrapper, NegotiationActionClipWrapper]

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

# Mini Hanabi (short-context cooperative inference)
register_with_versions(
    id="MiniHanabi-v0",
    entry_point="mindgames.envs.MiniHanabi.env:MiniHanabiEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
    prompt_profile=PromptProfile(
        template_name="qwen3",
        system_prompt=MINI_HANABI_SYSTEM_PROMPT,
        response_format="boxed",
    ),
    action_parser=mini_hanabi_parse_available_actions,
    reward_mode="team_score",
    obs_mode="board_state",
    info_tokens=2,
    fuse_tokens=2,
    max_turns=12,
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

# Negotiation (2-player private-value bargaining)
register_with_versions(
    id="Negotiation-v0",
    entry_point="mindgames.envs.Negotiation.env:NegotiationEnv",
    wrappers={"default": NEGOTIATION_WRAPPERS, "-train": NEGOTIATION_TRAIN_WRAPPERS},
    prompt_profile=PromptProfile(
        template_name="qwen3",
        system_prompt=NEGOTIATION_SYSTEM_PROMPT,
        response_format="boxed",
        action_mode="chat",
    ),
    action_parser=negotiation_parse_available_actions,
    reward_mode="value_gain",
    obs_mode="public_private_chat",
    max_turns=20,
)
register_with_versions(
    id="Negotiation-v0-short",
    entry_point="mindgames.envs.Negotiation.env:NegotiationEnv",
    wrappers={"default": NEGOTIATION_WRAPPERS, "-train": NEGOTIATION_TRAIN_WRAPPERS},
    prompt_profile=PromptProfile(
        template_name="qwen3",
        system_prompt=NEGOTIATION_SYSTEM_PROMPT,
        response_format="boxed",
        action_mode="chat",
    ),
    action_parser=negotiation_parse_available_actions,
    reward_mode="value_gain",
    obs_mode="public_private_chat",
    max_turns=10,
)
register_with_versions(
    id="Negotiation-v0-long",
    entry_point="mindgames.envs.Negotiation.env:NegotiationEnv",
    wrappers={"default": NEGOTIATION_WRAPPERS, "-train": NEGOTIATION_TRAIN_WRAPPERS},
    prompt_profile=PromptProfile(
        template_name="qwen3",
        system_prompt=NEGOTIATION_SYSTEM_PROMPT,
        response_format="boxed",
        action_mode="chat",
    ),
    action_parser=negotiation_parse_available_actions,
    reward_mode="value_gain",
    obs_mode="public_private_chat",
    max_turns=50,
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

# Codenames (2v2 word deduction)
register_with_versions(
    id="Codenames-v0",
    entry_point="mindgames.envs.Codenames.env:CodenamesEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
    hardcore=False,
)
register_with_versions(
    id="Codenames-v0-hardcore",
    entry_point="mindgames.envs.Codenames.env:CodenamesEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
    hardcore=True,
)

# Colonel Blotto (2-player simultaneous allocation)
register_with_versions(
    id="ColonelBlotto-v0",
    entry_point="mindgames.envs.ColonelBlotto.env:ColonelBlottoEnv",
    wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
    prompt_profile=PromptProfile(
        template_name="qwen3",
        system_prompt=COLONEL_BLOTTO_SYSTEM_PROMPT,
        response_format="boxed",
        action_mode="structured",
    ),
    action_parser=colonel_blotto_parse_available_actions,
    reward_mode="zero_sum_terminal",
    obs_mode="board_state",
    num_fields=3,
    num_total_units=20,
    num_rounds=10,
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
