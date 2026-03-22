"""Register only the three supported environments for this branch."""

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
    ActionFormattingWrapper,
    GameMessagesAndCurrentBoardObservationWrapper,
    LLMObservationWrapper,
    NegotiationActionClipWrapper,
    NegotiationObservationWrapper,
)

DEFAULT_WRAPPERS = [LLMObservationWrapper, ActionFormattingWrapper]
BOARDGAME_WRAPPERS = [GameMessagesAndCurrentBoardObservationWrapper, ActionFormattingWrapper]
NEGOTIATION_WRAPPERS = [NegotiationObservationWrapper]
NEGOTIATION_TRAIN_WRAPPERS = [NegotiationObservationWrapper, NegotiationActionClipWrapper]

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
    info_tokens=3,
    fuse_tokens=2,
    max_turns=28,
)

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
