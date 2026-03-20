from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_COMPETITIVE_SYSTEM_PROMPT = (
    "You are playing a competitive text game. Read the rules carefully, "
    "make only valid moves, and maximize your final outcome."
)

DEFAULT_COOPERATIVE_SYSTEM_PROMPT = (
    "You are playing a cooperative hidden-information game. Reason carefully "
    "from public information, make only valid moves, and maximize the team's "
    "final score."
)

MINI_HANABI_SYSTEM_PROMPT = (
    "You are playing a two-player cooperative hidden-information game. "
    "Reason carefully from public information, make only valid moves, and "
    "maximize the team's final score."
)

COLONEL_BLOTTO_SYSTEM_PROMPT = (
    "You are playing a two-player simultaneous-allocation strategy game. "
    "Reason carefully about the public score state, submit one valid "
    "allocation, and maximize your match win probability."
)

NEGOTIATION_SYSTEM_PROMPT = (
    "You are playing a two-player bargaining game with public inventories "
    "and private values. Reason strategically, make only valid public "
    "messages, and maximize your own final value gain."
)


@dataclass(frozen=True)
class PromptProfile:
    template_name: str = "chat"
    system_prompt: Optional[str] = None
    response_format: str = "boxed"
    action_mode: str = "structured"


def apply_chat_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return observation


def apply_qwen3_template(observation: str, system_prompt: Optional[str] = None) -> str:
    system_message = system_prompt or DEFAULT_COMPETITIVE_SYSTEM_PROMPT
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(question: str, system_prompt: Optional[str] = None) -> str:
    system_message = system_prompt or "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_r1_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return (
        "A conversation between User and Assistant. The User presents the observation "
        "of a game, and the Assistant makes a valid action. "
        "The Assistant first thinks about the reasoning process in the mind and "
        "then provides the action. User: You must put your answer inside "
        "\\boxed{} and your final answer will be extracted automatically by the "
        "\\boxed{} tag.\n"
        f"Observation: {observation}\n"
        "Assistant:"
    )


def apply_r1_general_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the "
        "answer. User: You must put your answer inside \\boxed{} and your final "
        "answer will be extracted automatically by the \\boxed{} tag.\n"
        f"Question: {observation}\n"
        "Assistant:"
    )


def apply_llama_instruct_template(observation: str, system_prompt: Optional[str] = None) -> str:
    system_message = system_prompt or DEFAULT_COMPETITIVE_SYSTEM_PROMPT
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Current Observation: {observation}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def apply_llama_instruct_general_template(
    observation: str, system_prompt: Optional[str] = None
) -> str:
    system_message = system_prompt or "You are a helpful assistant."
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Question: {observation}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


TEMPLATE_FACTORY: dict[str, Callable[[str, Optional[str]], str]] = {
    "chat": apply_chat_template,
    "qwen3": apply_qwen3_template,
    "qwen3_general": apply_qwen3_general_template,
    "r1": apply_r1_template,
    "r1_general": apply_r1_general_template,
    "llama_instruct": apply_llama_instruct_template,
    "llama_instruct_general": apply_llama_instruct_general_template,
}


def render_prompt(observation: str, prompt_profile: Optional[PromptProfile]) -> str:
    if prompt_profile is None:
        return observation

    template_name = prompt_profile.template_name or "chat"
    if template_name not in TEMPLATE_FACTORY:
        raise ValueError(f"Unsupported prompt template: {template_name}")

    return TEMPLATE_FACTORY[template_name](
        observation,
        system_prompt=prompt_profile.system_prompt,
    )
