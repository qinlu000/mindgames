from __future__ import annotations

import re
from typing import Iterable, Optional

from mindgames.prompting.templates import PromptProfile


_ACTION_LINE_RE = re.compile(
    r"^\s*(play|discard|reveal|hint|bid|call|accept|deny|offer)\b",
    flags=re.IGNORECASE,
)
_MARKER_ACTION_RE = re.compile(
    r"(?:final answer|final action|answer|action|move|result)\s*[:：-]?\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.IGNORECASE | re.DOTALL)


def _strip_response_artifacts(text: str) -> str:
    cleaned_text = text or ""
    replacements = {
        "<|im_start|>assistant": " ",
        "<|im_end|>": " ",
        "<|begin_of_text|>": " ",
        "<|start_header_id|>": " ",
        "<|end_header_id|>": " ",
        "<|eot_id|>": " ",
        "<|assistant|>": " ",
        "<|endoftext|>": " ",
        "<think>": " ",
        "</think>": "\n",
        "<answer>": " ",
        "</answer>": "\n",
    }
    for source, target in replacements.items():
        cleaned_text = cleaned_text.replace(source, target)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def _last_boxed_only_string(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def _remove_boxed(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    left = "\\boxed{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left) : -1]
    left = "\\fbox{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left) : -1]
    return None


def _remove_text_boxed(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    left = "\\text{"
    if text.startswith(left) and text.endswith("}"):
        return text[len(left) : -1]
    return text


def extract_boxed_answer(text: str) -> Optional[str]:
    answer = _last_boxed_only_string(text)
    answer = _remove_boxed(answer)
    answer = _remove_text_boxed(answer)
    if answer is None:
        return None
    return _strip_response_artifacts(answer)


def extract_raw_action(text: str, prompt_profile: Optional[PromptProfile] = None) -> str:
    cleaned_text = _strip_response_artifacts(text)
    if not cleaned_text:
        return ""

    template_name = prompt_profile.template_name if prompt_profile is not None else "chat"
    response_format = (
        prompt_profile.response_format if prompt_profile is not None else "raw"
    ).lower()

    if response_format == "boxed":
        boxed = extract_boxed_answer(cleaned_text)
        if boxed:
            return boxed

    if template_name == "r1":
        answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        if answer_match and answer_match.group(1).strip():
            return _strip_response_artifacts(answer_match.group(1).strip())

    think_match = re.search(r"</think>\s*(.*)", text or "", flags=re.IGNORECASE | re.DOTALL)
    if think_match and think_match.group(1).strip():
        return _strip_response_artifacts(think_match.group(1).strip())

    marker_match = list(_MARKER_ACTION_RE.finditer(cleaned_text))
    if marker_match:
        return _strip_response_artifacts(marker_match[-1].group(1))

    return cleaned_text


def _extract_json_action(text: str) -> Optional[str]:
    for source in (text or "",):
        match = _FENCED_JSON_RE.search(source)
        if not match:
            continue
        try:
            import json

            obj = json.loads(match.group(1))
        except Exception:
            continue
        action = obj.get("action")
        if isinstance(action, str) and action.strip():
            return action.strip()
    return None


def _iter_action_candidates(raw_action: str, full_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(candidate: Optional[str]) -> None:
        if not candidate:
            return
        normalized = _strip_response_artifacts(
            re.sub(r"\\boxed\{([^}]*)\}", r"[\1]", candidate)
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    add_candidate(_extract_json_action(full_text))

    for source in (full_text, raw_action):
        if not source:
            continue
        for match in _MARKER_ACTION_RE.findall(source):
            add_candidate(match)
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        if lines:
            for line in lines:
                if "[" in line and "]" in line:
                    add_candidate(line[line.index("[") :].strip())
            for match in re.findall(r"\[[^\[\]\n]+\]", source):
                add_candidate(match)
            add_candidate(lines[-1])
            for line in reversed(lines):
                if _ACTION_LINE_RE.match(line):
                    add_candidate(line)
                    break

    add_candidate(raw_action)
    add_candidate(full_text)

    return candidates


def normalize_action_text(action: str) -> str:
    if not isinstance(action, str):
        action = str(action)

    text = action.strip()
    if not text:
        return ""

    bracket_lines = [line.strip() for line in text.splitlines() if "[" in line and "]" in line]
    if bracket_lines:
        line = bracket_lines[-1]
        return line[line.index("[") :].strip()

    match = re.match(r"^\s*(play|discard)\s+([A-Za-z0-9]+)\s*$", text, flags=re.IGNORECASE)
    if match:
        verb = match.group(1).capitalize()
        arg = match.group(2).strip()
        if re.fullmatch(r"[A-Za-z]", arg):
            return f"[{verb} {arg.upper()}]"
        return f"[{verb}] {arg}"

    match = re.match(r"^\s*hint\s+(color|rank)\s+([A-Za-z0-9]+)\s*$", text, flags=re.IGNORECASE)
    if match:
        hint_type = match.group(1).capitalize()
        value = match.group(2).strip()
        if hint_type == "Color":
            value = value.capitalize()
        return f"[Hint {hint_type} {value}]"

    match = re.match(
        r"^\s*reveal\s+player\s+(\d+)(?:\s+card\s+([A-Za-z0-9]+))?\s+(color|rank)\s+([A-Za-z0-9]+)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        player_id = match.group(1)
        card_id = match.group(2)
        hint_type = match.group(3).lower()
        hint_value = match.group(4)
        if hint_type == "color":
            hint_value = hint_value.lower()
        if card_id is None:
            return f"[Reveal] player {player_id} {hint_type} {hint_value}"
        return f"[Reveal] player {player_id} card {card_id} {hint_type} {hint_value}"

    match = re.match(r"^\s*bid\s*[:\-]?\s*(\d+)\s*,\s*(\d+)\s*$", text, flags=re.IGNORECASE)
    if match:
        return f"[Bid: {match.group(1)}, {match.group(2)}]"

    match = re.match(r"^\s*(call|accept|deny)\s*$", text, flags=re.IGNORECASE)
    if match:
        return f"[{match.group(1).capitalize()}]"

    match = re.match(r"^\s*offer\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE)
    if match:
        return f"[Offer: {match.group(1).strip()}]"

    return f"[{text}]"


def normalize_chat_action_text(action: str) -> str:
    if not isinstance(action, str):
        action = str(action)

    text = action.strip()
    if not text:
        return ""

    bracket_lines = [line.strip() for line in text.splitlines() if "[" in line and "]" in line]
    if bracket_lines:
        line = bracket_lines[-1]
        return line[line.index("[") :].strip()

    match = re.match(r"^\s*(accept|deny|call)\s*$", text, flags=re.IGNORECASE)
    if match:
        return f"[{match.group(1).capitalize()}]"

    match = re.match(r"^\s*offer\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE)
    if match:
        return f"[Offer: {match.group(1).strip()}]"

    return text


def _canonical_action_key(action: str) -> str:
    text = _strip_response_artifacts(action).strip()
    if not text:
        return ""

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return re.sub(r"\s+", " ", inner).lower()

    bracket_match = re.match(r"^\[(.*?)\]\s*(.*)$", text)
    if bracket_match:
        head = re.sub(r"\s+", " ", bracket_match.group(1).strip()).lower()
        tail = re.sub(r"\s+", " ", bracket_match.group(2).strip()).lower()
        return f"{head} {tail}".strip()

    normalized = normalize_action_text(text)
    if normalized != text:
        return _canonical_action_key(normalized)

    return re.sub(r"\s+", " ", text).lower()


def _get_action_mode(prompt_profile: Optional[PromptProfile]) -> str:
    if prompt_profile is None:
        return "structured"
    return (prompt_profile.action_mode or "structured").lower()


def select_legal_action(candidates: Iterable[str], legal_actions: Iterable[str]) -> Optional[str]:
    legal_map: dict[str, str] = {}
    for legal_action in legal_actions:
        if not isinstance(legal_action, str):
            continue
        key = _canonical_action_key(legal_action)
        if key:
            legal_map.setdefault(key, legal_action)

    if not legal_map:
        return None

    for candidate in candidates:
        direct_key = _canonical_action_key(candidate)
        if direct_key in legal_map:
            return legal_map[direct_key]

        normalized_candidate = normalize_action_text(candidate)
        normalized_key = _canonical_action_key(normalized_candidate)
        if normalized_key in legal_map:
            return legal_map[normalized_key]

    return None


def normalize_model_action(
    raw_action: str,
    *,
    prompt_profile: Optional[PromptProfile] = None,
    legal_actions: Optional[Iterable[str]] = None,
) -> str:
    raw_text = raw_action if isinstance(raw_action, str) else str(raw_action)
    extracted = extract_raw_action(raw_text, prompt_profile=prompt_profile)
    candidates = _iter_action_candidates(extracted, raw_text)
    action_mode = _get_action_mode(prompt_profile)

    if legal_actions is not None:
        legal_match = select_legal_action(candidates, legal_actions)
        if legal_match:
            return legal_match

    if candidates:
        if action_mode == "chat":
            return normalize_chat_action_text(candidates[0])
        return normalize_action_text(candidates[0])
    if action_mode == "chat":
        return normalize_chat_action_text(extracted or raw_text)
    return normalize_action_text(extracted or raw_text)
