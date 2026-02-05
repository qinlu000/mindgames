#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from swift.plugin.orm import ORM, orms


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CHOICE_ITEM_RE = re.compile(r"([A-O])\s*\.\s*([^,]+)")
_LETTER_RE = re.compile(r"(?:^|\b)(?:option|answer|choice)?\s*([A-O])\b", re.IGNORECASE)


def _normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_choices(choices: Optional[str]) -> Dict[str, str]:
    if not choices:
        return {}
    return {letter.upper(): choice.strip() for letter, choice in _CHOICE_ITEM_RE.findall(choices)}


def _extract_predicted_answer(completion: str, choices: Dict[str, str], answer: str) -> Optional[str]:
    match = _ANSWER_TAG_RE.search(completion)
    if match:
        return match.group(1).strip()

    completion_norm = _normalize(completion)
    answer_norm = _normalize(answer)
    if answer_norm and answer_norm in completion_norm:
        return answer

    if choices:
        letters = _LETTER_RE.findall(completion)
        if letters:
            letter = letters[-1].upper()
            if letter in choices:
                return choices[letter]

        choice_matches = [
            choice for choice in choices.values() if _normalize(choice) and _normalize(choice) in completion_norm
        ]
        if choice_matches:
            choice_matches.sort(key=lambda c: len(c), reverse=True)
            return choice_matches[0]

    return None


class HiToMAccuracy(ORM):
    """Exact-match accuracy for Hi-ToM multiple-choice answers."""

    def __call__(
        self,
        completions: Iterable[str],
        answer: Optional[List[str]] = None,
        solution: Optional[List[str]] = None,
        choices: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        if answer is None:
            answer = solution
        if answer is None:
            return [0.0 for _ in completions]

        rewards = []
        for i, completion in enumerate(completions):
            ans = answer[i] if i < len(answer) else None
            choice_str = choices[i] if choices and i < len(choices) else None
            choice_map = _parse_choices(choice_str)

            pred = _extract_predicted_answer(completion, choice_map, ans or "")
            reward = float(_normalize(pred) == _normalize(ans))
            rewards.append(reward)
        return rewards


orms["hitom_accuracy"] = HiToMAccuracy
