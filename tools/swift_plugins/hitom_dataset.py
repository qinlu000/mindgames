#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional

from swift.llm.dataset import DatasetMeta, RowPreprocessor, register_dataset


SYSTEM_PROMPT = (
    "Answer the question using the story. Respond with the exact answer string or the option letter."
)


class HiToMGRPOPreprocessor(RowPreprocessor):
    def __init__(self, system_prompt: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def preprocess(self, row: Dict[str, Any]):
        story = row.get("story", "")
        question = row.get("question", "")
        choices = row.get("choices", "")
        answer = row.get("answer")

        user_prompt = (
            "Story:\n"
            f"{story.strip()}\n\n"
            f"Question: {question.strip()}\n"
            f"Choices: {choices.strip()}\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        row["messages"] = messages
        if answer is not None:
            row["answer"] = answer
            row["solution"] = answer
        if choices is not None:
            row["choices"] = choices
        return row


register_dataset(
    DatasetMeta(
        hf_dataset_id="Hi-ToM/Hi-ToM_Dataset",
        preprocess_func=HiToMGRPOPreprocessor(),
        tags=["grpo", "hitom"],
    ),
    exist_ok=True,
)
