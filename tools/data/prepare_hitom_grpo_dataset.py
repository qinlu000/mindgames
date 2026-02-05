#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import pyarrow.ipc as ipc


SYSTEM_PROMPT = (
    "Answer the question using the story. Respond with the exact answer string or the option letter."
)


def _candidate_cache_roots() -> Iterable[Path]:
    roots = []
    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        roots.append(Path(env_cache))
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home) / "datasets")
    roots.append(Path("/home/cql/hf_cache/datasets"))
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        yield root


def _find_arrow_file(cache_root: Path) -> Optional[Path]:
    ds_root = cache_root / "Hi-ToM___hi-to_m_dataset"
    if not ds_root.exists():
        return None
    arrow_files = list(ds_root.rglob("*train.arrow"))
    if not arrow_files:
        return None
    return max(arrow_files, key=lambda p: p.stat().st_mtime)


def _load_table(arrow_path: Path):
    with arrow_path.open("rb") as f:
        reader = ipc.open_stream(f)
        return reader.read_all()


def _build_messages(story: str, question: str, choices: str):
    prompt = (
        "Story:\n"
        f"{story.strip()}\n\n"
        f"Question: {question.strip()}\n"
        f"Choices: {choices.strip()}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Hi-ToM GRPO JSONL with messages + answer.")
    parser.add_argument("--out", default="data/hitom_grpo.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit.")
    args = parser.parse_args()

    arrow_path = None
    for root in _candidate_cache_roots():
        arrow_path = _find_arrow_file(root)
        if arrow_path:
            break
    if not arrow_path:
        raise SystemExit("Hi-ToM cache not found. Run a HF download first or set HF_DATASETS_CACHE.")

    table = _load_table(arrow_path)
    names = table.schema.names

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = table.to_pydict()
    total = table.num_rows
    limit = total if args.max_rows is None else min(total, args.max_rows)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(limit):
            row = {name: rows[name][i] for name in names}
            story = row.get("story", "")
            question = row.get("question", "")
            choices = row.get("choices", "")
            answer = row.get("answer", "")

            record = {
                "messages": _build_messages(story, question, choices),
                "answer": answer,
                "choices": choices,
                "question": question,
                "story": story,
                "sample_id": row.get("sample_id"),
                "deception": row.get("deception"),
                "prompting_type": row.get("prompting_type"),
                "question_order": row.get("question_order"),
                "story_length": row.get("story_length"),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {limit} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
