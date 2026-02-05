#!/usr/bin/env python3
"""
Minimal chat-style SFT (LoRA) with TRL.

Intended input: JSONL produced by `tools/data/rollouts_to_sft_jsonl.py`:
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _require_imports():
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Install:\n"
            "  pip install -U trl transformers accelerate peft datasets\n\n"
            f"Import error: {e}"
        ) from e


def _load_jsonl_preview(path: Path, n: int = 1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or local path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--data", required=True, help="Chat JSONL dataset path")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    _require_imports()
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"--data not found: {data_path}")

    preview = _load_jsonl_preview(data_path, n=1)
    if not preview or "messages" not in preview[0]:
        raise SystemExit(
            "Dataset must be JSONL with a top-level `messages` field (see tools/data/rollouts_to_sft_jsonl.py)."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def formatting_func(example: Dict[str, Any]) -> str:
        messages = example.get("messages") or []
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)

    ds = load_dataset("json", data_files=str(data_path), split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TRL has evolved over time (SFTConfig vs TrainingArguments, tokenizer vs processing_class).
    # Keep this script usable across a wider range of TRL versions.
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": ds,
        "formatting_func": formatting_func,
        "peft_config": peft_config,
    }
    try:
        from trl import SFTConfig  # type: ignore

        trainer_kwargs["args"] = SFTConfig(
            output_dir=str(out_dir),
            max_seq_length=int(args.max_seq_len),
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            seed=int(args.seed),
            bf16=bool(args.bf16),
            fp16=bool(args.fp16),
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            report_to=[],
        )
    except Exception:
        from transformers import TrainingArguments

        trainer_kwargs["args"] = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            seed=int(args.seed),
            bf16=bool(args.bf16),
            fp16=bool(args.fp16),
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            report_to=[],
        )

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
