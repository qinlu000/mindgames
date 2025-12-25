#!/usr/bin/env python3
"""
Probe whether a model can solve TruthAndDeception fact items from *prior knowledge alone*.

This is a "knowledge-only baseline": show the two candidate facts and force the model to pick
which one is true without any gameplay conversation.

Outputs:
  - JSONL with per-item results (prompt, raw output, parsed choice, correctness)
  - Optional JSON summary with aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))


_ensure_pkg_importable()

import mindgames as mg  # noqa: E402


TAG_FACT1_RE = re.compile(r"\[Fact\s*1\]", re.IGNORECASE)
TAG_FACT2_RE = re.compile(r"\[Fact\s*2\]", re.IGNORECASE)
_FACT_LINE_RE = re.compile(r"^\s*\[?\s*fact\s*([12])\s*\]?\s*$", re.IGNORECASE)
_NUM_LINE_RE = re.compile(r"^\s*([12])\s*$")


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    model: Optional[str] = None


@dataclass
class _ExistingStats:
    seen_idxs: set[int]
    total: int = 0
    parsable: int = 0
    correct: int = 0
    infer_ms_sum: int = 0


def _parse_agent_spec(s: str) -> AgentSpec:
    s = s.strip()
    if ":" in s:
        kind, model = s.split(":", 1)
        kind = kind.strip().lower()
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid --agent spec (missing model): {s}")
        if kind == "vllm":
            kind = "openai"
        return AgentSpec(kind=kind, model=model)
    kind = s.lower()
    if kind == "vllm":
        kind = "openai"
    return AgentSpec(kind=kind, model=None)


def _openai_like_kwargs(gen_kwargs: Dict[str, Any], request_timeout_s: Optional[float]) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = dict(gen_kwargs.get("extra_body") or {})
    if gen_kwargs.get("chat_template_kwargs") is not None:
        extra_body["chat_template_kwargs"] = dict(gen_kwargs["chat_template_kwargs"])
    if gen_kwargs.get("top_k") is not None:
        extra_body["top_k"] = int(gen_kwargs["top_k"])
    if gen_kwargs.get("repetition_penalty") is not None:
        extra_body["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
    if gen_kwargs.get("gen_seed") is not None:
        extra_body["seed"] = int(gen_kwargs["gen_seed"])

    kwargs: Dict[str, Any] = {
        "temperature": gen_kwargs.get("temperature", 0.0),
        "top_p": gen_kwargs.get("top_p", 1.0),
        "max_tokens": gen_kwargs.get("max_tokens", 2048),
    }
    if gen_kwargs.get("presence_penalty") is not None:
        kwargs["presence_penalty"] = float(gen_kwargs["presence_penalty"])
    if gen_kwargs.get("frequency_penalty") is not None:
        kwargs["frequency_penalty"] = float(gen_kwargs["frequency_penalty"])
    if extra_body:
        kwargs["extra_body"] = extra_body
    if request_timeout_s is not None:
        kwargs["timeout"] = float(request_timeout_s)
    return kwargs


def _build_agent(
    spec: AgentSpec,
    system_prompt: str,
    gen_kwargs: Dict[str, Any],
    *,
    model_name: Optional[str],
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    request_timeout_s: Optional[float],
) -> mg.Agent:
    if spec.kind == "human":
        return mg.agents.HumanAgent()

    if spec.kind == "hf":
        if not spec.model and not model_name:
            raise ValueError("Missing model name (use --agent hf:<model> or pass --model)")
        agent = mg.agents.HFLocalAgent(
            model_name=(spec.model or model_name),  # type: ignore[arg-type]
            max_new_tokens=int(gen_kwargs.get("max_new_tokens", 256)),
        )
        agent.system_prompt = system_prompt
        return agent

    if spec.kind == "openai":
        if not spec.model and not model_name:
            raise ValueError("Missing model name (use --agent openai:<model> or pass --model)")

        return mg.agents.OpenAIAgent(
            model_name=(spec.model or model_name),  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **_openai_like_kwargs(gen_kwargs, request_timeout_s),
        )

    if spec.kind == "qwen":
        if not spec.model and not model_name:
            raise ValueError("Missing model name (use --agent qwen:<model> or pass --model)")

        return mg.agents.QwenAgent(
            model_name=(spec.model or model_name),  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **_openai_like_kwargs(gen_kwargs, request_timeout_s),
        )

    if spec.kind == "openrouter":
        if not spec.model and not model_name:
            raise ValueError("Missing model name (use --agent openrouter:<model> or pass --model)")

        return mg.agents.OpenRouterAgent(
            model_name=(spec.model or model_name),  # type: ignore[arg-type]
            system_prompt=system_prompt,
            **_openai_like_kwargs(gen_kwargs, request_timeout_s),
        )

    if spec.kind == "ollama":
        if not spec.model and not model_name:
            raise ValueError("Missing model name (use --agent ollama:<model> or pass --model)")
        return mg.agents.OllamaAgent(
            model_name=(spec.model or model_name),  # type: ignore[arg-type]
            system_prompt=system_prompt,
            temperature=gen_kwargs.get("temperature", 0.0),
            top_p=gen_kwargs.get("top_p", 1.0),
            max_tokens=gen_kwargs.get("max_tokens", 2048),
        )

    raise ValueError(f"Unknown agent kind: {spec.kind} (supported: human, hf, openai, openrouter, ollama)")


def _load_fact_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in: {path}")
    for i, item in enumerate(data[:10]):
        if not isinstance(item, dict) or "facts" not in item or "correct_fact" not in item:
            raise SystemExit(f"Unexpected schema at item {i} in {path} (need keys: facts, correct_fact)")
    return data


def _sample_indices(total_items: int, num_items: int, rng: random.Random) -> List[int]:
    idxs = list(range(total_items))
    rng.shuffle(idxs)
    return idxs[: min(num_items, total_items)]


def _load_existing_stats(
    path: Path,
    *,
    allowed_idxs: set[int],
    strict_allowed: bool,
    strict_prompt_style: Optional[str],
) -> _ExistingStats:
    stats = _ExistingStats(seen_idxs=set())
    if not path.exists():
        return stats

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise SystemExit(f"Invalid JSONL in {path}:{line_no}: {e}") from e
        if not isinstance(obj, dict):
            continue
        idx = obj.get("item_idx")
        if not isinstance(idx, int):
            continue

        if idx not in allowed_idxs:
            if strict_allowed:
                raise SystemExit(
                    f"{path} contains item_idx={idx} not in this run's sample; "
                    "use a new --out-jsonl or run without --resume."
                )
            continue

        if strict_prompt_style is not None:
            ps = obj.get("prompt_style")
            if isinstance(ps, str) and ps != strict_prompt_style:
                raise SystemExit(
                    f"{path} contains prompt_style={ps!r} but you requested {strict_prompt_style!r}; "
                    "use a new --out-jsonl."
                )

        if idx in stats.seen_idxs:
            # Duplicate entries make resuming ambiguous.
            raise SystemExit(f"{path} has duplicate item_idx={idx} (line {line_no}); use a new --out-jsonl.")
        stats.seen_idxs.add(idx)

        stats.total += 1
        if obj.get("parsable") is True:
            stats.parsable += 1
        if obj.get("correct") is True:
            stats.correct += 1
        infer_ms = obj.get("infer_ms")
        if isinstance(infer_ms, int):
            stats.infer_ms_sum += infer_ms

    return stats


def _build_prompt(f1: str, f2: str, prompt_style: str) -> str:
    if prompt_style == "tag":
        return (
            "Two statements are shown below. Exactly one is true.\n"
            "Choose the true statement and respond with ONLY '[Fact 1]' or '[Fact 2]'.\n\n"
            f"Fact 1: {f1}\n"
            f"Fact 2: {f2}\n"
        )
    if prompt_style == "number":
        return (
            "Two statements are shown below. Exactly one is true.\n"
            "Choose the true statement and respond with ONLY '1' or '2'.\n\n"
            f"1) {f1}\n"
            f"2) {f2}\n"
        )
    raise ValueError(f"Unknown --prompt-style: {prompt_style} (expected: tag, number)")


def _parse_choice(raw: str, prompt_style: str) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None

    # Avoid spurious matches when the model rambles. Prefer parsing the final
    # non-empty line for "1/2" fallbacks.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    tail = lines[-1] if lines else raw

    if prompt_style == "tag":
        if TAG_FACT1_RE.search(raw):
            return 1
        if TAG_FACT2_RE.search(raw):
            return 2
        # fall back to short, explicit final answers only
        m = _FACT_LINE_RE.match(tail)
        if m:
            return int(m.group(1))
        m = _NUM_LINE_RE.match(tail)
        return int(m.group(1)) if m else None

    if prompt_style == "number":
        m = _NUM_LINE_RE.match(tail)
        if m:
            return int(m.group(1))
        m = _FACT_LINE_RE.match(tail)
        if m:
            return int(m.group(1))
        if TAG_FACT1_RE.search(raw):
            return 1
        if TAG_FACT2_RE.search(raw):
            return 2
        return None

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    default_facts = (
        Path(__file__).resolve().parents[1]
        / "mindgames"
        / "envs"
        / "TruthAndDeception"
        / "facts.json"
    )
    ap.add_argument("--facts", type=Path, default=default_facts, help="Path to TruthAndDeception facts.json")
    ap.add_argument(
        "--agent",
        required=True,
        help="Spec: hf[:<model>] | openai[:<model>] | qwen[:<model>] | vllm[:<model>] | openrouter[:<model>] | ollama[:<model>] | human",
    )
    ap.add_argument("--model", default=None, help="Model name (used when --agent omits ':<model>')")
    ap.add_argument("--openai-base-url", default=None, help="Override OPENAI_BASE_URL (for OpenAI/vLLM-compatible servers)")
    ap.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY (for OpenAI/vLLM-compatible servers)")
    ap.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds (passed to the client when supported)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-items", type=int, default=200, help="Number of fact items to probe (random sample)")
    ap.add_argument("--prompt-style", choices=["tag", "number"], default="tag")
    ap.add_argument(
        "--system-prompt",
        default="You are doing a strict multiple-choice knowledge check. Output ONLY the final choice in the required format.",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--extra-body",
        default=None,
        help="JSON dict merged into OpenAI `extra_body` (vLLM/OpenAI-compatible backends only).",
    )
    ap.add_argument(
        "--chat-template-kwargs",
        default=None,
        help='JSON dict merged into `extra_body.chat_template_kwargs` (vLLM-specific). Example: \'{"enable_thinking": false}\'',
    )
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Convenience flag for vLLM Qwen3-style thinking: sets chat_template_kwargs.enable_thinking=false (ignored if --chat-template-kwargs is set)",
    )
    ap.add_argument("--top-k", type=int, default=None, help="Top-k sampling (passed via extra_body for OpenAI-compatible backends when supported)")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (passed via extra_body for OpenAI-compatible backends when supported)")
    ap.add_argument("--presence-penalty", type=float, default=None, help="OpenAI-compatible presence_penalty")
    ap.add_argument("--frequency-penalty", type=float, default=None, help="OpenAI-compatible frequency_penalty")
    ap.add_argument("--gen-seed", type=int, default=None, help="Generation seed (passed via extra_body as seed when supported; independent of --seed sampling)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument(
        "--output-minimal",
        action="store_true",
        help="Write a compact JSONL containing only item_idx + reasoning + content.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially-written --out-jsonl by skipping item_idx already present (requires matching prompt-style).",
    )
    args = ap.parse_args()

    items = _load_fact_items(args.facts)
    rng = random.Random(args.seed)
    sample_idxs = _sample_indices(len(items), args.num_items, rng)
    allowed_idxs = set(sample_idxs)
    spec = _parse_agent_spec(args.agent)

    if args.chat_template_kwargs is not None:
        try:
            chat_template_kwargs = json.loads(args.chat_template_kwargs)
        except Exception as e:
            raise SystemExit(f"Invalid --chat-template-kwargs JSON: {e}") from e
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit("--chat-template-kwargs must be a JSON object (dict)")
    elif args.disable_thinking:
        chat_template_kwargs = {"enable_thinking": False}
    else:
        chat_template_kwargs = None

    if args.extra_body is not None:
        try:
            extra_body = json.loads(args.extra_body)
        except Exception as e:
            raise SystemExit(f"Invalid --extra-body JSON: {e}") from e
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object (dict)")
    else:
        extra_body = None

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": extra_body,
        "chat_template_kwargs": chat_template_kwargs,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "gen_seed": args.gen_seed,
        "max_tokens": args.max_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    agent = _build_agent(
        spec,
        args.system_prompt,
        gen_kwargs,
        model_name=args.model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        request_timeout_s=args.timeout,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_stats(
        args.out_jsonl,
        allowed_idxs=allowed_idxs,
        strict_allowed=bool(args.resume),
        strict_prompt_style=args.prompt_style if args.resume else None,
    )

    total = existing.total
    parsable = existing.parsable
    correct = existing.correct
    infer_ms_sum = existing.infer_ms_sum
    start_s = time.time()

    file_mode = "a" if args.resume and args.out_jsonl.exists() else "w"
    with args.out_jsonl.open(file_mode, encoding="utf-8") as f:
        for idx in sample_idxs:
            if args.resume and idx in existing.seen_idxs:
                continue
            item = items[idx]
            fact1 = str(item["facts"]["fact1"])
            fact2 = str(item["facts"]["fact2"])
            correct_key = str(item["correct_fact"])
            if correct_key not in {"fact1", "fact2"}:
                raise SystemExit(f"Unexpected correct_fact at item {idx}: {correct_key}")

            # Shuffle display order to avoid position bias.
            options = [("fact1", fact1), ("fact2", fact2)]
            rng.shuffle(options)
            shown_f1 = options[0][1]
            shown_f2 = options[1][1]
            correct_choice = 1 if options[0][0] == correct_key else 2

            prompt = _build_prompt(shown_f1, shown_f2, args.prompt_style)
            t0 = time.time()
            raw = agent(prompt)
            infer_ms = int((time.time() - t0) * 1000)
            raw_content = None
            raw_reasoning = None
            get_last = getattr(agent, "get_last_content_reasoning", None)
            if callable(get_last):
                c, r = get_last()
                raw_content = c
                raw_reasoning = r
            if raw_content is None:
                last_message = getattr(agent, "last_message", None)
                if isinstance(last_message, dict):
                    raw_content = last_message.get("content")
                    raw_reasoning = last_message.get("reasoning") or last_message.get("reasoning_content")

            # Prefer parsing from assistant content (excludes reasoning when a reasoning parser is enabled).
            choice = _parse_choice(str(raw_content) if raw_content is not None else raw, args.prompt_style)

            total += 1
            is_parsable = choice in {1, 2}
            is_correct = bool(is_parsable and choice == correct_choice)
            parsable += 1 if is_parsable else 0
            correct += 1 if is_correct else 0
            infer_ms_sum += infer_ms

            if args.output_minimal:
                _jsonl_write(
                    f,
                    {
                        "item_idx": idx,
                        "reasoning": raw_reasoning,
                        "content": raw_content if raw_content is not None else raw,
                    },
                )
            else:
                _jsonl_write(
                    f,
                    {
                        "type": "fact_probe",
                        "item_idx": idx,
                        "prompt_style": args.prompt_style,
                        "shown": {"fact1": shown_f1, "fact2": shown_f2},
                        "correct_choice": correct_choice,
                        "raw_output": raw,
                        "raw_response": getattr(agent, "last_response", None),
                        "raw_message": getattr(agent, "last_message", None),
                        "raw_content": raw_content,
                        "raw_reasoning": raw_reasoning,
                        "usage": getattr(agent, "last_usage", None),
                        "parsed_choice": choice,
                        "parsable": is_parsable,
                        "correct": is_correct,
                        "infer_ms": infer_ms,
                    },
                )

    chance = 0.5  # TruthAndDeception is 2-way multiple-choice.
    acc = (correct / total) if total else 0.0
    parse_rate = (parsable / total) if total else 0.0
    leakage_score = (acc - chance) / (1.0 - chance) if total else 0.0
    summary: Dict[str, Any] = {
        "facts_path": str(args.facts),
        "agent": args.agent,
        "seed": args.seed,
        "openai_base_url": args.openai_base_url or os.getenv("OPENAI_BASE_URL"),
        "gen": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "chat_template_kwargs": chat_template_kwargs,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "gen_seed": args.gen_seed,
            "max_tokens": args.max_tokens,
        },
        "num_items": len(sample_idxs),
        "processed_items": total,
        "resume": bool(args.resume),
        "prompt_style": args.prompt_style,
        "metrics": {
            "accuracy": acc,
            "chance": chance,
            "leakage_score": leakage_score,
            "parse_rate": parse_rate,
            "avg_infer_ms": (infer_ms_sum / total) if total else None,
        },
        "elapsed_s": time.time() - start_s,
    }

    if args.out_summary is not None:
        args.out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
