#!/usr/bin/env python3
"""Replay split SFT log segments into a single W&B run."""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, Iterable, Tuple


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _iter_metric_dicts(path: str) -> Iterable[dict]:
    with open(path, "r", errors="ignore") as f:
        for line in f:
            # 1) JSON logging.jsonl style
            raw = line.strip()
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    d = json.loads(raw)
                except Exception:
                    d = None
                if isinstance(d, dict) and (
                    "global_step/max_steps" in d or ("global_step" in d and "step" in d)
                ):
                    yield d
                    continue

            # 2) python-dict style in terminal logs with carriage returns
            if "global_step/max_steps" not in line:
                continue
            for seg in line.split("\r"):
                if "{" not in seg or "global_step/max_steps" not in seg:
                    continue
                start = seg.find("{")
                if start < 0:
                    continue
                payload = seg[start:].strip()
                try:
                    d = ast.literal_eval(payload)
                except Exception:
                    continue
                if isinstance(d, dict):
                    yield d


def parse_logs(paths: Iterable[str]) -> Tuple[Dict[int, dict], int | None]:
    """Return merged points keyed by global step, and max_steps if known."""
    points: Dict[int, dict] = {}
    max_steps = None

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        for d in _iter_metric_dicts(path):
            gsm = d.get("global_step/max_steps")
            gs = None
            ms = None
            if gsm:
                try:
                    gs_s, ms_s = str(gsm).split("/")
                    gs = int(gs_s)
                    ms = int(ms_s)
                except Exception:
                    gs = None
            # fallback for json logging entries that have step/global_step only
            if gs is None:
                try:
                    gs = int(d.get("global_step", d.get("step")))
                except Exception:
                    continue
            max_steps = ms
            # last-write-wins for duplicate steps across resumed segments
            points[gs] = d

    return points, max_steps


def to_wandb_row(step: int, d: dict, max_steps: int | None) -> dict:
    row = {"train/global_step": step}
    if max_steps is not None:
        row["train/max_steps"] = max_steps

    mapped = [
        ("loss", "train/loss"),
        ("grad_norm", "train/grad_norm"),
        ("learning_rate", "train/learning_rate"),
        ("token_acc", "train/token_acc"),
        ("epoch", "train/epoch"),
        ("memory(GiB)", "train/memory_gib"),
        ("train_speed(iter/s)", "train/iter_per_s"),
    ]
    for src, dst in mapped:
        if src in d:
            fv = _safe_float(d[src])
            if fv is not None:
                row[dst] = fv

    pct = d.get("percentage")
    if isinstance(pct, str) and pct.endswith("%"):
        fv = _safe_float(pct[:-1])
        if fv is not None:
            row["train/percentage"] = fv

    return row


def main():
    ap = argparse.ArgumentParser(description="Replay SFT logs into one W&B run.")
    ap.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="SFT log files in chronological order (earlier segment first).",
    )
    ap.add_argument("--project", default=os.getenv("WANDB_PROJECT", "mindgames"))
    ap.add_argument("--entity", default=os.getenv("WANDB_ENTITY", ""))
    ap.add_argument("--run-name", default="sft-merged-replay")
    ap.add_argument(
        "--wandb-mode",
        default=os.getenv("WANDB_MODE", "offline"),
        choices=["offline", "online", "disabled"],
    )
    ap.add_argument("--dry-run", action="store_true", help="Only parse and print stats.")
    args = ap.parse_args()

    points, max_steps = parse_logs(args.logs)
    if not points:
        raise SystemExit("No metric points parsed from provided logs.")

    steps = sorted(points.keys())
    print(f"parsed_points={len(steps)} first_step={steps[0]} last_step={steps[-1]} max_steps={max_steps}")

    if args.dry_run:
        return

    os.environ["WANDB_MODE"] = args.wandb_mode
    import wandb

    run = wandb.init(
        project=args.project,
        entity=(args.entity or None),
        name=args.run_name,
        job_type="replay_merge",
        config={"source_logs": args.logs, "replay_points": len(steps)},
    )

    for step in steps:
        row = to_wandb_row(step, points[step], max_steps)
        wandb.log(row, step=step)

    run.summary["merged_points"] = len(steps)
    run.summary["first_step"] = steps[0]
    run.summary["last_step"] = steps[-1]
    if max_steps is not None:
        run.summary["target_max_steps"] = max_steps
    run.finish()


if __name__ == "__main__":
    main()
