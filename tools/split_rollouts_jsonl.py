#!/usr/bin/env python3
"""
Split a rollouts JSONL file (from tools/run_rollouts.py) into one JSON per episode.

Example:
  cd mindgames
  python tools/split_rollouts_jsonl.py experiments/hanabi_eval/runs/<run_id>/rollouts.jsonl --out-dir experiments/hanabi_eval/runs/<run_id>/episodes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Input rollouts.jsonl")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for per-episode JSON files")
    ap.add_argument("--indent", type=int, default=2, help="Pretty-print indent (0 = minified)")
    args = ap.parse_args()

    in_path = Path(args.path)
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else int(args.indent)

    steps_by_ep: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    end_by_ep: Dict[int, Dict[str, Any]] = {}
    meta_by_ep: Dict[int, Dict[str, Any]] = {}

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {line_no} in {in_path}: {e}") from e
            if not isinstance(obj, dict):
                continue
            ep = obj.get("episode_id", None)
            if not isinstance(ep, int):
                continue

            t = obj.get("type", None)
            if t == "step":
                steps_by_ep[ep].append(obj)
                meta_by_ep.setdefault(
                    ep,
                    {"env_id": obj.get("env_id", None), "seed": obj.get("seed", None)},
                )
            elif t == "episode_end":
                end_by_ep[ep] = obj
                meta_by_ep.setdefault(
                    ep,
                    {"env_id": obj.get("env_id", None), "seed": obj.get("seed", None)},
                )

    for ep, steps in sorted(steps_by_ep.items(), key=lambda kv: kv[0]):
        meta = meta_by_ep.get(ep, {})
        out_path = out_dir / f"episode_{ep:06d}.json"
        out_path.write_text(
            json.dumps(
                {
                    "env_id": meta.get("env_id", None),
                    "episode_id": ep,
                    "seed": meta.get("seed", None),
                    "steps": steps,
                    "episode_end": end_by_ep.get(ep, None),
                },
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

