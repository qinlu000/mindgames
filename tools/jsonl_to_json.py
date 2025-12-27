#!/usr/bin/env python3
"""
Convert JSONL logs (one JSON object per line) into a single JSON array.

Example:
  cd mindgames
  python tools/jsonl_to_json.py data/rollouts.jsonl --out data/rollouts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Input JSONL path")
    ap.add_argument("--out", required=True, type=str, help="Output JSON path (JSON array)")
    ap.add_argument("--indent", type=int, default=2, help="Pretty-print indent (0 = minified)")
    args = ap.parse_args()

    in_path = Path(args.path)
    out_path = Path(args.out)
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")

    indent = None if args.indent == 0 else int(args.indent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        out_f.write("[\n" if indent is not None else "[")
        first = True
        for obj in _iter_jsonl(in_path):
            if not first:
                out_f.write(",\n" if indent is not None else ",")
            first = False
            out_f.write(json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=True))
        out_f.write("\n]\n" if indent is not None else "]\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

