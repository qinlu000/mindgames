#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Hanabi GRPO dataset with MARSHAL-style env_config keys "
            "(turn-level dense reward + per-player reward normalization)."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset path.")
    parser.add_argument("--output", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--env-id", default="Hanabi-v0-train", help="Hanabi env id.")
    parser.add_argument("--num-players", type=int, default=2, help="Number of Hanabi players.")
    parser.add_argument(
        "--marshal-dense-reward",
        type=_parse_bool,
        default=True,
        help="Enable per-turn dense rewards in Hanabi env. (default: true)",
    )
    parser.add_argument(
        "--marshal-fuse-penalty",
        type=float,
        default=0.0,
        help="Penalty coefficient for fuse loss in dense rewards. (default: 0.0)",
    )
    parser.add_argument(
        "--marshal-invalid-penalty",
        type=float,
        default=0.0,
        help="Penalty added on invalid moves in dense rewards. (default: 0.0)",
    )
    parser.add_argument(
        "--marshal-agent-norm",
        type=_parse_bool,
        default=True,
        help="Enable per-player online reward normalization in rollout plugin. (default: true)",
    )
    parser.add_argument(
        "--marshal-agent-norm-method",
        choices=["mean", "mean_std"],
        default="mean_std",
        help="Per-player normalization method. (default: mean_std)",
    )
    parser.add_argument(
        "--marshal-agent-norm-warmup",
        type=int,
        default=8,
        help="Warmup samples per player before normalization is applied. (default: 8)",
    )
    parser.add_argument(
        "--marshal-agent-norm-clip",
        type=float,
        default=None,
        help="Optional clip value for normalized rewards.",
    )
    return parser


def _patch_env_config(env_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    patched = dict(env_config)
    patched.setdefault("name", "hanabi_env")
    patched["env_id"] = args.env_id
    patched["num_players"] = int(args.num_players)
    patched["marshal_dense_reward"] = bool(args.marshal_dense_reward)
    patched["marshal_fuse_penalty"] = float(args.marshal_fuse_penalty)
    patched["marshal_invalid_penalty"] = float(args.marshal_invalid_penalty)
    patched["marshal_agent_norm"] = bool(args.marshal_agent_norm)
    patched["marshal_agent_norm_method"] = args.marshal_agent_norm_method
    patched["marshal_agent_norm_warmup"] = int(args.marshal_agent_norm_warmup)
    if args.marshal_agent_norm_clip is not None:
        patched["marshal_agent_norm_clip"] = float(args.marshal_agent_norm_clip)
    return patched


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")

            obj["env_config"] = _patch_env_config(obj.get("env_config") or {}, args)
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} rows to {output_path}")


if __name__ == "__main__":
    main()
