from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def _is_truthy(s: str) -> bool:
    return s.strip().lower() not in {"0", "false", "no", "off", ""}


def _iter_candidate_env_files() -> list[Path]:
    candidates: list[Path] = []

    # 1) Search from CWD upwards (common when running `cd mindgames`).
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        candidates.append(p / ".env")

    # 2) Also search from the mindgames project root upwards (covers running from repo root).
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    for p in [project_root, *project_root.parents]:
        candidates.append(p / ".env")

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def load_dotenv(*, override: bool = False) -> Optional[Path]:
    """
    Load environment variables from a `.env` file.

    - By default, does NOT override existing `os.environ` keys.
    - Controlled by `MINDGAMES_LOAD_DOTENV` (set to 0/false to disable).
    """
    global _LOADED
    if _LOADED:
        return None

    if not _is_truthy(os.getenv("MINDGAMES_LOAD_DOTENV", "1")):
        _LOADED = True
        return None

    env_path: Optional[Path] = None
    for candidate in _iter_candidate_env_files():
        if candidate.is_file():
            env_path = candidate
            break

    if env_path is None:
        _LOADED = True
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]

        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    _LOADED = True
    return env_path

