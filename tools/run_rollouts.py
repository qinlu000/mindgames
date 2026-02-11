#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "rollout" / "run_rollouts.py"
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
