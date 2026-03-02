#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "mindgames" / "apps" / "hanabi_human_ai_gui.py"
    sys.path.insert(0, str(target.parent.parent.parent))
    runpy.run_path(str(target), run_name="__main__")
