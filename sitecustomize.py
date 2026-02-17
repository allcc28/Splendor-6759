"""Auto-add local modules/ to PYTHONPATH for legacy imports."""

from __future__ import annotations

import sys
from pathlib import Path


MODULES_DIR = Path(__file__).resolve().parent / "modules"
if MODULES_DIR.exists():
    sys.path.insert(0, str(MODULES_DIR))
