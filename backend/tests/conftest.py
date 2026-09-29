"""Pytest configuration shared across backend tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `import backend.*` works without install.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
