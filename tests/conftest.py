"""
Pytest configuration for tests/.
"""

import sys
from pathlib import Path

# Ensure repo root is at the front of sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
