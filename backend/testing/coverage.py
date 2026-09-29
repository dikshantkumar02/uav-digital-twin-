"""
Coverage baseline helpers (PHASE 16).

Wraps ``pytest --cov=backend`` so the operator-facing
``scripts/coverage_report.py`` and the ``TestCoverage`` test
suite can both ask for a per-module breakdown without
duplicating CLI plumbing.

This module is the thin Python side; the heavy lifting is
``pytest-cov``. The helpers here return a structured dict so a
test can assert "every critical module is above 70 %" without
parsing the terminal output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def measure_coverage(
    *,
    testpaths: Optional[List[str]] = None,
    root: Optional[Path] = None,
) -> Dict[str, object]:
    """Run ``pytest --cov=backend --cov-report=json:...`` and
    return the parsed JSON.

    Parameters
    ----------
    testpaths:
        Optional list of test directories (default: ``backend/tests``).
    root:
        Optional repo root (default: this file's parents[1]).

    Returns
    -------
    A dict with the keys ``"total_percent"`` and ``"per_module"``
    (mapping dotted module path → coverage percent).
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    if testpaths is None:
        testpaths = ["backend/tests"]
    out_path = root / ".coverage.json"
    cmd = [
        sys.executable, "-m", "pytest",
        *testpaths,
        "--cov=backend",
        f"--cov-report=json:{out_path}",
        "-q", "--no-header", "-x",
    ]
    try:
        subprocess.run(cmd, cwd=root, check=False, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("pytest not found; install dev deps") from exc
    if not out_path.is_file():
        raise RuntimeError("pytest-cov did not produce a JSON report")
    try:
        data = json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)
    total = float(data.get("totals", {}).get("percent_covered", 0.0))
    per_module: Dict[str, float] = {}
    files = data.get("files", {}) or {}
    for path, info in files.items():
        # path looks like "backend/health/index.py"
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != "backend":
            continue
        # Collapse "backend/health/index.py" → "backend/health".
        module = "/".join(parts[:2])
        per_module[module] = max(
            per_module.get(module, 0.0),
            float(info.get("summary", {}).get("percent_covered", 0.0)),
        )
    return {
        "total_percent": total,
        "per_module": per_module,
    }


def module_coverage(
    measure: Dict[str, object],
    module: str,
) -> float:
    """Return the coverage percent for ``module``
    (e.g. ``"backend/health"``)."""
    return float(measure.get("per_module", {}).get(module, 0.0))


__all__ = [
    "measure_coverage",
    "module_coverage",
]
