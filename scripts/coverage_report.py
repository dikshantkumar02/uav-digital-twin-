#!/usr/bin/env python3
"""
Operator-facing coverage reporter (PHASE 16).

Runs ``pytest --cov=backend --cov-report=term --cov-report=html:htmlcov``
against the full backend test suite and prints the per-module
breakdown to stdout. The HTML report is written to ``htmlcov/``
(already gitignored) for human inspection.

Usage::

    python scripts/coverage_report.py                # default testpaths
    python scripts/coverage_report.py --open         # open htmlcov/ after run
    python scripts/coverage_report.py --path backend/tests/test_phase10_health.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_cmd(args: argparse.Namespace) -> List[str]:
    cmd: List[str] = [
        sys.executable, "-m", "pytest",
        *args.paths,
        "--cov=backend",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-q", "--no-header",
    ]
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run pytest with coverage on backend/ and print the "
            "per-module breakdown."
        ),
    )
    p.add_argument(
        "--path", action="append", dest="paths", default=None,
        help=(
            "Test file or directory to run (repeatable). "
            "Default: backend/tests/"
        ),
    )
    p.add_argument(
        "--open", action="store_true", dest="open_html",
        help="Open htmlcov/index.html in the default browser after run.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the pytest command without running it.",
    )
    args = p.parse_args(argv)
    if not args.paths:
        args.paths = ["backend/tests/"]

    cmd = _build_cmd(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    print(">>", shlex.join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if args.open_html and rc == 0:
        import webbrowser
        index = REPO_ROOT / "htmlcov" / "index.html"
        if index.is_file():
            webbrowser.open(index.as_uri())
        else:
            print(f"htmlcov/ not produced; cannot open {index}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
