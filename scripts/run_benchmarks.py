#!/usr/bin/env python3
"""
Operator-facing benchmark runner (PHASE 16).

Wraps ``pytest --benchmark-enable`` against the PHASE 16
benchmark suite so the operator can refresh the committed
``.benchmarks/`` baselines in one shot.

Usage::

    python scripts/run_benchmarks.py                # one round, fast
    python scripts/run_benchmarks.py --rounds 5     # 5 rounds, mean is stable
    python scripts/run_benchmarks.py --save         # write to .benchmarks/

The script exits with the pytest exit code; non-zero means
something failed or a regression was detected against the
committed baseline.
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
        "backend/tests/test_phase16_testing.py::TestBenchmarks",
        "-v",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,ops",
        "--benchmark-sort=mean",
    ]
    if args.rounds:
        cmd.append(f"--benchmark-min-rounds={args.rounds}")
    if args.save:
        cmd.append("--benchmark-autosave")
    else:
        # No committed baseline to compare against, so don't fail
        # on a "regression" we haven't recorded yet.
        cmd.append("--benchmark-disable-gc")
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run the PHASE 16 benchmark suite and (optionally) save "
            "the results to .benchmarks/."
        ),
    )
    p.add_argument(
        "--rounds", type=int, default=1,
        help="Minimum number of rounds per test (default: 1 — fast).",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Persist the run to .benchmarks/ (committable baseline).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the pytest command without running it.",
    )
    args = p.parse_args(argv)

    cmd = _build_cmd(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    print(">>", shlex.join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
