#!/usr/bin/env python3
"""
Operator-facing helper: regenerate the committed ``data/golden/``
artefacts from the current code.

This script is the canonical "rebaseline the golden set" tool. It
walks every scenario in ``config/hil.yaml`` and produces a fresh
``wire.bin`` + ``snapshots.jsonl`` + ``manifest.yaml`` for each
under ``data/golden/<scenario>/``.

The committed golden set anchors the HIL validation harness. When
the pipeline changes in a way that *intentionally* shifts the
per-tick outputs (e.g. a better digital-twin integrator, a tuned
fault classifier), the operator runs this script to refresh the
reference, reviews the diff, and commits the result.

Usage::

    python scripts/generate_golden.py
    python scripts/generate_golden.py --root /tmp/golden  # write elsewhere
    python scripts/generate_golden.py --only healthy_60s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Make the repo root importable when this script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.hil import HilRunner, HILScenarioRegistry  # noqa: E402
from backend.hil.runner import make_default_runner  # noqa: E402


def _build_runner(root: Path) -> HilRunner:
    default = make_default_runner()
    return HilRunner(
        default._cfg,  # type: ignore[attr-defined]
        golden_root=root,
        registry=default.registry,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Regenerate the committed data/golden/ HIL reference set.",
    )
    p.add_argument(
        "--root", type=Path, default=REPO_ROOT / "data" / "golden",
        help="Where to write the golden artefacts (default: data/golden).",
    )
    p.add_argument(
        "--only", nargs="*", default=None,
        help="Optional subset of scenarios to regenerate.",
    )
    p.add_argument(
        "--note", action="append", default=[], dest="notes",
        help="Free-form note to attach to each manifest. Repeatable.",
    )
    args = p.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    runner = _build_runner(args.root)
    targets = (
        args.only if args.only else [sc.name for sc in runner.registry]
    )
    rc = 0
    for name in targets:
        try:
            out = runner.record(name, notes=list(args.notes))
        except KeyError as exc:
            print(f"skip: {exc}", file=sys.stderr)
            rc = 1
            continue
        print(f"regenerated: {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
