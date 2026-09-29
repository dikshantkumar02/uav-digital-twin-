"""
CLI entry point for the HIL validation harness.

Usage::

    python -m backend.hil record --scenario engine_degradation_60s [--root PATH]
    python -m backend.hil validate --scenario engine_degradation_60s [--root PATH]
    python -m backend.hil validate-all [--root PATH]
    python -m backend.hil list

The default action is ``validate-all`` (matches the operator's
mental model: "run the HIL checks" = "validate every golden").
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .runner import HilRunner, ValidateResult, make_default_runner


# ---------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------
def _cmd_record(args: argparse.Namespace) -> int:
    runner = make_default_runner()
    if args.root:
        runner = HilRunner(
            runner._cfg,  # type: ignore[attr-defined]
            golden_root=_resolve_root(args.root),
            registry=runner.registry,
        )
    out = runner.record(args.scenario, notes=args.notes or [])
    print(f"recorded golden set: {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    runner = make_default_runner()
    if args.root:
        runner = HilRunner(
            runner._cfg,  # type: ignore[attr-defined]
            golden_root=_resolve_root(args.root),
            registry=runner.registry,
        )
    res = runner.validate(args.scenario)
    return _print_result(res)


def _cmd_validate_all(args: argparse.Namespace) -> int:
    runner = make_default_runner()
    if args.root:
        runner = HilRunner(
            runner._cfg,  # type: ignore[attr-defined]
            golden_root=_resolve_root(args.root),
            registry=runner.registry,
        )
    results = runner.validate_all()
    rc = 0
    for name, res in results.items():
        rc |= _print_result(res)
    return rc


def _cmd_list(args: argparse.Namespace) -> int:
    runner = make_default_runner()
    for sc in runner.registry:
        print(f"- {sc.name}  ({sc.spec.fault_class.value}, "
              f"severity={sc.spec.severity}, run_classifier={sc.spec.run_classifier})")
    return 0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_root(path: str):
    from pathlib import Path
    return Path(path)


def _print_result(res: ValidateResult) -> int:
    rep = res.report
    status = "PASS" if rep.passed else "FAIL"
    print(
        f"{status}: {res.scenario_name} "
        f"({rep.n_ticks_compared} ticks, "
        f"{rep.n_mismatches} mismatches, "
        f"{res.frames_decoded} frames decoded, "
        f"{res.bytes_replayed} bytes replayed)"
    )
    if not rep.passed:
        for m in rep.mismatches[:10]:
            print(
                f"  - tick={m.tick_index} path={m.field_path} "
                f"expected={m.expected!r} actual={m.actual!r} "
                f"delta={m.delta} tol={m.tolerance}"
            )
        if len(rep.mismatches) > 10:
            print(f"  ... and {len(rep.mismatches) - 10} more")
        return 1
    return 0


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.hil",
        description="HIL validation harness (PHASE 15).",
    )
    sub = p.add_subparsers(dest="action")
    # record
    pr = sub.add_parser("record", help="Record a fresh golden set for one scenario.")
    pr.add_argument("--scenario", required=True, help="Scenario name from SCENARIO_REGISTRY.")
    pr.add_argument("--root", default=None, help="Override golden_root (default: from hil.yaml).")
    pr.add_argument("--note", action="append", default=[], dest="notes",
                    help="Free-form note to attach to the manifest. Repeatable.")
    # validate
    pv = sub.add_parser("validate", help="Validate one scenario against its golden set.")
    pv.add_argument("--scenario", required=True, help="Scenario name.")
    pv.add_argument("--root", default=None, help="Override golden_root.")
    # validate-all
    pva = sub.add_parser("validate-all", help="Validate every covered scenario.")
    pva.add_argument("--root", default=None, help="Override golden_root.")
    # list
    sub.add_parser("list", help="List scenarios covered by the HIL harness.")
    return p


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "record":
        return _cmd_record(args)
    if args.action == "validate":
        return _cmd_validate(args)
    if args.action == "validate-all":
        return _cmd_validate_all(args)
    if args.action == "list":
        return _cmd_list(args)
    # Default: validate-all.
    args.root = None
    return _cmd_validate_all(args)


if __name__ == "__main__":
    sys.exit(main())
