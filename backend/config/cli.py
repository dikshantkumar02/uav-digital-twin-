"""
Tiny CLI: print a summary of the loaded configuration.

Usage::

    python -m backend.config.cli            # uses default config/
    python -m backend.config.cli --config /path/to/config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import load_config


def _print_section(title: str, payload: object) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")
    print(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aero-dt")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the directory containing engine/environment/sensors/faults YAML.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Exit 0 if configuration loads, non-zero otherwise.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print("OK: configuration loaded and validated.")
        return 0

    print(f"Configuration loaded from: {cfg.source_dir}")
    _print_section("Engine", cfg.engine.model_dump())
    _print_section("Environment", cfg.environment.model_dump())
    _print_section("Sensors (channels)", {k: v.model_dump() for k, v in cfg.sensors.items()})
    _print_section("Faults", {k: v.model_dump() for k, v in cfg.faults.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
