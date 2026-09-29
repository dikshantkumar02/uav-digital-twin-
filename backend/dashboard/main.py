"""
``python -m backend.dashboard`` entrypoint (PHASE 13).

Boots the FastAPI app defined in :mod:`backend.dashboard.app` and
runs it under uvicorn. All knobs are CLI-overridable; the YAML is
the default. The server is runnable from a fresh checkout with no
arguments other than the optional ``--config <dir>``.

Usage::

    python -m backend.dashboard                            # default scenario
    python -m backend.dashboard --scenario healthy_60s     # explicit scenario
    python -m backend.dashboard --port 9000                 # custom port
    python -m backend.dashboard --tick-rate 20              # faster cadence
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import uvicorn

from backend.config import load_config, load_dashboard_config

from .app import create_app
from .scenarios import SCENARIO_REGISTRY, ScenarioSpec


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backend.dashboard",
        description="Real-time dashboard for the aero-piston digital twin.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the directory containing dashboard.yaml.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override dashboard.host (default from YAML).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override dashboard.port (default from YAML).",
    )
    parser.add_argument(
        "--tick-rate",
        type=float,
        default=None,
        help="Override dashboard.tick_rate_hz (default from YAML).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Initial scenario name from SCENARIO_REGISTRY.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn's auto-reload (dev only).",
    )
    args = parser.parse_args(argv)

    # Configure logging early.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    dc = load_dashboard_config(args.config)

    if args.host is not None:
        dc = dc.model_copy(update={"host": args.host})
    if args.port is not None:
        dc = dc.model_copy(update={"port": args.port})
    if args.tick_rate is not None:
        dc = dc.model_copy(update={"tick_rate_hz": float(args.tick_rate)})

    initial: Optional[ScenarioSpec] = None
    if args.scenario is not None:
        if args.scenario not in SCENARIO_REGISTRY:
            available = ", ".join(sorted(SCENARIO_REGISTRY))
            print(
                f"ERROR: unknown scenario '{args.scenario}'. "
                f"Available: {available}",
                flush=True,
            )
            return 2
        initial = SCENARIO_REGISTRY[args.scenario]

    app = create_app(cfg, dc, initial_scenario=initial)
    print(
        f"Starting dashboard on http://{dc.host}:{dc.port}  "
        f"(scenario={initial.name if initial else dc.default_scenario}, "
        f"tick_rate_hz={dc.tick_rate_hz})",
        flush=True,
    )
    uvicorn.run(
        app,
        host=dc.host,
        port=dc.port,
        log_level=dc.log_level.lower(),
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
