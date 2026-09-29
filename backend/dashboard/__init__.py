"""
Dashboard package (PHASE 13) — real-time dashboard server.

Public re-exports::

    from backend.dashboard import (
        # The FastAPI app
        create_app,
        # The pipeline orchestrator
        PipelineRunner, DEFAULT_DT_S, DEFAULT_HISTORY_SIZE,
        # The wire format
        DashboardSnapshot, WIRE_KEYS,
        # The scenario registry
        ScenarioSpec, SCENARIO_REGISTRY, DEFAULT_SCENARIO_NAME, get_scenario,
        # PHASE 19 — control-room dashboard additions
        Alert, derive_alerts, recommendation_for,
        # The CLI entrypoint
        main,
    )
"""

from .alerts import Alert, derive_alerts, recommendation_for
from .app import create_app
from .main import main
from .runner import DEFAULT_DT_S, DEFAULT_HISTORY_SIZE, PipelineRunner
from .scenarios import (
    DEFAULT_SCENARIO_NAME,
    SCENARIO_REGISTRY,
    ScenarioSpec,
    get_scenario,
)
from .snapshot import WIRE_KEYS, DashboardSnapshot

__all__ = [
    "Alert",
    "DEFAULT_DT_S",
    "DEFAULT_HISTORY_SIZE",
    "DEFAULT_SCENARIO_NAME",
    "DashboardSnapshot",
    "PipelineRunner",
    "SCENARIO_REGISTRY",
    "ScenarioSpec",
    "WIRE_KEYS",
    "create_app",
    "derive_alerts",
    "get_scenario",
    "main",
    "recommendation_for",
]
