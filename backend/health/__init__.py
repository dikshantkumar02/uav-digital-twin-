"""
Health package — per-subsystem health index (PHASE 10).

Public re-exports::

    from backend.health import (
        HealthIndexCalculator, HealthIndex, HealthLabel, HealthTrend,
        Subsystem, SubsystemHealth,
        score_thermal, score_lubrication, score_performance,
        score_mechanical, score_sensors,
        aggregate, normalize_weights,
    )
"""

from .aggregator import (
    MIN_OVERALL_CONFIDENCE,
    TREND_EPSILON,
    TREND_WINDOW,
    aggregate,
    new_trend_history,
    normalize_weights,
    update_trend_history,
)
from .index import FAULT_CONTRIBUTION_THRESHOLD, HealthIndexCalculator
from .subsystems import (
    score_lubrication,
    score_mechanical,
    score_performance,
    score_sensors,
    score_thermal,
)
from .types import (
    DEFAULT_WEIGHTS,
    HealthIndex,
    HealthLabel,
    HealthTrend,
    Subsystem,
    SubsystemHealth,
    label_for,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "FAULT_CONTRIBUTION_THRESHOLD",
    "HealthIndex",
    "HealthIndexCalculator",
    "HealthLabel",
    "HealthTrend",
    "MIN_OVERALL_CONFIDENCE",
    "Subsystem",
    "SubsystemHealth",
    "TREND_EPSILON",
    "TREND_WINDOW",
    "aggregate",
    "label_for",
    "new_trend_history",
    "normalize_weights",
    "score_lubrication",
    "score_mechanical",
    "score_performance",
    "score_sensors",
    "score_thermal",
    "update_trend_history",
]
