"""
Diagnostics package — anomaly detection (PHASE 8) + unified
diagnostic state (PHASE 22).

Public re-exports::

    from backend.diagnostics import (
        AnomalyDetector, AnomalyAssessment, ChannelAnomaly,
        AnomalyLabel, AnomalyThresholds, EwmaSmoother,
        DiagnosticState, DiagnosticAssembler,
    )
"""

from .detector import AnomalyDetector
from .diagnostic_assembler import DiagnosticAssembler
from .diagnostic_state import (
    TREND_EPSILON,
    TREND_WINDOW,
    DiagnosticState,
    SensorQuality,
    TrendDirection,
)
from .residual_score import EwmaSmoother
from .sensor_health import DEFAULT_SENSOR_HEALTH_TABLE
from .types import (
    AnomalyAssessment,
    AnomalyLabel,
    AnomalyThresholds,
    ChannelAnomaly,
)

__all__ = [
    "AnomalyAssessment",
    "AnomalyDetector",
    "AnomalyLabel",
    "AnomalyThresholds",
    "ChannelAnomaly",
    "DEFAULT_SENSOR_HEALTH_TABLE",
    "DiagnosticAssembler",
    "DiagnosticState",
    "EwmaSmoother",
    "SensorQuality",
    "TREND_EPSILON",
    "TREND_WINDOW",
    "TrendDirection",
]
