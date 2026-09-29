"""
Risk package — mission risk engine (PHASE 12) + mission reliability
view (PHASE 23).

Public re-exports::

    from backend.risk import (
        # PHASE 12 — go/no-go mission risk
        MissionRiskCalculator, RiskAssessment, RiskStatus, RiskLevel,
        RiskTrend, RiskDriver, DriverSeverity, MissionPhase,
        DEFAULT_THRESHOLD_CAUTION, DEFAULT_THRESHOLD_RETURN_TO_BASE,
        DEFAULT_THRESHOLD_ABORT, DEFAULT_MIN_CONFIDENCE,
        TREND_EPSILON_RISK, TREND_WINDOW, MAX_RECOMMENDATIONS,
        phase_for, hours_to_destination, PHASE_MODIFIERS,
        level_for, status_for, trend_for, aggregate,
        # PHASE 23 — mission reliability view (advisory, read-only)
        ReliabilityBand, ReliabilityDriver, ReliabilityConfig,
        MissionReliabilityExplanation, MissionReliabilityInputs,
        MissionReliabilityAssessment, aggregate_reliability,
    )
"""

from .aggregate import aggregate
from .calculator import DEFAULT_DT_S, MissionRiskCalculator
from .mission_phase import PHASE_MODIFIERS, hours_to_destination, phase_for
from .reliability import (
    DEFAULT_BAND_THRESHOLD_HIGH,
    DEFAULT_BAND_THRESHOLD_MEDIUM,
    DEFAULT_WEIGHT_ENGINE_LOAD,
    DEFAULT_WEIGHT_ENV_SEVERITY,
    DEFAULT_WEIGHT_FAULT_PROBABILITY,
    DEFAULT_WEIGHT_PHASE_RISK,
    DEFAULT_WEIGHT_RUL_UNCERTAINTY,
    DEFAULT_WEIGHT_SENSOR_AMBIGUITY,
    ENGINE_LOAD_CRITICAL,
    ENV_SEVERITY_CRITICAL,
    FAULT_PROB_CRITICAL,
    MIN_CONFIDENCE_FOR_BAND,
    MissionReliabilityAssessment,
    MissionReliabilityExplanation,
    MissionReliabilityInputs,
    ReliabilityBand,
    ReliabilityConfig,
    ReliabilityDriver,
    aggregate_reliability,
)
from .types import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_THRESHOLD_ABORT,
    DEFAULT_THRESHOLD_CAUTION,
    DEFAULT_THRESHOLD_RETURN_TO_BASE,
    MAX_RECOMMENDATIONS,
    DriverSeverity,
    MissionPhase,
    RiskAssessment,
    RiskDriver,
    RiskLevel,
    RiskStatus,
    RiskTrend,
    TREND_EPSILON_RISK,
    TREND_WINDOW,
    level_for,
    status_for,
    trend_for,
)

__all__ = [
    "DEFAULT_BAND_THRESHOLD_HIGH",
    "DEFAULT_BAND_THRESHOLD_MEDIUM",
    "DEFAULT_DT_S",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_THRESHOLD_ABORT",
    "DEFAULT_THRESHOLD_CAUTION",
    "DEFAULT_THRESHOLD_RETURN_TO_BASE",
    "DEFAULT_WEIGHT_ENGINE_LOAD",
    "DEFAULT_WEIGHT_ENV_SEVERITY",
    "DEFAULT_WEIGHT_FAULT_PROBABILITY",
    "DEFAULT_WEIGHT_PHASE_RISK",
    "DEFAULT_WEIGHT_RUL_UNCERTAINTY",
    "DEFAULT_WEIGHT_SENSOR_AMBIGUITY",
    "DriverSeverity",
    "ENGINE_LOAD_CRITICAL",
    "ENV_SEVERITY_CRITICAL",
    "FAULT_PROB_CRITICAL",
    "MAX_RECOMMENDATIONS",
    "MIN_CONFIDENCE_FOR_BAND",
    "MissionPhase",
    "MissionReliabilityAssessment",
    "MissionReliabilityExplanation",
    "MissionReliabilityInputs",
    "MissionRiskCalculator",
    "PHASE_MODIFIERS",
    "ReliabilityBand",
    "ReliabilityConfig",
    "ReliabilityDriver",
    "RiskAssessment",
    "RiskDriver",
    "RiskLevel",
    "RiskStatus",
    "RiskTrend",
    "TREND_EPSILON_RISK",
    "TREND_WINDOW",
    "aggregate",
    "aggregate_reliability",
    "hours_to_destination",
    "level_for",
    "phase_for",
    "status_for",
    "trend_for",
]
