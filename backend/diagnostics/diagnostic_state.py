"""
PHASE 22 — Unified :class:`DiagnosticState`.

The :class:`DiagnosticState` is the **single common state
representation** consumed by the dashboard, the API, and the
telemetry replay. Every per-tick consumer of the digital twin +
AI chain reads this object; nobody reads the individual
PHASE 8/9/10/11/12 outputs directly anymore (the
:class:`~backend.dashboard.snapshot.DashboardSnapshot` and
:class:`DiagnosticState` share the same backing components but
present them through two different wire formats for backwards
compatibility).

The 12 user-facing fields the diagnostic state exposes::

    1. observed_state           — flat per-channel dict of sensor
                                  readings from this tick
    2. expected_state           — flat per-channel dict of
                                  digital-twin predicted values
    3. residual                 — per-channel observed − predicted
                                  with z-score and confidence
    4. residual_trend           — short-window slope of the
                                  per-channel mean |z|, signed
                                  ("WORSENING" / "STABLE" /
                                  "IMPROVING" / "INSUFFICIENT_DATA")
    5. sensor_quality           — per-channel health view
                                  (mode + 0..1 score)
    6. environment_context      — compact mission + env snapshot
    7. ai_anomaly_score         — single 0..1 number
    8. fault_probabilities      — class → probability mapping
                                  (grouped by category: engine /
                                  sensor / environment)
    9. health_index             — overall + per-subsystem
   10. rul                      — estimate + lower / upper /
                                  confidence / status / trend
   11. uncertainty              — single 0..1 number, the
                                  confidence gap across the
                                  upstream components
   12. mission_risk             — score + level + status + phase

The wire format is the user-facing JSON shown in the spec::

    {
      "engine_state": "DEGRADED",
      "health_index": 0.78,
      "anomaly_score": 0.84,
      "engine_fault_probability": 0.82,
      "sensor_fault_probability": 0.06,
      "environment_probability": 0.12,
      "rul": {
        "estimate": ...,
        "lower": ...,
        "upper": ...,
        "confidence": ...
      },
      "evidence": [...],
      "data_quality": ...,
      "model_version": ...
    }

That is :meth:`DiagnosticState.to_dict` — a strict subset of
the rich per-tick payload, designed for the consumer that
wants the *answer* not the full chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from backend._common import DEFAULT_TREND_WINDOW
from backend.ml import CalibrationStatus

# Local type-only imports — these are referenced only in
# annotations (the dataclass field types and the method
# signatures). We avoid importing the live classes at module
# load time because they form a cycle: backend.diagnostics →
# backend.health → backend.diagnostics (AnomalyAssessment).
if TYPE_CHECKING:
    from backend.environment import EnvironmentState
    from backend.faults import FaultClass
    from backend.health import HealthIndex
    from backend.ml import FaultClassification
    from backend.risk import MissionReliabilityAssessment, RiskAssessment
    from backend.rul import RulEstimate
    from backend.sensors import SensorSample
    from backend.simulation import EngineState
    from backend.diagnostics import AnomalyAssessment


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
class TrendDirection(str, Enum):
    """Short-window direction of a quantity.

    * ``IMPROVING`` — moving toward healthier / lower risk.
    * ``STABLE`` — within the epsilon band.
    * ``WORSENING`` — moving away from healthy / toward higher risk.
    * ``INSUFFICIENT_DATA`` — not enough history to decide.
    """

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    WORSENING = "WORSENING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SensorQuality(str, Enum):
    """Coarse per-channel sensor health."""

    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    UNRELIABLE = "UNRELIABLE"
    INVALID = "INVALID"


# Trend thresholds. |delta| <= TREND_EPSILON → STABLE.
TREND_EPSILON: float = 0.02
# Number of recent residual-trend samples kept for slope.
# Re-exported from backend._common so the four trend-detector
# modules (RUL, risk, diagnostics, health) cannot drift apart
# silently. Override locally only if the residual-trend
# buffer needs a different size.
TREND_WINDOW: int = DEFAULT_TREND_WINDOW


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _trend_for(current: Optional[float], reference: Optional[float]) -> TrendDirection:
    if current is None or reference is None:
        return TrendDirection.INSUFFICIENT_DATA
    delta = float(current) - float(reference)
    if delta < -TREND_EPSILON:
        return TrendDirection.IMPROVING
    if delta > TREND_EPSILON:
        return TrendDirection.WORSENING
    return TrendDirection.STABLE


# ---------------------------------------------------------------------
# Engine state label
# ---------------------------------------------------------------------
def _engine_state_label(
    health: HealthIndex,
) -> str:
    """Map the coarse health label to a simple string for the dashboard."""
    return health.overall_label.value


# ---------------------------------------------------------------------
# DiagnosticState — the 12-field unified state
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosticState:
    """The unified per-tick diagnostic state.

    Every consumer (dashboard, API, replay) reads this object.
    No consumer reads the underlying per-tick components directly
    — that is the point of PHASE 22.
    """

    time_s: float
    # ---- the 12 user-facing fields ----
    observed_state: Dict[str, Any]               # 1
    expected_state: Dict[str, float]             # 2
    residual: Dict[str, Dict[str, Any]]         # 3
    residual_trend: Dict[str, Any]               # 4
    sensor_quality: Dict[str, Dict[str, Any]]   # 5
    environment_context: Dict[str, Any]          # 6
    ai_anomaly_score: float                     # 7
    fault_probabilities: Dict[str, float]        # 8 — group-keyed
    health_index: Dict[str, Any]                 # 9
    rul: Dict[str, Any]                          # 10
    uncertainty: float                           # 11
    mission_risk: Dict[str, Any]                 # 12
    # ---- companion / audit fields ----
    evidence: Tuple[str, ...] = ()
    data_quality: float = 1.0
    model_version: str = "phase22-diagnostic-1.0.0"
    engine_state: str = "UNKNOWN"               # mirrored label
    # ---- raw upstream payloads (debug / advanced consumers) ----
    raw_engine: Optional[EngineState] = field(default=None, repr=False)
    raw_environment: Optional[EnvironmentState] = field(default=None, repr=False)
    raw_anomaly: Optional[AnomalyAssessment] = field(default=None, repr=False)
    raw_classification: Optional[FaultClassification] = field(default=None, repr=False)
    raw_health: Optional[HealthIndex] = field(default=None, repr=False)
    raw_rul: Optional[RulEstimate] = field(default=None, repr=False)
    raw_risk: Optional[RiskAssessment] = field(default=None, repr=False)
    # PHASE 23 — mission reliability view (advisory, read-only).
    # Populated by the dashboard's per-tick reliability aggregator
    # so the unified state carries the new view alongside the
    # 12 user-facing fields. The 12 user-facing fields are
    # unchanged.
    reliability: Optional[MissionReliabilityAssessment] = field(
        default=None, repr=False,
    )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        """True if the classifier is calibrated and emitted a real
        probability vector (not the default ``MODEL_NOT_CALIBRATED``
        response)."""
        if self.raw_classification is None:
            return False
        return self.raw_classification.status is CalibrationStatus.CALIBRATED

    @property
    def is_insufficient(self) -> bool:
        """True when the risk assessment reports
        :attr:`RiskStatus.INSUFFICIENT_DATA` (no fake outputs)."""
        if self.raw_risk is None:
            return True
        return self.raw_risk.is_insufficient

    @property
    def is_go(self) -> bool:
        return self.raw_risk is not None and self.raw_risk.is_go

    @property
    def is_abort(self) -> bool:
        return self.raw_risk is not None and self.raw_risk.is_abort

    # ------------------------------------------------------------------
    # Wire format (the user-spec example)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """User-spec wire format.

        A strict, stable subset of the rich per-tick payload.
        Consumed by the dashboard, the API, and the replay.
        """
        return {
            # Coarse label that drives the dashboard colour map.
            "engine_state": self.engine_state,
            "health_index": float(self.health_index.get("overall_score", 0.0)),
            "anomaly_score": float(self.ai_anomaly_score),
            # Per-category fault probabilities.
            "engine_fault_probability": float(
                self.fault_probabilities.get("engine", 0.0)
            ),
            "sensor_fault_probability": float(
                self.fault_probabilities.get("sensor", 0.0)
            ),
            "environment_probability": float(
                self.fault_probabilities.get("environment", 0.0)
            ),
            "unknown_probability": float(
                self.fault_probabilities.get("unknown", 0.0)
            ),
            "healthy_probability": float(
                self.fault_probabilities.get("healthy", 0.0)
            ),
            # RUL block.
            "rul": dict(self.rul),
            # Audit / provenance.
            "evidence": list(self.evidence),
            "data_quality": float(self.data_quality),
            "model_version": self.model_version,
        }

    def to_rich_dict(self) -> Dict[str, Any]:
        """Full per-tick payload — the 12 user-facing fields
        expanded, plus audit / provenance.

        This is the format the API surfaces when the caller
        asks for ``?rich=1`` or for the replay dump.
        """
        return {
            "time_s": float(self.time_s),
            "observed_state": dict(self.observed_state),
            "expected_state": dict(self.expected_state),
            "residual": {k: dict(v) for k, v in self.residual.items()},
            "residual_trend": dict(self.residual_trend),
            "sensor_quality": {k: dict(v) for k, v in self.sensor_quality.items()},
            "environment_context": dict(self.environment_context),
            "ai_anomaly_score": float(self.ai_anomaly_score),
            "fault_probabilities": dict(self.fault_probabilities),
            "health_index": dict(self.health_index),
            "rul": dict(self.rul),
            "uncertainty": float(self.uncertainty),
            "mission_risk": dict(self.mission_risk),
            "evidence": list(self.evidence),
            "data_quality": float(self.data_quality),
            "model_version": self.model_version,
            "engine_state": self.engine_state,
            # PHASE 23 — advisory mission reliability view. None
            # when the dashboard has not yet produced an
            # assessment (e.g. tests that build the state
            # directly). Never contains control commands or a
            # success-probability number.
            "reliability": (
                self.reliability.to_dict()
                if self.reliability is not None else None
            ),
        }


__all__ = [
    "DiagnosticState",
    "SensorQuality",
    "TrendDirection",
    "TREND_EPSILON",
    "TREND_WINDOW",
]
