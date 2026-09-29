"""
Public types for the mission-risk layer (PHASE 12).

Three dataclasses form the **stable surface** the rest of the system
consumes:

* :class:`RiskDriver` — a single per-signal contribution to the
  risk score, with its raw value, contribution, and severity. This
  is the explainability hook: every score increment traces back to
  a named driver.
* :class:`RiskAssessment` — the per-tick output. Always carries a
  :class:`RiskStatus` so consumers can tell a confident decision
  apart from "I don't know yet" (``INSUFFICIENT_DATA``).
* :class:`MissionPhase` — coarse mission-context classification
  (PRE_FLIGHT / TAKEOFF / CLIMB / CRUISE / DESCENT / LANDING /
  POST_MISSION) used to nudge the risk score and the
  recommendations.

Coarse-grained :class:`RiskStatus` (the go/no-go answer) and
:class:`RiskLevel` (a numeric band for the dashboard colour map)
follow the "no fake outputs" pattern: missing evidence is
reported as such, never faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend._common import DEFAULT_TREND_WINDOW


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
class RiskStatus(str, Enum):
    """Coarse-grained mission go/no-go decision.

    * ``GO`` — nominal; continue the mission.
    * ``CAUTION`` — something is degrading; continue with awareness.
    * ``RETURN_TO_BASE`` — abort the mission segment, return safely.
    * ``ABORT`` — land immediately; engine failure is likely.
    * ``INSUFFICIENT_DATA`` — not enough evidence to decide; the
      risk score is not meaningful in this state. Per spec:
      "no fake outputs."
    """

    GO = "GO"
    CAUTION = "CAUTION"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    ABORT = "ABORT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskLevel(str, Enum):
    """Numeric band of the risk score. The dashboard maps this to
    a colour (LOW = green, MODERATE = yellow, HIGH = orange,
    SEVERE = red)."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class MissionPhase(str, Enum):
    """Coarse mission-context classification.

    The mission profile (PHASE 2) carries no explicit phase tag;
    this enum is derived heuristically from the profile + time
    (see :func:`~backend.risk.mission_phase.phase_for`).
    """

    PRE_FLIGHT = "PRE_FLIGHT"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    LANDING = "LANDING"
    POST_MISSION = "POST_MISSION"


class RiskTrend(str, Enum):
    """Trend of the risk score over recent ticks.

    * ``IMPROVING`` — risk_now < risk_ref - epsilon.
    * ``STABLE`` — |delta| <= epsilon.
    * ``DEGRADING`` — risk_now > risk_ref + epsilon.
    * ``INSUFFICIENT_DATA`` — not enough history.
    """

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DriverSeverity(str, Enum):
    """Severity label of a single :class:`RiskDriver`."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
# Risk-trend threshold: |Δrisk| < TREND_EPSILON_RISK is STABLE.
TREND_EPSILON_RISK: float = 0.03
# Number of recent risk scores kept for the trend.
# Re-exported from backend._common so the four trend-detector
# modules (RUL, risk, diagnostics, health) cannot drift apart
# silently. Override locally only if the risk-specific
# buffer needs a different size.
TREND_WINDOW: int = DEFAULT_TREND_WINDOW

# Status / level band edges (overridable via RiskConfig in YAML).
DEFAULT_THRESHOLD_CAUTION: float = 0.25
DEFAULT_THRESHOLD_RETURN_TO_BASE: float = 0.55
DEFAULT_THRESHOLD_ABORT: float = 0.80
# Below this overall confidence, the assessment is forced to
# INSUFFICIENT_DATA.
DEFAULT_MIN_CONFIDENCE: float = 0.20
# Hard caps on a single recommendation list.
MAX_RECOMMENDATIONS: int = 5


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------
def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def level_for(score: float) -> RiskLevel:
    """Map a risk score in [0, 1] to a :class:`RiskLevel` band."""
    s = _clip(score)
    if s >= DEFAULT_THRESHOLD_ABORT:
        return RiskLevel.SEVERE
    if s >= DEFAULT_THRESHOLD_RETURN_TO_BASE:
        return RiskLevel.HIGH
    if s >= DEFAULT_THRESHOLD_CAUTION:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def trend_for(
    current: Optional[float],
    reference: Optional[float],
    epsilon: float = TREND_EPSILON_RISK,
) -> RiskTrend:
    """Compare a current risk score to a reference risk score."""
    if current is None or reference is None:
        return RiskTrend.INSUFFICIENT_DATA
    delta = float(current) - float(reference)
    if delta < -float(epsilon):
        return RiskTrend.IMPROVING
    if delta > float(epsilon):
        return RiskTrend.DEGRADING
    return RiskTrend.STABLE


def status_for(
    *,
    health_label: Optional[str] = None,
    rul_status: Optional[str] = None,
    risk_score: float = 0.0,
    confidence: float = 0.0,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> RiskStatus:
    """Conservative go/no-go mapping.

    Mirrors the spec's "in doubt, escalate" rule. The first
    sufficient escalation wins.

    Order of escalation:

    1. **Insufficient evidence** — if any input is explicitly
       insufficient, or the overall confidence is below the
       floor, return :attr:`RiskStatus.INSUFFICIENT_DATA`.
    2. **Hard signal** — health CRITICAL or RUL CRITICAL ->
       :attr:`RiskStatus.ABORT`.
    3. **Score bands** — :attr:`RiskStatus.ABORT` > RETURN >
       CAUTION > GO.
    """
    if (
        health_label == "INSUFFICIENT_DATA"
        or rul_status == "RUL_UNCERTAIN"
        or float(confidence) < float(min_confidence)
    ):
        return RiskStatus.INSUFFICIENT_DATA
    if health_label == "CRITICAL" or rul_status == "RUL_CRITICAL":
        return RiskStatus.ABORT
    s = _clip(risk_score)
    if s >= DEFAULT_THRESHOLD_ABORT:
        return RiskStatus.ABORT
    if s >= DEFAULT_THRESHOLD_RETURN_TO_BASE:
        return RiskStatus.RETURN_TO_BASE
    if s >= DEFAULT_THRESHOLD_CAUTION:
        return RiskStatus.CAUTION
    return RiskStatus.GO


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RiskDriver:
    """A single per-signal contribution to the risk score.

    The ``contribution`` is the marginal amount the named signal
    added to the final score (>= 0). ``severity`` is a coarse
    label used by the dashboard for colouring.
    """

    signal: str                       # e.g. "health.overall_score"
    value: float                      # the input value
    contribution: float               # >= 0
    severity: DriverSeverity
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "value": float(self.value),
            "contribution": float(self.contribution),
            "severity": self.severity.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class RiskAssessment:
    """Per-tick output of the mission risk engine.

    The ``risk_score`` field is **always present** (no NaN), but
    ``status`` is the source of truth for whether the score is
    meaningful. When ``status`` is :attr:`RiskStatus.INSUFFICIENT_DATA`,
    the score is a placeholder reported for API stability;
    consumers MUST check ``status`` first.
    """

    time_s: float
    risk_score: float                                    # 0..1
    risk_level: RiskLevel
    status: RiskStatus
    confidence: float                                    # 0..1
    mission_phase: MissionPhase
    hours_to_destination: Optional[float] = None
    hours_to_critical: Optional[float] = None
    drivers: List[RiskDriver] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    contributing_faults: Dict[str, float] = field(default_factory=dict)
    trend: RiskTrend = RiskTrend.INSUFFICIENT_DATA
    notes: List[str] = field(default_factory=list)
    # Compact snapshot of upstream labels, for the audit log.
    inputs_meta: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------
    @property
    def is_go(self) -> bool:
        return self.status is RiskStatus.GO

    @property
    def is_caution(self) -> bool:
        return self.status is RiskStatus.CAUTION

    @property
    def is_return(self) -> bool:
        return self.status is RiskStatus.RETURN_TO_BASE

    @property
    def is_abort(self) -> bool:
        return self.status is RiskStatus.ABORT

    @property
    def is_insufficient(self) -> bool:
        return self.status is RiskStatus.INSUFFICIENT_DATA

    @property
    def is_terminal(self) -> bool:
        """True when the status is RETURN_TO_BASE, ABORT, or INSUFFICIENT_DATA.

        Used by the dashboard to decide whether to surface a
        prominent banner vs. a subtle chip.
        """
        return self.status in (
            RiskStatus.RETURN_TO_BASE,
            RiskStatus.ABORT,
            RiskStatus.INSUFFICIENT_DATA,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": float(self.time_s),
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "confidence": float(self.confidence),
            "mission_phase": self.mission_phase.value,
            "hours_to_destination": self.hours_to_destination,
            "hours_to_critical": self.hours_to_critical,
            "drivers": [d.to_dict() for d in self.drivers],
            "recommendations": list(self.recommendations),
            "contributing_faults": dict(self.contributing_faults),
            "trend": self.trend.value,
            "notes": list(self.notes),
            "inputs_meta": dict(self.inputs_meta),
        }


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_THRESHOLD_ABORT",
    "DEFAULT_THRESHOLD_CAUTION",
    "DEFAULT_THRESHOLD_RETURN_TO_BASE",
    "DriverSeverity",
    "MAX_RECOMMENDATIONS",
    "MissionPhase",
    "RiskAssessment",
    "RiskDriver",
    "RiskLevel",
    "RiskStatus",
    "RiskTrend",
    "TREND_EPSILON_RISK",
    "TREND_WINDOW",
    "level_for",
    "status_for",
    "trend_for",
]
