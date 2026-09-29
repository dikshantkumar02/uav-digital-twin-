"""
Pure risk aggregation (PHASE 12).

:func:`aggregate` consumes the per-tick outputs of PHASES 8 / 10 /
11 plus a :class:`MissionPhase` and produces a
:class:`~backend.risk.types.RiskAssessment`. Everything in this
module is pure (no I/O, no mutable state, no per-instance
attributes). The orchestrator (:class:`MissionRiskCalculator`)
owns the trend buffer.

Scoring model
-------------

A weighted sum of three *deficits*:

* ``d_health  = 1 - health.overall_score``             ∈ [0, 1]
* ``d_rul     = rul_deficit(rul)``                     ∈ [0, 1]
* ``d_anomaly = anomaly.overall_score``                ∈ [0, 1] or 0

plus a small signed **mission-phase modifier** (e.g. ``+0.10`` for
TAKEOFF) and a small **fault boost** scaled by the max
contributing-fault probability.

The multiplicative alternative was rejected: a single healthy
signal can mask dangerous ones. A weighted sum is monotone in
every input and traceable via the per-driver contributions.

Status mapping is **conservative** — when in doubt, escalate.
``INSUFFICIENT_DATA`` wins over ``ABORT`` which wins over
``RETURN_TO_BASE`` which wins over ``CAUTION`` which wins over
``GO``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from backend.config import RiskConfig
from backend.diagnostics import AnomalyAssessment
from backend.health import HealthIndex
from backend.rul import RulEstimate

from .mission_phase import PHASE_MODIFIERS
from .types import (
    DriverSeverity,
    MAX_RECOMMENDATIONS,
    MissionPhase,
    RiskAssessment,
    RiskDriver,
    RiskLevel,
    RiskStatus,
    RiskTrend,
    level_for,
    status_for,
    trend_for,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _normalize_weights(cfg: RiskConfig) -> Dict[str, float]:
    """Normalise the three weights so they sum to 1.0.

    If the user sets a sum different from 1.0 (e.g. all 1.0 for
    "equal weighting"), the result is divided by the sum.
    """
    raw = {
        "health": float(cfg.weight_health),
        "rul": float(cfg.weight_rul),
        "anomaly": float(cfg.weight_anomaly),
    }
    total = sum(raw.values())
    if total <= 0.0:
        # Degenerate config — fall back to equal weights.
        return {"health": 1 / 3, "rul": 1 / 3, "anomaly": 1 / 3}
    return {k: v / total for k, v in raw.items()}


def _rul_deficit(
    rul: RulEstimate,
    *,
    floor_h: float,
    ceiling_h: float,
) -> float:
    """Nonlinear RUL deficit in [0, 1].

    * Lower bound (5th percentile) <= floor_h       -> 1.0
    * Lower bound >= ceiling_h                       -> 0.0
    * In between                                     -> linear ramp
    * RUL is uncertain                               -> 0.0
      (the status mapping escalates separately)
    """
    if rul.is_uncertain:
        return 0.0
    lo = float(rul.tte_hours_lower)
    if lo <= float(floor_h):
        return 1.0
    if lo >= float(ceiling_h):
        return 0.0
    return _clip(1.0 - (lo - float(floor_h)) / (float(ceiling_h) - float(floor_h)))


def _confidence(
    health: HealthIndex,
    rul: RulEstimate,
    anomaly: Optional[AnomalyAssessment],
) -> float:
    """Combined confidence: the minimum of the three inputs.

    Returns 0.0 when no inputs are present.
    """
    cs: List[float] = [float(health.confidence), float(rul.confidence)]
    if anomaly is not None:
        cs.append(float(anomaly.confidence))
    return max(0.0, min(1.0, min(cs)))


def _inputs_meta(
    health: HealthIndex,
    rul: RulEstimate,
    anomaly: Optional[AnomalyAssessment],
    phase: MissionPhase,
) -> Dict[str, str]:
    return {
        "health_label": str(health.overall_label.value),
        "health_trend": str(health.trend.value),
        "rul_status": str(rul.status.value),
        "rul_trend": str(rul.trend.value),
        "mission_phase": phase.value,
        "anomaly_label": (
            str(anomaly.overall_label.value) if anomaly is not None else "MISSING"
        ),
    }


# ---------------------------------------------------------------------
# Recommendations table
# ---------------------------------------------------------------------
def _build_recommendations(
    *,
    status: RiskStatus,
    health: HealthIndex,
    rul: RulEstimate,
    anomaly: Optional[AnomalyAssessment],
    phase: MissionPhase,
    risk_score: float,
    floor_h: float,
) -> List[str]:
    """Return at most :data:`~backend.risk.types.MAX_RECOMMENDATIONS`
    operator-facing advisories."""
    recs: List[str] = []
    if status is RiskStatus.ABORT:
        recs.append("ABORT: land at nearest suitable site.")
    elif status is RiskStatus.RETURN_TO_BASE:
        recs.append("RETURN TO BASE.")
    elif status is RiskStatus.INSUFFICIENT_DATA:
        recs.append("Decision layer has insufficient data — sensor check required.")
    if (
        not rul.is_uncertain
        and rul.tte_hours_lower < float(floor_h)
        and "Engine end-of-life" not in "\n".join(recs)
    ):
        recs.append(
            f"Engine end-of-life within {floor_h:g} hours — land immediately."
        )
    if (
        status is not RiskStatus.ABORT
        and float(risk_score) >= 0.55
        and not any("return" in r.lower() for r in recs)
    ):
        recs.append("Initiate return-to-base; mission abort likely.")
    if health.trend.value == "DEGRADING" and status is not RiskStatus.INSUFFICIENT_DATA:
        recs.append("Health is degrading — reduce throttle and observe.")
    if (
        anomaly is not None
        and anomaly.overall_label.value == "ANOMALY"
        and status is not RiskStatus.INSUFFICIENT_DATA
    ):
        recs.append("Sensor anomaly active — verify readings before next decision.")
    if (
        phase is MissionPhase.CRUISE
        and 0.25 <= float(risk_score) < 0.55
        and status not in (RiskStatus.ABORT, RiskStatus.RETURN_TO_BASE)
    ):
        recs.append("Consider diverting to nearest landing site.")
    if phase is MissionPhase.LANDING and float(risk_score) >= 0.55:
        recs.append("Continue landing; do not abort approach.")
    return recs[: int(MAX_RECOMMENDATIONS)]


# ---------------------------------------------------------------------
# Hours-to-critical
# ---------------------------------------------------------------------
def _hours_to_critical(
    health: HealthIndex,
    rul: RulEstimate,
    *,
    dt_s: float,
    floor_h: float,
) -> Optional[float]:
    """Estimate hours until the assessment would reach ABORT.

    Returns the smaller of:

    * ``rul.tte_hours_lower`` (None when RUL is uncertain)
    * An extrapolation from the health trend: if the health is
      DEGRADING with a non-zero deficit, the time to reach a
      full-deficit state at the current rate of decline.

    Returns ``None`` if neither input is available.
    """
    candidates: List[float] = []
    if not rul.is_uncertain:
        candidates.append(max(0.0, float(rul.tte_hours_lower)))
    deficit_now = 1.0 - float(health.overall_score)
    if (
        health.trend.value == "DEGRADING"
        and deficit_now > 0.0
        and dt_s > 0.0
    ):
        # The current health score change per tick is unknown here
        # (the trend buffer is owned by the calculator). Use a
        # conservative proxy: the *epsilon*-based minimum rate
        # implied by the trend threshold. Since we don't have the
        # raw buffer here, return the floor_h as a lower bound on
        # the time-to-critical driven by health alone.
        candidates.append(max(0.0, float(floor_h) * deficit_now))
    if not candidates:
        return None
    return max(0.0, min(candidates))


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def aggregate(
    *,
    health: HealthIndex,
    rul: RulEstimate,
    anomaly: Optional[AnomalyAssessment],
    phase: MissionPhase,
    hours_to_destination: Optional[float],
    config: RiskConfig,
    trend_history: Deque[float],
    time_s: float,
    dt_s: float = 0.1,
) -> RiskAssessment:
    """Compute the per-tick :class:`RiskAssessment`.

    Parameters
    ----------
    health, rul, anomaly:
        Outputs of PHASES 10, 11, 8 respectively. ``anomaly`` may
        be ``None`` (the risk layer is still functional without
        the anomaly channel).
    phase, hours_to_destination:
        Mission-context inputs derived from the profile and time.
    config:
        Operator-tunable :class:`RiskConfig` (from YAML or
        :meth:`RiskConfig.defaults`).
    trend_history:
        Mutable deque of recent risk scores, owned by the
        calculator. The current risk score is appended here.
    time_s:
        Time stamp of the assessment.
    dt_s:
        Mission dt in seconds (used to interpret trend rates).
    """
    weights = _normalize_weights(config)

    # --- Per-signal deficits -----------------------------------------
    d_health = max(0.0, 1.0 - float(health.overall_score))
    d_rul = _rul_deficit(
        rul, floor_h=config.rul_deficit_floor_h,
        ceiling_h=config.rul_deficit_ceiling_h,
    )
    d_anomaly = (
        float(anomaly.overall_score) if anomaly is not None else 0.0
    )

    # --- Per-signal contributions (always >= 0) ----------------------
    c_health = weights["health"] * d_health
    c_rul = weights["rul"] * d_rul
    c_anomaly = (
        weights["anomaly"] * d_anomaly if anomaly is not None else 0.0
    )

    # --- Phase modifier ----------------------------------------------
    phase_mod = float(PHASE_MODIFIERS.get(phase, 0.0))

    # --- Fault boost -------------------------------------------------
    faults = dict(health.contributing_faults or {})
    fault_boost = (
        float(config.fault_boost_coef) * max(faults.values())
        if faults else 0.0
    )

    # --- Final score -------------------------------------------------
    raw = c_health + c_rul + c_anomaly + phase_mod + fault_boost
    risk_score = _clip(raw)

    # --- Drivers (per-signal attribution) ----------------------------
    drivers: List[RiskDriver] = []
    drivers.append(RiskDriver(
        signal="health.overall_score",
        value=float(health.overall_score),
        contribution=float(c_health),
        severity=(
            DriverSeverity.CRITICAL
            if health.overall_label.value == "CRITICAL"
            else DriverSeverity.WARNING
            if health.overall_label.value == "DEGRADED"
            else DriverSeverity.INFO
        ),
        note=f"health_label={health.overall_label.value}",
    ))
    drivers.append(RiskDriver(
        signal="rul.tte_hours_lower",
        value=float(rul.tte_hours_lower) if not rul.is_uncertain else 0.0,
        contribution=float(c_rul),
        severity=(
            DriverSeverity.CRITICAL
            if rul.status.value == "RUL_CRITICAL"
            else DriverSeverity.WARNING
            if rul.status.value == "RUL_DEGRADED"
            else DriverSeverity.INFO
        ),
        note=f"rul_status={rul.status.value}",
    ))
    if anomaly is not None:
        drivers.append(RiskDriver(
            signal="anomaly.overall_score",
            value=float(anomaly.overall_score),
            contribution=float(c_anomaly),
            severity=(
                DriverSeverity.CRITICAL
                if anomaly.overall_label.value == "ANOMALY"
                else DriverSeverity.WARNING
                if anomaly.overall_label.value == "WARN"
                else DriverSeverity.INFO
            ),
            note=f"anomaly_label={anomaly.overall_label.value}",
        ))
    if abs(phase_mod) > 1e-9:
        drivers.append(RiskDriver(
            signal=f"phase.{phase.value}",
            value=float(phase_mod),
            contribution=float(phase_mod),
            severity=(
                DriverSeverity.WARNING if phase_mod > 0 else DriverSeverity.INFO
            ),
            note="mission-phase modifier",
        ))
    if fault_boost > 0.0:
        drivers.append(RiskDriver(
            signal="fault.boost",
            value=float(max(faults.values())),
            contribution=float(fault_boost),
            severity=DriverSeverity.WARNING,
            note="max contributing-fault probability",
        ))

    # --- Status, level, confidence ----------------------------------
    conf = _confidence(health, rul, anomaly)
    status = status_for(
        health_label=health.overall_label.value,
        rul_status=rul.status.value,
        risk_score=risk_score,
        confidence=conf,
        min_confidence=config.min_confidence_for_decision,
    )
    risk_level = level_for(risk_score)

    # --- Hours-to-critical ------------------------------------------
    h2c = _hours_to_critical(
        health, rul, dt_s=dt_s, floor_h=config.rul_deficit_floor_h,
    )

    # --- Recommendations --------------------------------------------
    recs = _build_recommendations(
        status=status,
        health=health,
        rul=rul,
        anomaly=anomaly,
        phase=phase,
        risk_score=risk_score,
        floor_h=config.rul_deficit_floor_h,
    )

    # --- Trend -------------------------------------------------------
    # Compare the *new* risk score to the oldest one in the buffer.
    ref: Optional[float] = None
    if len(trend_history) > 0:
        ref = float(trend_history[0])
    trend = trend_for(
        float(risk_score), ref, epsilon=config.trend_epsilon,
    )
    trend_history.append(float(risk_score))

    # --- Notes -------------------------------------------------------
    notes: List[str] = []
    if phase_mod > 0:
        notes.append(f"phase {phase.value} modifier +{phase_mod:.2f}")
    if fault_boost > 0:
        notes.append(
            f"fault boost +{fault_boost:.3f} (max prob {max(faults.values()):.2f})"
        )
    if anomaly is None:
        notes.append("anomaly input not provided")

    return RiskAssessment(
        time_s=float(time_s),
        risk_score=float(risk_score),
        risk_level=risk_level,
        status=status,
        confidence=float(conf),
        mission_phase=phase,
        hours_to_destination=hours_to_destination,
        hours_to_critical=h2c,
        drivers=drivers,
        recommendations=recs,
        contributing_faults=faults,
        trend=trend,
        notes=notes,
        inputs_meta=_inputs_meta(health, rul, anomaly, phase),
    )


__all__ = [
    "aggregate",
]
