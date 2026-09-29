"""
Pipeline invariant checkers (PHASE 16).

Pure functions that the fault-detection contract, soak tests, and
property tests all share. Each checker raises
:class:`InvariantViolation` with a structured message on failure,
so a CI report reads cleanly.

Invariants:

* ``assert_no_nan_inf`` — every numeric in a snapshot dict is
  finite.
* ``assert_health_invariants`` — ``HealthIndex.overall_score`` and
  per-subsystem scores are in [0, 1]; confidence is in [0, 1].
* ``assert_rul_invariants`` — ``RulEstimate`` lower <= central
  <= upper, all non-negative.
* ``assert_risk_invariants`` — ``RiskAssessment.risk_score`` in
  [0, 1], status is a known ``RiskStatus``.

A failure here is a regression in a per-tick contract — the
dashboard or downstream consumers depend on these.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from backend.diagnostics import AnomalyAssessment
from backend.health import HealthIndex, SubsystemHealth
from backend.risk import RiskAssessment, RiskStatus
from backend.rul import RulEstimate


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------
class InvariantViolation(AssertionError):
    """Raised when a pipeline invariant is broken.

    Carries ``path`` (dotted field location), ``value`` (the bad
    value), and ``invariant`` (which rule fired) so the test
    report is actionable.
    """

    def __init__(self, path: str, value: Any, invariant: str) -> None:
        self.path = path
        self.value = value
        self.invariant = invariant
        super().__init__(
            f"invariant violated at {path!r}: {invariant} (got {value!r})"
        )


# ---------------------------------------------------------------------
# NaN / inf
# ---------------------------------------------------------------------
def assert_no_nan_inf(obj: Any, path: str = "") -> None:
    """Walk ``obj`` (dict / list / scalar) and raise on any NaN/inf.

    Strings, bools, and ``None`` are skipped — only floats + ints
    are checked. The traversal is shallow on purpose: per-snapshot
    the dict is flat enough (no nested user data) that a recursive
    walk is cheap and catches everything.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_nan_inf(v, f"{path}.{k}" if path else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_nan_inf(v, f"{path}[{i}]")
        return
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            raise InvariantViolation(path, obj, "must be a finite number")
        return
    # Other types (str, Enum) — skip.


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
def assert_health_invariants(h: HealthIndex) -> None:
    """HealthIndex: overall_score in [0, 1]; per-subsystem in [0, 1]."""
    if not (0.0 <= h.overall_score <= 1.0):
        raise InvariantViolation(
            f"health.overall_score", h.overall_score, "must be in [0, 1]"
        )
    if not (0.0 <= h.confidence <= 1.0):
        raise InvariantViolation(
            f"health.confidence", h.confidence, "must be in [0, 1]"
        )
    for name, sub in (h.subsystems or {}).items():
        if not isinstance(sub, SubsystemHealth):
            continue
        if not (0.0 <= sub.score <= 1.0):
            raise InvariantViolation(
                f"health.subsystems.{name}.score", sub.score,
                "must be in [0, 1]",
            )
        if not (0.0 <= sub.confidence <= 1.0):
            raise InvariantViolation(
                f"health.subsystems.{name}.confidence", sub.confidence,
                "must be in [0, 1]",
            )


# ---------------------------------------------------------------------
# RUL
# ---------------------------------------------------------------------
def assert_rul_invariants(r: RulEstimate) -> None:
    """RulEstimate: lower <= central <= upper; all non-negative.

    Each estimate may be ``None`` (RUL_UNCERTAIN); in that case
    we only assert the bound ordering when all three are
    present. The non-negativity check is also skipped on None.
    """
    lo = r.tte_hours_lower
    ce = r.tte_hours_central
    up = r.tte_hours_upper
    if lo is not None and lo < 0:
        raise InvariantViolation(
            "rul.tte_hours_lower", lo, "must be non-negative",
        )
    if ce is not None and ce < 0:
        raise InvariantViolation(
            "rul.tte_hours_central", ce, "must be non-negative",
        )
    if up is not None and up < 0:
        raise InvariantViolation(
            "rul.tte_hours_upper", up, "must be non-negative",
        )
    if lo is not None and ce is not None and lo > ce:
        raise InvariantViolation(
            "rul.bounds", (lo, ce),
            "lower must be <= central",
        )
    if ce is not None and up is not None and ce > up:
        raise InvariantViolation(
            "rul.bounds", (ce, up),
            "central must be <= upper",
        )


# ---------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------
def assert_risk_invariants(a: RiskAssessment) -> None:
    """RiskAssessment: risk_score in [0, 1]; status is a known enum."""
    if not (0.0 <= a.risk_score <= 1.0):
        raise InvariantViolation(
            "risk.risk_score", a.risk_score, "must be in [0, 1]"
        )
    valid = {s.value for s in RiskStatus}
    if a.status.value not in valid:
        raise InvariantViolation(
            "risk.status", a.status, f"must be one of {sorted(valid)}"
        )


# ---------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------
def assert_anomaly_invariants(a: AnomalyAssessment) -> None:
    """AnomalyAssessment: overall_score in [0, 1]."""
    if not (0.0 <= a.overall_score <= 1.0):
        raise InvariantViolation(
            "anomaly.overall_score", a.overall_score, "must be in [0, 1]"
        )
    if not (0.0 <= a.confidence <= 1.0):
        raise InvariantViolation(
            "anomaly.confidence", a.confidence, "must be in [0, 1]"
        )


# ---------------------------------------------------------------------
# Snapshot aggregate (typed input)
# ---------------------------------------------------------------------
def assert_typed_snapshot_invariants(
    health: HealthIndex,
    rul: RulEstimate,
    risk: RiskAssessment,
    anomaly: AnomalyAssessment | None = None,
) -> None:
    """Run every per-snapshot invariant over typed objects."""
    assert_health_invariants(health)
    assert_rul_invariants(rul)
    assert_risk_invariants(risk)
    if anomaly is not None:
        assert_anomaly_invariants(anomaly)


# ---------------------------------------------------------------------
# Snapshot aggregate (dict input — best-effort)
# ---------------------------------------------------------------------
def assert_snapshot_dict_invariants(snap_dict: Dict[str, Any]) -> None:
    """Check NaN/inf on the dict; structural checks on the per-tick
    block-level fields that we know how to validate without
    round-tripping through the typed objects.

    The typed invariants (health/RUL/risk in [0,1] etc.) require
    re-materialising the typed objects, which the snapshot
    DashboardSnapshot does for us. Callers that already have a
    DashboardSnapshot should call :func:`assert_typed_snapshot_invariants`
    directly.
    """
    assert_no_nan_inf(snap_dict)
    # Best-effort float-range checks on the known float fields.
    _range_check(snap_dict.get("health"), "overall_score", 0.0, 1.0)
    _range_check(snap_dict.get("health"), "confidence", 0.0, 1.0)
    _range_check(snap_dict.get("rul"), "tte_hours_central", 0.0, None)
    _range_check(snap_dict.get("rul"), "tte_hours_lower", 0.0, None)
    _range_check(snap_dict.get("rul"), "tte_hours_upper", 0.0, None)
    _range_check(snap_dict.get("risk"), "risk_score", 0.0, 1.0)
    _range_check(snap_dict.get("risk"), "confidence", 0.0, 1.0)
    if snap_dict.get("anomaly") is not None:
        _range_check(snap_dict["anomaly"], "overall_score", 0.0, 1.0)
        _range_check(snap_dict["anomaly"], "confidence", 0.0, 1.0)
    # RUL bounds
    rul = snap_dict.get("rul") or {}
    lower = rul.get("tte_hours_lower")
    central = rul.get("tte_hours_central")
    upper = rul.get("tte_hours_upper")
    if lower is not None and central is not None and lower > central:
        raise InvariantViolation(
            "rul.bounds", (lower, central), "lower must be <= central",
        )
    if central is not None and upper is not None and central > upper:
        raise InvariantViolation(
            "rul.bounds", (central, upper), "central must be <= upper",
        )


def _range_check(d: Dict[str, Any] | None, key: str, lo: float, hi: float | None) -> None:
    if d is None or key not in d:
        return
    v = d[key]
    if v is None or isinstance(v, bool):
        return
    if not (lo <= float(v)):
        raise InvariantViolation(
            f"{key}", v, f"must be >= {lo}",
        )
    if hi is not None and not (float(v) <= hi):
        raise InvariantViolation(
            f"{key}", v, f"must be <= {hi}",
        )


__all__ = [
    "InvariantViolation",
    "assert_anomaly_invariants",
    "assert_health_invariants",
    "assert_no_nan_inf",
    "assert_risk_invariants",
    "assert_rul_invariants",
    "assert_snapshot_dict_invariants",
    "assert_typed_snapshot_invariants",
]
