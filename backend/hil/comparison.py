"""
Tolerance-banded comparison of HIL replay runs (PHASE 15).

A :func:`compare_runs` call walks two JSONL streams of
``DashboardSnapshot.to_dict()`` outputs tick-by-tick and emits a flat
list of :class:`Mismatch` records for any per-field drift that
exceeds the supplied :class:`Tolerance`. Floats are compared with a
per-field absolute or relative band; structural fields (booleans,
ints, enums, strings) are compared exactly; ``None`` is sacred — a
``None`` in the golden run is never silently coerced to a default.

The point of the walker is to surface *semantic* regressions (the
overall risk score moved 5 points) without false-failing on the
floating-point jitter that a different BLAS / SIMD kernel would
introduce (a 1e-12 drift in CHT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    """Per-field tolerance bands for the comparator.

    Absolutes are in the natural unit of the field. Relatives are
    fractions of ``max(|expected|, 1)`` so small magnitudes still have
    a meaningful bound. See :data:`backend.config.HilToleranceConfig`
    for the YAML / Pydantic defaults.

    The ``default_abs`` band covers top-level numeric fields (e.g.
    ``time_s``, ``tick_index``) that are not under a sub-tree with a
    domain-specific tolerance. ``1e-9`` is large enough to absorb the
    IEEE-754 round-trip noise introduced by a wire codec that stores
    floats at 6 decimal places but small enough to catch a real
    regression.
    """

    default_abs: float = 1e-9
    health_abs: float = 0.01
    rul_abs: float = 1.0              # hours
    risk_abs: float = 0.02
    engine_state_rel: float = 0.005   # 0.5 %
    environment_rel: float = 0.005
    anomaly_abs: float = 0.02


# ---------------------------------------------------------------------
# Mismatch
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Mismatch:
    """One field-level drift outside the tolerance band."""

    tick_index: int
    field_path: str
    expected: Any
    actual: Any
    delta: Optional[float] = None
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_index": int(self.tick_index),
            "field_path": self.field_path,
            "expected": self.expected,
            "actual": self.actual,
            "delta": self.delta,
            "tolerance": self.tolerance,
        }


# ---------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ComparisonReport:
    """Aggregate result of one :func:`compare_runs` call."""

    n_ticks_compared: int
    n_mismatches: int
    mismatches: List[Mismatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_mismatches == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ticks_compared": int(self.n_ticks_compared),
            "n_mismatches": int(self.n_mismatches),
            "passed": bool(self.passed),
            "mismatches": [m.to_dict() for m in self.mismatches],
        }


# ---------------------------------------------------------------------
# Field-classification table
# ---------------------------------------------------------------------
# Which sub-tree gets which tolerance. Keys are top-level paths
# under DashboardSnapshot.to_dict().
_ABSOLUTE_TOPLEVEL: Dict[str, float] = {
    "health": 0.0,        # filled from Tolerance.health_abs at call time
    "rul": 0.0,           # Tolerance.rul_abs (hours)
    "risk": 0.0,          # Tolerance.risk_abs
}
_RELATIVE_TOPLEVEL: Dict[str, float] = {
    "engine_state": 0.0,  # Tolerance.engine_state_rel
    "environment": 0.0,   # Tolerance.environment_rel
}


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _abs_tol_for(path: str, tol: Tolerance) -> Optional[float]:
    """Return the absolute tolerance to apply at ``path`` (or None)."""
    parts = path.split(".")
    head = parts[0]
    if head == "health":
        return tol.health_abs
    if head == "rul":
        # The RUL block contains a `confidence` float in [0, 1] —
        # apply the smaller of risk_abs and rul_abs so a confidence
        # drop of >2 % is flagged.
        return tol.rul_abs if "remaining_hours" in path else tol.risk_abs
    if head == "risk":
        return tol.risk_abs
    if head == "anomaly":
        return tol.anomaly_abs
    # Top-level numeric fields (time_s, etc.) use the default band.
    return tol.default_abs


def _rel_tol_for(path: str, tol: Tolerance) -> Optional[float]:
    parts = path.split(".")
    head = parts[0]
    if head == "engine_state":
        return tol.engine_state_rel
    if head == "environment":
        return tol.environment_rel
    return None


def _compare_floats(
    expected: float,
    actual: float,
    *,
    abs_tol: Optional[float],
    rel_tol: Optional[float],
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Return (within_tol, delta, applied_tol)."""
    delta = float(actual) - float(expected)
    if delta == 0.0:
        return True, 0.0, abs_tol if abs_tol is not None else rel_tol
    if abs_tol is not None and abs(delta) <= abs_tol:
        return True, delta, abs_tol
    denom = max(abs(float(expected)), 1.0)
    if rel_tol is not None and abs(delta) / denom <= rel_tol:
        return True, delta, rel_tol
    applied = abs_tol if abs_tol is not None else rel_tol
    return False, delta, applied


def _walk(
    expected: Any,
    actual: Any,
    path: str,
    tick_index: int,
    tol: Tolerance,
    out: List[Mismatch],
) -> None:
    # Both missing -> nothing to do.
    if expected is None and actual is None:
        return
    # None on one side only -> exact mismatch (no silent substitution).
    if expected is None or actual is None:
        out.append(Mismatch(
            tick_index=tick_index,
            field_path=path,
            expected=expected,
            actual=actual,
        ))
        return
    # Dict: recurse.
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual))
        for k in keys:
            _walk(expected.get(k), actual.get(k), f"{path}.{k}" if path else k,
                  tick_index, tol, out)
        return
    # Booleans: exact.
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            out.append(Mismatch(
                tick_index=tick_index,
                field_path=path,
                expected=expected,
                actual=actual,
            ))
        return
    # Floats / ints: tolerance-banded.
    if _is_finite_number(expected) and _is_finite_number(actual):
        abs_tol = _abs_tol_for(path, tol)
        rel_tol = _rel_tol_for(path, tol)
        ok, delta, applied = _compare_floats(
            float(expected), float(actual), abs_tol=abs_tol, rel_tol=rel_tol,
        )
        if not ok:
            out.append(Mismatch(
                tick_index=tick_index,
                field_path=path,
                expected=expected,
                actual=actual,
                delta=delta,
                tolerance=applied,
            ))
        return
    # Strings / enums / anything else: exact.
    if expected != actual:
        out.append(Mismatch(
            tick_index=tick_index,
            field_path=path,
            expected=expected,
            actual=actual,
        ))


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def compare_runs(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    tolerance: Tolerance,
) -> ComparisonReport:
    """Walk both run dicts tick-by-tick and return a :class:`ComparisonReport`.

    Both inputs are lists of ``DashboardSnapshot.to_dict()`` payloads
    (one per tick). The lists must be in the same tick order; the
    comparison is positional.

    A length mismatch is reported as a single ``Mismatch`` with
    ``field_path = "<tick_count>"`` and the lengths as ``expected`` /
    ``actual``.
    """
    mismatches: List[Mismatch] = []
    if len(expected) != len(actual):
        mismatches.append(Mismatch(
            tick_index=-1,
            field_path="<tick_count>",
            expected=len(expected),
            actual=len(actual),
        ))
    n = min(len(expected), len(actual))
    for i in range(n):
        _walk(expected[i], actual[i], "", i, tolerance, mismatches)
    return ComparisonReport(
        n_ticks_compared=int(n),
        n_mismatches=len(mismatches),
        mismatches=mismatches,
    )


__all__ = [
    "ComparisonReport",
    "Mismatch",
    "Tolerance",
    "compare_runs",
]
