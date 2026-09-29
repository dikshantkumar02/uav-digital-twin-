"""
Health aggregator (PHASE 10).

Combines per-subsystem :class:`~backend.health.types.SubsystemHealth`
scores into a single :class:`~backend.health.types.HealthIndex`
using configurable weights, and tracks a **trend** over time.

The aggregator is **stateless** except for the trend buffer. The
orchestrator (:class:`HealthIndexCalculator`) owns the trend
buffer's lifetime.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Mapping, Optional, Tuple

from .types import (
    DEFAULT_WEIGHTS,
    HealthIndex,
    HealthLabel,
    HealthTrend,
    Subsystem,
    SubsystemHealth,
    label_for,
)
from backend._common import DEFAULT_TREND_WINDOW


# Trend comparison threshold: |Δscore| < TREND_EPSILON is STABLE.
TREND_EPSILON: float = 0.02

# Number of recent overall scores to keep for the trend calculation.
# Re-exported from backend._common so the four trend-detector
# modules (RUL, risk, diagnostics, health) cannot drift apart
# silently. Override locally only if the health-index
# buffer needs a different size.
TREND_WINDOW: int = DEFAULT_TREND_WINDOW

# Confidence floor below which the overall index is reported as
# INSUFFICIENT_DATA (conservative).
MIN_OVERALL_CONFIDENCE: float = 0.20


def normalize_weights(
    weights: Mapping[Subsystem, float],
) -> Dict[Subsystem, float]:
    """Return a copy of ``weights`` rescaled to sum to 1.0.

    If the input is empty or sums to 0, the function falls back to
    :data:`DEFAULT_WEIGHTS`.
    """
    if not weights:
        return dict(DEFAULT_WEIGHTS)
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0.0:
        return dict(DEFAULT_WEIGHTS)
    return {k: float(max(0.0, v)) / total for k, v in weights.items()}


def aggregate(
    subsystems: Mapping[Subsystem, SubsystemHealth],
    *,
    weights: Optional[Mapping[Subsystem, float]] = None,
    time_s: float = 0.0,
    wear: Optional[float] = None,
    contributing_faults: Optional[Dict[str, float]] = None,
    trend_history: Optional[Deque[float]] = None,
    notes: Optional[list[str]] = None,
) -> HealthIndex:
    """Combine subsystem scores into a :class:`HealthIndex`.

    Parameters
    ----------
    subsystems:
        Mapping from subsystem name to its :class:`SubsystemHealth`.
    weights:
        Optional override for the subsystem weights. Defaults to
        :data:`DEFAULT_WEIGHTS`.
    time_s:
        Time stamp (s) of this update.
    wear:
        Optional engine wear value in [0, 1] (from the truth model).
    contributing_faults:
        Optional ``{fault_class_str: probability}`` dict from the
        PHASE 9 classifier.
    trend_history:
        Optional deque of recent overall scores. If provided and
        the deque has at least ``TREND_WINDOW`` entries, the trend
        is computed from the oldest entry vs. the current score.
    notes:
        Optional list of notes to attach to the index.
    """
    w = normalize_weights(weights or DEFAULT_WEIGHTS)

    # --- 1) overall score (weighted mean) --------------------------
    total_w = 0.0
    total = 0.0
    confidence_acc = 0.0
    confidence_n = 0
    if subsystems:
        for name, sub in subsystems.items():
            weight = float(w.get(name, 0.0))
            if weight <= 0.0:
                continue
            # Zero-confidence subsystems (no evidence) contribute
            # zero to the score and zero to the weighted mean. The
            # confidence gate below ensures the overall label
            # degrades to INSUFFICIENT_DATA when too much of the
            # index has no evidence.
            if sub.confidence <= 0.0:
                continue
            total += weight * float(sub.score)
            total_w += weight
            confidence_acc += float(sub.confidence)
            confidence_n += 1

    if total_w > 0.0:
        overall_score = max(0.0, min(1.0, total / total_w))
    else:
        overall_score = 0.0
    overall_conf = (
        confidence_acc / float(confidence_n) if confidence_n > 0 else 0.0
    )

    # --- 2) overall label ------------------------------------------
    if overall_conf < MIN_OVERALL_CONFIDENCE:
        overall_label = HealthLabel.INSUFFICIENT_DATA
    else:
        overall_label = label_for(overall_score)

    # --- 3) trend --------------------------------------------------
    trend = _compute_trend(overall_score, trend_history)

    # --- 4) notes --------------------------------------------------
    final_notes = list(notes or [])
    if overall_label is HealthLabel.INSUFFICIENT_DATA:
        final_notes.append("insufficient evidence to score health")
    if wear is not None and wear >= 0.5:
        final_notes.append(f"engine wear elevated ({wear:.2f})")

    return HealthIndex(
        time_s=float(time_s),
        overall_score=float(overall_score),
        overall_label=overall_label,
        confidence=float(overall_conf),
        subsystems=dict(subsystems),
        trend=trend,
        wear=wear,
        contributing_faults=dict(contributing_faults or {}),
        notes=final_notes,
    )


def _compute_trend(
    current_score: float,
    history: Optional[Deque[float]],
) -> HealthTrend:
    """Compare the current score to a reference point in the history."""
    if history is None or len(history) < TREND_WINDOW:
        return HealthTrend.INSUFFICIENT_DATA
    # ``history`` is appended in chronological order: index 0 is
    # the oldest, index -1 is the most recent (one tick ago).
    reference = history[0]
    delta = float(current_score) - float(reference)
    if delta > TREND_EPSILON:
        return HealthTrend.IMPROVING
    if delta < -TREND_EPSILON:
        return HealthTrend.DEGRADING
    return HealthTrend.STABLE


def update_trend_history(
    history: Deque[float],
    score: float,
) -> None:
    """Append ``score`` to ``history`` (FIFO, maxlen=:data:`TREND_WINDOW`)."""
    history.append(float(score))


# ---------------------------------------------------------------------
# Convenience: trend buffer factory
# ---------------------------------------------------------------------
def new_trend_history() -> Deque[float]:
    """Return a fresh, empty trend-history deque."""
    return deque(maxlen=TREND_WINDOW)


__all__ = [
    "MIN_OVERALL_CONFIDENCE",
    "TREND_EPSILON",
    "TREND_WINDOW",
    "aggregate",
    "new_trend_history",
    "normalize_weights",
    "update_trend_history",
]
