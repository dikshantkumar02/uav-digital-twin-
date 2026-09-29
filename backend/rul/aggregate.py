"""
RUL aggregator (PHASE 11).

Combines a :class:`~backend.rul.monte_carlo.TteDistribution` and
the upstream :class:`~backend.health.HealthIndex` into a
:class:`~backend.rul.types.RulEstimate`.

The aggregator is **stateless** except for the trend buffer. The
:class:`RulCalculator` orchestrator owns the trend buffer's
lifetime.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

from backend.health import HealthIndex, HealthLabel, HealthTrend

from .monte_carlo import TteDistribution
from .types import (
    RulEstimate,
    RulModelStatus,
    RulStatus,
    RulTrend,
    TREND_WINDOW,
    status_for,
    trend_for,
)


def aggregate(
    *,
    time_s: float,
    health: HealthIndex,
    tte: TteDistribution,
    wear_rate_per_hour: float,
    model_status: RulModelStatus,
    trend_history: Optional[Deque[float]] = None,
    notes: Optional[list[str]] = None,
) -> RulEstimate:
    """Combine TTE distribution + health into a :class:`RulEstimate`.

    Parameters
    ----------
    time_s:
        Time stamp (s) of this update.
    health:
        The current :class:`HealthIndex` (PHASE 10).
    tte:
        Empirical TTE distribution from the Monte-Carlo layer.
    wear_rate_per_hour:
        Current wear rate used as the MC central rate.
    model_status:
        Whether the estimate came from a trained model or the
        closed-form fallback.
    trend_history:
        Optional deque of recent central TTE values for the trend
        comparison. If provided and the deque has at least
        ``TREND_WINDOW`` entries, the trend is computed.
    notes:
        Optional list of notes to attach to the estimate.
    """
    # Map the health label/trend to a coarse-grained RUL status.
    health_label_value = (
        health.overall_label.value if health.overall_label is not None
        else HealthLabel.INSUFFICIENT_DATA.value
    )
    health_trend_value = (
        health.trend.value if health.trend is not None
        else HealthTrend.INSUFFICIENT_DATA.value
    )
    status = status_for(
        health_label=health_label_value,
        health_trend=health_trend_value,
        confidence=float(tte.confidence),
    )

    # Compute the trend from the TTE history (if available).
    trend = _compute_trend(tte.central, trend_history)

    # Build the notes.
    final_notes: list[str] = list(notes or [])
    if status is RulStatus.RUL_UNCERTAIN:
        if tte.confidence <= 0.0:
            final_notes.append("no samples reached EOL within horizon")
        elif tte.confidence < 0.5:
            final_notes.append("low confidence in TTE estimate")
        else:
            final_notes.append("insufficient health evidence")
    if status is RulStatus.RUL_CRITICAL:
        final_notes.append("engine health critical or estimate low-confidence")
    if tte.spread_hours > 0 and tte.spread_hours > 0.5 * tte.central:
        final_notes.append("wide uncertainty bounds")

    return RulEstimate(
        time_s=float(time_s),
        tte_hours_central=float(tte.central),
        tte_hours_lower=float(tte.lower),
        tte_hours_upper=float(tte.upper),
        wear_rate_per_hour=float(wear_rate_per_hour),
        confidence=float(tte.confidence),
        status=status,
        trend=trend,
        model_status=model_status,
        contributing_faults=dict(health.contributing_faults or {}),
        notes=final_notes,
    )


def new_trend_history() -> Deque[float]:
    """Return a fresh, empty trend-history deque."""
    return deque(maxlen=TREND_WINDOW)


def update_trend_history(history: Deque[float], central_tte: float) -> None:
    """Append a central TTE value to the history (FIFO)."""
    history.append(float(central_tte))


def _compute_trend(
    current: float,
    history: Optional[Deque[float]],
) -> RulTrend:
    if history is None or len(history) < TREND_WINDOW:
        return RulTrend.INSUFFICIENT_DATA
    reference = history[0]
    return trend_for(current, reference)


__all__ = [
    "aggregate",
    "new_trend_history",
    "update_trend_history",
]
