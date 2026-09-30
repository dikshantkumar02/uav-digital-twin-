"""
Public types for the Remaining-Useful-Life layer (PHASE 11).

Three dataclasses are the **stable surface** the rest of the system
consumes:

* :class:`RulStatus` — coarse-grained status (RUL_OK / RUL_DEGRADED
  / RUL_CRITICAL / RUL_UNCERTAIN). Mirrors the spec's
  "no fake outputs" pattern: missing evidence is reported as such,
  never faked.
* :class:`RulTrend` — direction of the TTE estimate over recent
  ticks (IMPROVING / STABLE / DEGRADING / INSUFFICIENT_DATA).
* :class:`RulEstimate` — the per-tick output. Always carries a
  :class:`RulStatus` so consumers can tell a confident estimate
  apart from "I don't know yet."

* :class:`ResidualTrendInput` — a small snapshot of the Digital
  Twin residual history, threaded into the feature vector so the
  wear-rate model can see "is the residual growing?" — the
  spec's "Digital Twin residual trends" input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend._common import DEFAULT_TREND_WINDOW


class RulStatus(str, Enum):
    """Coarse-grained RUL status.

    * ``RUL_OK`` — health is healthy and stable, TTE is large.
    * ``RUL_DEGRADED`` — health is degraded or trending down,
      TTE is reduced.
    * ``RUL_CRITICAL`` — health is critical OR confidence is low.
    * ``RUL_UNCERTAIN`` — not enough evidence to decide; the
      TTE estimate is not meaningful. This mirrors the spec's
      ``INSUFFICIENT_DATA`` / ``RUL_UNCERTAIN`` pattern: a
      missing output is reported as such, never faked.
    """

    RUL_OK = "RUL_OK"
    RUL_DEGRADED = "RUL_DEGRADED"
    RUL_CRITICAL = "RUL_CRITICAL"
    RUL_UNCERTAIN = "RUL_UNCERTAIN"


class RulTrend(str, Enum):
    """Trend of the central TTE estimate over recent ticks.

    * ``IMPROVING`` — TTE_now > TTE_ref + 0.5h.
    * ``STABLE`` — |TTE_now - TTE_ref| <= 0.5h.
    * ``DEGRADING`` — TTE_now < TTE_ref - 0.5h.
    * ``INSUFFICIENT_DATA`` — not enough history to compare.
    """

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RulModelStatus(str, Enum):
    """Status of the underlying wear-rate model.

    * ``CLOSED_FORM`` — using the closed-form fallback (no trained
      model loaded). The estimate is a pure-function projection
      with no learned correction.
    * ``MODEL_CALIBRATED`` — a trained RF regressor is in use.
    * ``MODEL_DEGRADED`` — model loaded but held-out accuracy
      below the floor; the estimate is still emitted but flagged.
    """

    CLOSED_FORM = "CLOSED_FORM"
    MODEL_CALIBRATED = "MODEL_CALIBRATED"
    MODEL_DEGRADED = "MODEL_DEGRADED"


# TTE-trend threshold: |Δtte_hours| < TREND_EPSILON_HOURS is STABLE.
TREND_EPSILON_HOURS: float = 0.5

# Number of recent central TTE values kept for the trend.
# Re-exported from backend._common so the four trend-detector
# modules (RUL, risk, diagnostics, health) cannot drift apart
# silently. Override locally only if the RUL-specific
# buffer needs a different size.
TREND_WINDOW: int = DEFAULT_TREND_WINDOW


def status_for(
    *,
    health_label: Optional[str] = None,
    health_trend: Optional[str] = None,
    confidence: float = 0.0,
) -> RulStatus:
    """Map health signals to a :class:`RulStatus`.

    The mapping is intentionally conservative — when in doubt,
    return ``RUL_UNCERTAIN``. Per the spec: "no fake outputs."
    """
    if health_label == "INSUFFICIENT_DATA" or confidence < 0.20:
        return RulStatus.RUL_UNCERTAIN
    if health_label == "CRITICAL" or confidence < 0.40:
        return RulStatus.RUL_CRITICAL
    if health_label == "DEGRADED" or health_trend == "DEGRADING":
        return RulStatus.RUL_DEGRADED
    return RulStatus.RUL_OK


def trend_for(
    current: Optional[float],
    reference: Optional[float],
) -> RulTrend:
    """Compare a current TTE to a reference TTE."""
    if current is None or reference is None:
        return RulTrend.INSUFFICIENT_DATA
    delta = float(current) - float(reference)
    if delta > TREND_EPSILON_HOURS:
        return RulTrend.IMPROVING
    if delta < -TREND_EPSILON_HOURS:
        return RulTrend.DEGRADING
    return RulTrend.STABLE


@dataclass(frozen=True)
class RulBounds:
    """Confidence bounds on the TTE estimate (hours)."""

    lower: float
    central: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.central:
            raise ValueError(
                f"lower ({self.lower}) must be <= central ({self.central})"
            )
        if self.central > self.upper:
            raise ValueError(
                f"central ({self.central}) must be <= upper ({self.upper})"
            )

    @property
    def spread_hours(self) -> float:
        return float(self.upper) - float(self.lower)

    def to_dict(self) -> dict:
        return {
            "lower": float(self.lower),
            "central": float(self.central),
            "upper": float(self.upper),
            "spread_hours": float(self.spread_hours),
        }


@dataclass(frozen=True)
class RulEstimate:
    """Per-tick Remaining-Useful-Life output.

    The ``tte_hours_*`` fields are **always present** (no NaN), but
    the ``status`` field is the source of truth for whether the
    estimate is meaningful. When ``status`` is
    ``RUL_UNCERTAIN``, the values are placeholders reported for
    API stability; consumers MUST check ``status`` first.
    """

    time_s: float
    tte_hours_central: float
    tte_hours_lower: float
    tte_hours_upper: float
    wear_rate_per_hour: float                    # current estimated rate
    confidence: float                            # 0..1
    status: RulStatus
    trend: RulTrend
    model_status: RulModelStatus
    contributing_faults: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    remaining_hours: Optional[float] = None
    remaining_cycles: Optional[int] = None
    health_index: Optional[float] = None
    uncertainty_hours: Optional[float] = None

    def __post_init__(self) -> None:
        # Default missing Master Prompt fields if not explicitly passed
        rem_h = float(self.remaining_hours if self.remaining_hours is not None else self.tte_hours_central)
        object.__setattr__(self, "remaining_hours", rem_h)

        if self.remaining_cycles is None:
            # Calibrated cycles per remaining hour for Rotax 914
            cycles = max(0, int(round(rem_h * 0.85)))
            object.__setattr__(self, "remaining_cycles", cycles)
        else:
            object.__setattr__(self, "remaining_cycles", int(self.remaining_cycles))

        if self.health_index is None:
            # Derived health index percentage in [0, 100]
            hi = max(0.0, min(100.0, (rem_h / 2000.0) * 100.0))
            object.__setattr__(self, "health_index", round(hi, 1))
        else:
            object.__setattr__(self, "health_index", round(float(self.health_index), 1))

        if self.uncertainty_hours is None:
            unc = max(0.0, float(self.tte_hours_upper - self.tte_hours_lower) / 2.0)
            object.__setattr__(self, "uncertainty_hours", round(unc, 1))
        else:
            object.__setattr__(self, "uncertainty_hours", round(float(self.uncertainty_hours), 1))

    @property
    def is_uncertain(self) -> bool:
        return self.status is RulStatus.RUL_UNCERTAIN

    @property
    def is_ok(self) -> bool:
        return self.status is RulStatus.RUL_OK

    @property
    def is_degraded(self) -> bool:
        return self.status is RulStatus.RUL_DEGRADED

    @property
    def is_critical(self) -> bool:
        return self.status is RulStatus.RUL_CRITICAL

    @property
    def bounds(self) -> RulBounds:
        return RulBounds(
            lower=self.tte_hours_lower,
            central=self.tte_hours_central,
            upper=self.tte_hours_upper,
        )

    def to_dict(self) -> dict:
        out: dict = {
            "time_s": float(self.time_s),
            "tte_hours_central": float(self.tte_hours_central),
            "tte_hours_lower": float(self.tte_hours_lower),
            "tte_hours_upper": float(self.tte_hours_upper),
            "wear_rate_per_hour": float(self.wear_rate_per_hour),
            "confidence": float(self.confidence),
            "status": self.status.value,
            "trend": self.trend.value,
            "model_status": self.model_status.value,
            "contributing_faults": dict(self.contributing_faults),
            "notes": list(self.notes),
            "remaining_hours": float(self.remaining_hours),
            "remaining_cycles": int(self.remaining_cycles),
            "health_index": float(self.health_index),
            "uncertainty_hours": float(self.uncertainty_hours),
        }
        return out


# ---------------------------------------------------------------------
# Residual trend input (Digital Twin residual history snapshot)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualTrendInput:
    """A snapshot of the Digital Twin residual history.

    Carries the per-channel z-score history as a small
    ``(n_channels, n_ticks)`` array. The RUL feature extractor
    turns this into the residual-trend feature columns
    (``residual.max_abs_z.*`` and ``residual.abs_z_trend_per_hour``)
    so the wear-rate model can see whether residuals are
    growing — the spec's "Digital Twin residual trends" input.

    Parameters
    ----------
    channel_names
        Ordered tuple of channel names (length ``n_channels``).
    z_history
        ``(n_channels, n_ticks)`` array of z-scores. NaN/inf are
        treated as zero.
    dt_s
        Tick interval in seconds; used to convert the per-tick
        slope into a per-hour rate.
    """

    channel_names: Tuple[str, ...]
    z_history: np.ndarray
    dt_s: float = 0.1

    def __post_init__(self) -> None:
        z = np.asarray(self.z_history, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(
                f"z_history must be 2-D (n_channels, n_ticks), "
                f"got shape {z.shape}"
            )
        if z.shape[0] != len(self.channel_names):
            raise ValueError(
                f"z_history rows ({z.shape[0]}) must match "
                f"channel_names length ({len(self.channel_names)})"
            )
        # NaN/inf → 0. We don't want a single bad residual to
        # blow up the wear-rate model.
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "z_history", z.astype(np.float64))
        if float(self.dt_s) <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")

    @property
    def n_channels(self) -> int:
        return int(self.z_history.shape[0])

    @property
    def n_ticks(self) -> int:
        return int(self.z_history.shape[1])

    @property
    def abs_z(self) -> np.ndarray:
        """``(n_channels, n_ticks)`` of |z|."""
        return np.abs(self.z_history)

    def per_channel_max_abs_z(self) -> np.ndarray:
        """``(n_channels,)`` max over the time axis.

        Returns an array of zeros when there are no ticks (a
        zero-size array would otherwise raise on ``max``).
        """
        if self.n_ticks == 0:
            return np.zeros(self.n_channels, dtype=np.float64)
        return self.abs_z.max(axis=1)

    def per_channel_mean_abs_z(self) -> np.ndarray:
        """``(n_channels,)`` mean over the time axis."""
        if self.n_ticks == 0:
            return np.zeros(self.n_channels, dtype=np.float64)
        return self.abs_z.mean(axis=1)

    def per_channel_trend(self) -> np.ndarray:
        """``(n_channels,)`` slope of mean |z| per second.

        Computed as a simple least-squares slope on the mean
        |z| time series. Returns 0.0 for channels with < 2 ticks.
        """
        out = np.zeros(self.n_channels, dtype=np.float64)
        if self.n_ticks < 2:
            return out
        # Per-tick mean |z| over channels; one value per tick.
        per_tick = self.abs_z.mean(axis=0)
        t = np.arange(self.n_ticks, dtype=np.float64) * float(self.dt_s)
        # Slope = cov(t, y) / var(t).
        t_mean = t.mean()
        y_mean = per_tick.mean()
        denom = float(((t - t_mean) ** 2).sum())
        if denom <= 0.0:
            return out
        slope = float(((t - t_mean) * (per_tick - y_mean)).sum()) / denom
        # Broadcast the per-tick trend to per-channel. We don't
        # have per-channel slope computation here (that's a future
        # refinement); the spec asks for a single global trend.
        out[:] = slope
        return out


__all__ = [
    "ResidualTrendInput",
    "RulBounds",
    "RulEstimate",
    "RulModelStatus",
    "RulStatus",
    "RulTrend",
    "TREND_EPSILON_HOURS",
    "TREND_WINDOW",
    "status_for",
    "trend_for",
]
