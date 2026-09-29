"""
Residual scoring (PHASE 8).

Converts a :class:`~backend.digital_twin.ResidualFrame` into a
per-channel score in [0, 1] using the channel's z-score.

A single ``|z|`` spike shouldn't be enough to flip the detector —
we therefore expose an **EWMA smoother** here and let the
:class:`~backend.diagnostics.detector.AnomalyDetector` own the
state. The pure function :func:`residual_score` returns a *raw*
score; the smoother is applied by the detector.
"""

from __future__ import annotations

from typing import Dict, Optional

from backend.digital_twin import ResidualFrame


def raw_residual_score(z_score: Optional[float]) -> Optional[float]:
    """Map a z-score to a [0, 1] score.

    ``|z| = 0`` → 0.0 (perfectly normal), ``|z| >= 6`` → 1.0
    (saturated anomaly). Linear in between. ``None`` (channel
    missing) → ``None``.
    """
    if z_score is None:
        return None
    v = abs(float(z_score)) / 6.0
    return max(0.0, min(1.0, v))


def residual_scores(residual_frame: ResidualFrame) -> Dict[str, Optional[float]]:
    """Return the raw per-channel score dict for a residual frame.

    Channels missing from the frame (no observation yet) are mapped
    to ``None`` so the detector can distinguish "no evidence" from
    "evidence of normality."
    """
    out: Dict[str, Optional[float]] = {}
    for ch, r in residual_frame.residuals.items():
        out[ch] = raw_residual_score(r.z_score)
    return out


def residual_confidences(residual_frame: ResidualFrame) -> Dict[str, float]:
    """Return the per-channel confidence from the residual frame."""
    return {ch: float(r.confidence) for ch, r in residual_frame.residuals.items()}


class EwmaSmoother:
    """Per-channel exponential weighted moving average.

    The smoother is *stateful* — owned by the detector — so it can
    survive across ticks but be reset between missions. The default
    ``alpha`` of 0.20 means a single 4σ spike is smoothed into
    roughly a 0.2-spike over ~5 ticks.
    """

    def __init__(self, alpha: float = 0.20) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._state: Dict[str, float] = {}

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        self._state.clear()

    def update(self, channel: str, value: Optional[float]) -> Optional[float]:
        """Update the smoother for one channel and return the new value.

        A ``None`` value leaves the smoother untouched and returns
        the existing smoothed value (or ``None`` if the channel
        has never been seen).
        """
        if value is None:
            return self._state.get(channel)
        prev = self._state.get(channel)
        if prev is None:
            self._state[channel] = float(value)
        else:
            self._state[channel] = (1.0 - self._alpha) * prev + self._alpha * float(value)
        return self._state[channel]

    def state(self) -> Dict[str, float]:
        return dict(self._state)


__all__ = [
    "EwmaSmoother",
    "raw_residual_score",
    "residual_confidences",
    "residual_scores",
]
