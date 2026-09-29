"""
RUL evaluation metrics (PHASE 11 extension).

Pure-numpy implementation of the standard regression metrics used
to compare the closed-form baseline against the trained Random
Forest:

* **MAE** — mean absolute error in hours.
* **RMSE** — root mean squared error in hours.
* **MAPE** — mean absolute percentage error. Returned as
  ``Optional[float]``; the function returns ``None`` whenever any
  ground-truth value is zero (MAPE is undefined there, and
  reporting it would be the "fabricate precision" anti-pattern
  the spec calls out).
* **PICP** — prediction-interval coverage probability: the
  fraction of samples whose ground truth lies within the
  reported ``[lower, upper]`` interval. The target coverage
  for the default 5 / 95 % interval is 0.90; anything below
  that flags the bounds as under-confident.

The entry point is :func:`evaluate_rul`, which takes parallel
arrays of truth / central / lower / upper values and returns a
:class:`RulEvalReport`. The function is pure — no I/O, no
RNG, no side-effects — and cheap to call from a test.

For the closed-form vs ML comparison, the dashboard uses this
module's metrics as input to
:func:`~backend.rul.selector.select_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# Default PICP target for the 5 / 95 % interval. PICP < target
# → bounds are under-confident; PICP > target → over-confident.
DEFAULT_PICP_TARGET: float = 0.90


@dataclass(frozen=True)
class RulEvalReport:
    """Aggregate metrics for one RUL evaluation run.

    Attributes
    ----------
    n_samples
        Number of (truth, prediction) pairs.
    mae
        Mean absolute error (hours).
    rmse
        Root mean squared error (hours).
    mape
        Mean absolute percentage error, expressed as a fraction
        (e.g. 0.10 = 10 %). ``None`` if any truth value is 0 —
        MAPE is undefined at zero, and we do not fabricate
        precision.
    picp
        Prediction-interval coverage probability: fraction of
        samples whose truth lies in ``[lower, upper]``.
    picp_target
        Nominal target coverage (default 0.90 for the 5 / 95
        interval).
    interval_width_median
        Median of ``(upper - lower)`` across samples.
    bias
        ``mean(prediction - truth)``. Positive ⇒ the model
        over-estimates RUL on average.
    notes
        Human-readable notes, e.g. "MAPE skipped: 3 zero truths".
    """

    n_samples: int
    mae: float
    rmse: float
    mape: Optional[float]
    picp: float
    picp_target: float
    interval_width_median: float
    bias: float
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "n_samples": int(self.n_samples),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "mape": None if self.mape is None else float(self.mape),
            "picp": float(self.picp),
            "picp_target": float(self.picp_target),
            "interval_width_median": float(self.interval_width_median),
            "bias": float(self.bias),
            "notes": list(self.notes),
        }


def _to_1d(arr: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def mae(truth: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error in the same units as ``truth``/``pred``."""
    t = _to_1d(truth, "truth")
    p = _to_1d(pred, "pred")
    if t.size == 0:
        return 0.0
    return float(np.mean(np.abs(p - t)))


def rmse(truth: Sequence[float], pred: Sequence[float]) -> float:
    """Root mean squared error in the same units as ``truth``/``pred``."""
    t = _to_1d(truth, "truth")
    p = _to_1d(pred, "pred")
    if t.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mape(
    truth: Sequence[float],
    pred: Sequence[float],
    *,
    eps: float = 1e-9,
) -> Optional[float]:
    """Mean absolute percentage error. ``None`` if any truth is ~0.

    ``eps`` (default 1e-9) is a floor to avoid divide-by-zero
    on numerical noise. Any truth value with ``|truth| < eps``
    is treated as zero → MAPE returns ``None``.
    """
    t = _to_1d(truth, "truth")
    p = _to_1d(pred, "pred")
    if t.size == 0:
        return None
    if np.any(np.abs(t) < eps):
        return None
    return float(np.mean(np.abs((p - t) / t)))


def picp(
    truth: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Prediction-interval coverage probability.

    Returns the fraction of samples whose ``truth`` value lies
    in the closed interval ``[lower, upper]``. ``0.0`` if there
    are no samples.
    """
    t = _to_1d(truth, "truth")
    lo = _to_1d(lower, "lower")
    hi = _to_1d(upper, "upper")
    if t.size == 0:
        return 0.0
    inside = (t >= lo) & (t <= hi)
    return float(np.mean(inside.astype(np.float64)))


def evaluate_rul(
    truth: Sequence[float],
    central: Sequence[float],
    *,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    picp_target: float = DEFAULT_PICP_TARGET,
) -> RulEvalReport:
    """Compute the standard RUL evaluation report.

    Parameters
    ----------
    truth
        Ground-truth RUL values (hours).
    central
        Predicted central RUL values (hours).
    lower, upper
        Optional predicted interval bounds. When ``None``, the
        report's ``picp`` and ``interval_width_median`` are
        reported as ``0.0``.
    picp_target
        Nominal target coverage. Stored in the report for the
        consumer to interpret ``picp`` against.
    """
    t = _to_1d(truth, "truth")
    c = _to_1d(central, "central")
    n = int(t.size)
    notes: list[str] = []
    if n == 0:
        return RulEvalReport(
            n_samples=0, mae=0.0, rmse=0.0, mape=None, picp=0.0,
            picp_target=float(picp_target), interval_width_median=0.0,
            bias=0.0, notes=("no samples",),
        )
    if c.shape[0] != n:
        raise ValueError(
            f"central length ({c.shape[0]}) must match truth length ({n})"
        )
    if lower is None or upper is None:
        p = 0.0
        iw = 0.0
        notes.append("no bounds provided; PICP and interval width reported as 0")
    else:
        lo = _to_1d(lower, "lower")
        hi = _to_1d(upper, "upper")
        if lo.shape[0] != n or hi.shape[0] != n:
            raise ValueError(
                f"lower/upper length must match truth length ({n})"
            )
        p = picp(t, lo, hi)
        iw = float(np.median(hi - lo))
        if p < float(picp_target) - 0.05:
            notes.append(
                f"PICP {p:.2f} below target {picp_target:.2f} "
                f"(bounds under-confident)"
            )
        elif p > float(picp_target) + 0.05:
            notes.append(
                f"PICP {p:.2f} above target {picp_target:.2f} "
                f"(bounds over-confident)"
            )
    m = mape(t, c)
    if m is None:
        notes.append("MAPE skipped: ground truth contains zero values")
    return RulEvalReport(
        n_samples=n,
        mae=mae(t, c),
        rmse=rmse(t, c),
        mape=m,
        picp=float(p),
        picp_target=float(picp_target),
        interval_width_median=float(iw),
        bias=float(np.mean(c - t)),
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_PICP_TARGET",
    "RulEvalReport",
    "evaluate_rul",
    "mae",
    "mape",
    "picp",
    "rmse",
]
