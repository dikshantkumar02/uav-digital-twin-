"""
Bilinear interpolation of 2D performance maps.

Performance maps are stored as a 2D table on a (rpm_axis, map_axis) grid,
as defined in ``config/engine.yaml``. Real-time evaluation is bilinear
in (RPM, MAP). The implementation is pure NumPy and tolerant of queries
outside the table — they are clamped to the nearest axis endpoint.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def interp_axis(axis: Sequence[float], value: float) -> tuple[int, int, float]:
    """Find bracketing indices for ``value`` in a monotonically increasing
    ``axis``. Returns (i_lo, i_hi, u) where ``u in [0, 1]`` is the local
    fraction. Clamps ``value`` to the axis range.
    """
    n = len(axis)
    if n < 2:
        raise ValueError("axis must have at least 2 points")
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return n - 1, n - 1, 0.0
    # Binary search would be faster, but linear scan is fine for n <= 32.
    for i in range(n - 1):
        a, b = axis[i], axis[i + 1]
        if a <= value <= b:
            u = 0.0 if b == a else (value - a) / (b - a)
            return i, i + 1, u
    # Defensive: unreachable given the clamps above.
    return n - 1, n - 1, 0.0


def bilinear_interp(
    rpm_axis: Sequence[float],
    map_axis: Sequence[float],
    values: Sequence[Sequence[float]],
    rpm: float,
    map_: float,
) -> float:
    """Bilinear interpolation of ``values[rpm_idx, map_idx]`` at (rpm, map_).

    Both ``rpm`` and ``map_`` are clamped to their respective axes.
    """
    i_lo, i_hi, u = interp_axis(rpm_axis, rpm)
    j_lo, j_hi, v = interp_axis(map_axis, map_)

    v00 = values[i_lo][j_lo]
    v01 = values[i_lo][j_hi]
    v10 = values[i_hi][j_lo]
    v11 = values[i_hi][j_hi]

    return (
        (1.0 - u) * (1.0 - v) * v00
        + (1.0 - u) * v * v01
        + u * (1.0 - v) * v10
        + u * v * v11
    )


def map_evaluator(
    rpm_axis: Sequence[float],
    map_axis: Sequence[float],
    values: Sequence[Sequence[float]],
) -> Callable[[float, float], float]:
    """Return a closure that evaluates a 2D table for a given (rpm, map_)."""
    rows = [list(r) for r in values]
    return lambda rpm, map_: bilinear_interp(rpm_axis, map_axis, rows, rpm, map_)


__all__ = ["bilinear_interp", "interp_axis", "map_evaluator"]
