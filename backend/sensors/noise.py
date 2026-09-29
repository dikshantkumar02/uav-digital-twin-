"""
Per-sample noise primitives used by the sensor channels.

Every function is pure and deterministic given its RNG, so the sensor
bundle is fully reproducible.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np


class NoiseMode(str, Enum):
    """Operating mode of a virtual sensor channel at one sample."""

    NORMAL = "NORMAL"           # nominal, just noise + bias + calibration
    DRIFTING = "DRIFTING"       # an injected drift is active
    STUCK = "STUCK"             # frozen at the configured stuck value
    SPIKE = "SPIKE"             # current sample is a spike
    DROPPED = "DROPPED"         # current sample is invalid
    FAULT = "FAULT"             # generic injected sensor fault


# ---------------------------------------------------------------------
# Individual noise primitives
# ---------------------------------------------------------------------
def apply_gaussian_noise(rng: np.random.Generator, value: float, std: float) -> float:
    if std <= 0.0:
        return float(value)
    return float(value) + float(rng.normal(0.0, std))


def apply_bias(value: float, bias: float) -> float:
    return float(value) + float(bias)


def apply_drift(value: float, accumulated_drift: float) -> float:
    return float(value) + float(accumulated_drift)


def apply_calibration_error(value: float, calibration_error: float) -> float:
    """Multiplicative calibration error: 0.01 => +1% scale."""
    return float(value) * (1.0 + float(calibration_error))


def apply_dropout(rng: np.random.Generator, value: float, prob: float) -> Optional[float]:
    if prob <= 0.0:
        return float(value)
    if float(rng.uniform(0.0, 1.0)) < prob:
        return None
    return float(value)


def apply_spike(
    rng: np.random.Generator,
    value: float,
    prob: float,
    amplitude: float,
) -> float:
    if prob <= 0.0:
        return float(value)
    if float(rng.uniform(0.0, 1.0)) < prob:
        sign = 1.0 if float(rng.uniform(0.0, 1.0)) < 0.5 else -1.0
        return float(value) + sign * float(amplitude)
    return float(value)


def apply_stuck(value: float, is_stuck: bool, stuck_value: float) -> float:
    return float(stuck_value) if is_stuck else float(value)


__all__ = [
    "NoiseMode",
    "apply_bias",
    "apply_calibration_error",
    "apply_dropout",
    "apply_drift",
    "apply_gaussian_noise",
    "apply_spike",
    "apply_stuck",
]
