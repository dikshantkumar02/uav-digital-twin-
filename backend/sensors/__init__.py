"""
Sensors package — virtual sensor model.

Public re-exports::

    from backend.sensors import (
        SensorBundle, SensorReading, SensorSample, SENSOR_CHANNELS,
    )
"""

from .bundle import SensorBundle, SensorReading, SensorSample
from .channels import Channel, SENSOR_CHANNELS, _true_value, build_channel
from .noise import (
    NoiseMode,
    apply_bias,
    apply_calibration_error,
    apply_dropout,
    apply_drift,
    apply_gaussian_noise,
    apply_spike,
    apply_stuck,
)

__all__ = [
    "Channel",
    "NoiseMode",
    "SENSOR_CHANNELS",
    "SensorBundle",
    "SensorReading",
    "SensorSample",
    "_true_value",
    "apply_bias",
    "apply_calibration_error",
    "apply_dropout",
    "apply_drift",
    "apply_gaussian_noise",
    "apply_spike",
    "apply_stuck",
    "build_channel",
]
