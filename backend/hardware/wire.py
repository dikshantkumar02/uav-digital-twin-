"""
Wire-level types for the PHASE 14 hardware interface.

The codec (see :mod:`backend.hardware.protocol`) translates between
:class:`backend.telemetry.TelemetryFrame` and a one-line ASCII
representation suitable for serial/UART. This module holds the
small intermediate types and the canonical name/code tables.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from backend.sensors import NoiseMode, SENSOR_CHANNELS


# ---------------------------------------------------------------------
# NoiseMode wire codes
# ---------------------------------------------------------------------
# One ASCII character per NoiseMode value. Picked to be unique and
# to mirror the existing channel-mode naming where possible.
NOISE_MODE_CODE: Dict[NoiseMode, str] = {
    NoiseMode.NORMAL: "N",
    NoiseMode.DRIFTING: "D",
    NoiseMode.STUCK: "S",
    NoiseMode.SPIKE: "K",   # K = "spike" (avoid 'S' clash with STUCK)
    NoiseMode.DROPPED: "X", # X = "absent"
    NoiseMode.FAULT: "F",
}
NOISE_MODE_FROM_CODE: Dict[str, NoiseMode] = {v: k for k, v in NOISE_MODE_CODE.items()}

# Public alias.
NoiseModeCode = str


# ---------------------------------------------------------------------
# Channel name set (for fast validation in the decoder)
# ---------------------------------------------------------------------
SENSOR_NAME_SET: FrozenSet[str] = frozenset(SENSOR_CHANNELS)


__all__ = [
    "NOISE_MODE_CODE",
    "NOISE_MODE_FROM_CODE",
    "NoiseModeCode",
    "SENSOR_NAME_SET",
]
