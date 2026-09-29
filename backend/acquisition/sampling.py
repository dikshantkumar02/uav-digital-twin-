"""
Per-channel sampling configuration (PHASE 18 — embedded telemetry
acquisition).

A :class:`ChannelSampling` row tells the acquisition layer the
MCU's native ADC rate and the rate at which samples are emitted
onto the bus. The decimation factor (``bus_rate / native_rate``)
is the integer number of native samples averaged / dropped
between bus-level samples.

The default :data:`DEFAULT_SAMPLING` table is sized for a small
aero piston UAV:

* vibration and IMU are high-rate (1–2 kHz) on the MCU but
  decimated to 100–200 Hz on the bus;
* EGT / CHT / oil are slow-changing (10 Hz on the bus is enough);
* ambient air is updated once per second.

The default table is the **starting point** for a real
deployment; the per-channel rates are operator-tunable through
``config/acquisition.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------
# Per-channel sampling
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelSampling:
    """Per-channel decimation config.

    Attributes
    ----------
    channel:
        Sensor channel name (matches :data:`TELEMETRY_FIELDS`).
    native_rate_hz:
        The MCU's native ADC / sample rate for this channel.
    bus_rate_hz:
        The rate at which samples appear on the bus. Must be
        ``<= native_rate_hz``.
    decimation:
        Integer count of native samples aggregated per bus
        sample. ``1`` = no decimation, every native sample
        goes out. ``2`` = every other, ``10`` = the
        common 100 Hz → 10 Hz case.
    """

    channel: str
    native_rate_hz: float
    bus_rate_hz: float
    decimation: int = 1

    def __post_init__(self) -> None:
        if self.bus_rate_hz <= 0.0:
            raise ValueError(
                f"bus_rate_hz must be > 0, got {self.bus_rate_hz}"
            )
        if self.native_rate_hz < self.bus_rate_hz:
            raise ValueError(
                f"native_rate_hz ({self.native_rate_hz}) must be "
                f">= bus_rate_hz ({self.bus_rate_hz}) for {self.channel}"
            )
        if self.decimation < 1:
            raise ValueError(
                f"decimation must be >= 1, got {self.decimation}"
            )


# ---------------------------------------------------------------------
# Default table
# ---------------------------------------------------------------------
def _row(channel: str, native: float, bus: float) -> ChannelSampling:
    """Build a :class:`ChannelSampling` from native and bus rates."""
    decimation = max(1, int(round(native / bus)))
    return ChannelSampling(
        channel=channel,
        native_rate_hz=float(native),
        bus_rate_hz=float(bus),
        decimation=decimation,
    )


DEFAULT_SAMPLING: Dict[str, ChannelSampling] = {
    "rpm": _row("rpm", 1000.0, 100.0),
    "egt": _row("egt", 100.0, 10.0),
    "cht": _row("cht", 100.0, 10.0),
    "oil_pressure": _row("oil_pressure", 100.0, 10.0),
    "oil_temperature": _row("oil_temperature", 100.0, 10.0),
    "fuel_flow": _row("fuel_flow", 100.0, 10.0),
    "vibration": _row("vibration", 2000.0, 200.0),
    "accel_x": _row("accel_x", 1000.0, 100.0),
    "accel_y": _row("accel_y", 1000.0, 100.0),
    "accel_z": _row("accel_z", 1000.0, 100.0),
    "altitude": _row("altitude", 50.0, 10.0),
    "airspeed": _row("airspeed", 50.0, 10.0),
    "throttle": _row("throttle", 1000.0, 100.0),
    "ambient_temperature": _row("ambient_temperature", 10.0, 1.0),
    "ambient_pressure": _row("ambient_pressure", 10.0, 1.0),
}


def get_sampling(
    channel: str,
    fallback: Optional[ChannelSampling] = None,
) -> ChannelSampling:
    """Look up the sampling row for ``channel``.

    If the channel has no entry in :data:`DEFAULT_SAMPLING`, the
    optional ``fallback`` is returned (defaulting to a 1:1
    passthrough at 1 Hz).
    """
    if channel in DEFAULT_SAMPLING:
        return DEFAULT_SAMPLING[channel]
    if fallback is not None:
        return fallback
    return ChannelSampling(
        channel=channel,
        native_rate_hz=1.0,
        bus_rate_hz=1.0,
        decimation=1,
    )


__all__ = ["ChannelSampling", "DEFAULT_SAMPLING", "get_sampling"]
