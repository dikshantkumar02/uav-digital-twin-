"""
Sensor abstraction layer (PHASE 18 — embedded telemetry acquisition).

A :class:`SensorAbstraction` is the user's spec's
"sensor abstraction layer". It wraps:

* a :class:`SensorSource` (simulated / serial / CAN) that
  produces raw :class:`TelemetrySample` objects;
* a :class:`CalibrationRegistry` that adjusts each channel's
  value through gain / offset / polynomial;
* a sampling configuration that tags each sample with the
  right per-channel decimation factor;
* a :class:`SensorHealth` aggregator that picks up
  ``STALE``, ``OUT_OF_RANGE``, and ``CALIBRATION_DUE`` based on
  the calibration registry and the per-channel plausible
  ranges.

The abstraction's :meth:`read` returns a :class:`TelemetrySample`
that's ready to hand to the PHASE 17 streaming pipeline.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .calibration import CalibrationRegistry
from .sampling import ChannelSampling, DEFAULT_SAMPLING, get_sampling
from .schema import SensorHealth, TelemetrySample
from .simulated import DEFAULT_PLAUSIBLE_RANGES


# ---------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------
@runtime_checkable
class SensorSource(Protocol):
    """Anything that produces :class:`TelemetrySample` objects.

    A :class:`SimulatedAcquisition`, a serial source wrapping
    :class:`~backend.acquisition.serial_protocol.McuFrame`,
    and a CAN source built on :class:`CanPort` are all valid
    implementations.
    """

    def read(self, timeout_s: float = 0.0) -> Optional[TelemetrySample]: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def source_kind(self) -> str: ...            # "simulated" | "serial" | "can"


# ---------------------------------------------------------------------
# Per-channel config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelLimit:
    """The per-channel plausible range used to set ``OUT_OF_RANGE``."""

    channel: str
    min_value: float
    max_value: float


DEFAULT_LIMITS = {
    ch: ChannelLimit(ch, lo, hi)
    for ch, (lo, hi) in DEFAULT_PLAUSIBLE_RANGES.items()
}


# ---------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------
class SensorAbstraction:
    """The user's spec's "sensor abstraction layer".

    Parameters
    ----------
    source:
        The :class:`SensorSource` producing raw samples.
    calibration:
        :class:`CalibrationRegistry` describing per-channel
        gain / offset / polynomial correction. Use
        :meth:`CalibrationRegistry.identity` to disable.
    sampling:
        Optional override for the per-channel sampling config.
        Defaults to :data:`DEFAULT_SAMPLING`.
    plausible_limits:
        Per-channel :class:`ChannelLimit` used to set the
        ``OUT_OF_RANGE`` flag. Defaults to the simulator's
        built-in ranges.
    """

    def __init__(
        self,
        source: SensorSource,
        calibration: Optional[CalibrationRegistry] = None,
        sampling: Optional[dict[str, ChannelSampling]] = None,
        plausible_limits: Optional[dict[str, ChannelLimit]] = None,
    ) -> None:
        self._source = source
        self._calibration = calibration or CalibrationRegistry.identity()
        self._sampling = sampling if sampling is not None else DEFAULT_SAMPLING
        self._limits = plausible_limits if plausible_limits is not None else DEFAULT_LIMITS

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def source(self) -> SensorSource:
        return self._source

    @property
    def calibration(self) -> CalibrationRegistry:
        return self._calibration

    @property
    def source_kind(self) -> str:
        return self._source.source_kind

    def is_open(self) -> bool:
        return self._source.is_open

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read(self, timeout_s: float = 0.0) -> Optional[TelemetrySample]:
        """Read one sample, apply calibration, set health flags.

        Returns ``None`` if the underlying source is closed or
        has no data ready within ``timeout_s``.
        """
        if not self._source.is_open:
            return None
        raw = self._source.read(timeout_s=timeout_s)
        if raw is None:
            return None
        return self._calibrate_and_tag(raw)

    def close(self) -> None:
        self._source.close()

    # ------------------------------------------------------------------
    # Calibration + tagging
    # ------------------------------------------------------------------
    def _calibrate_and_tag(self, sample: TelemetrySample) -> TelemetrySample:
        d = sample.to_dict()
        now_ms = int(sample.timestamp_ms)

        # 1) Apply calibration per channel.
        calibrated = dict(d)
        for ch, table in self._calibration.tables.items():
            if ch in calibrated and ch != "sensor_health":
                try:
                    calibrated[ch] = float(table.apply(float(d[ch])))
                except (TypeError, ValueError):
                    pass

        # 2) Build the new health bitfield.
        health = SensorHealth(int(d.get("sensor_health", 0)))

        # OUT_OF_RANGE: from the plausible limits.
        for ch, lim in self._limits.items():
            v = calibrated.get(ch)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if vf < float(lim.min_value) or vf > float(lim.max_value):
                health = health | SensorHealth.OUT_OF_RANGE

        # CALIBRATION_DUE: from the calibration registry.
        for ch_name in self._calibration.due_channels(now_ms):
            health = health | SensorHealth.CALIBRATION_DUE

        return TelemetrySample(
            timestamp_ms=now_ms,
            rpm=float(calibrated.get("rpm", 0.0)),
            egt=float(calibrated.get("egt", 0.0)),
            cht=float(calibrated.get("cht", 0.0)),
            oil_pressure=float(calibrated.get("oil_pressure", 0.0)),
            oil_temperature=float(calibrated.get("oil_temperature", 0.0)),
            fuel_flow=float(calibrated.get("fuel_flow", 0.0)),
            vibration=float(calibrated.get("vibration", 0.0)),
            accel_x=float(calibrated.get("accel_x", 0.0)),
            accel_y=float(calibrated.get("accel_y", 0.0)),
            accel_z=float(calibrated.get("accel_z", 0.0)),
            altitude=float(calibrated.get("altitude", 0.0)),
            airspeed=float(calibrated.get("airspeed", 0.0)),
            throttle=float(calibrated.get("throttle", 0.0)),
            ambient_temperature=float(calibrated.get("ambient_temperature", 0.0)),
            ambient_pressure=float(calibrated.get("ambient_pressure", 0.0)),
            sensor_health=health,
            mission_id=sample.mission_id,
            vehicle_id=sample.vehicle_id,
            engine_id=sample.engine_id,
        )

    # ------------------------------------------------------------------
    # Sampling introspection
    # ------------------------------------------------------------------
    def sampling_for(self, channel: str) -> ChannelSampling:
        """Return the :class:`ChannelSampling` for ``channel``."""
        return get_sampling(channel)


__all__ = ["SensorSource", "SensorAbstraction", "ChannelLimit", "DEFAULT_LIMITS"]
