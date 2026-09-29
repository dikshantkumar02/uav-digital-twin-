"""
Simulated acquisition source (PHASE 18 — embedded telemetry
acquisition).

:class:`SimulatedAcquisition` is the **default** source — the
one used when no real hardware is configured. It wraps the
existing :class:`~backend.sensors.SensorBundle` and emits
:class:`TelemetrySample` objects in the same wire format a
physical MCU would produce.

The IMU is split into ``accel_x`` / ``accel_y`` / ``accel_z``
by extracting three orthogonal components from the
environment's vertical acceleration + the engine vibration
(deterministic, no extra noise). ``throttle`` is read from
``EngineState.throttle`` and scaled to the user-spec 0..100
percent range.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, List, Optional

from backend.sensors import SensorBundle, SensorSample
from backend.sensors.noise import NoiseMode

from .schema import SensorHealth, TelemetrySample


# ---------------------------------------------------------------------
# Lightweight per-channel plausibility bounds
# ---------------------------------------------------------------------
# These are intentionally generous — they're for flagging
# *obviously* bogus values (e.g. negative RPM, EGT of -300 °C).
# They are NOT thresholds for fault decisions; the PHASE 8
# anomaly detector handles that.
DEFAULT_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "rpm":               (-10.0,  10_000.0),
    "egt":               (-50.0,  1_200.0),
    "cht":               (-50.0,    400.0),
    "oil_pressure":      (-10.0,    200.0),
    "oil_temperature":   (-50.0,    250.0),
    "fuel_flow":         (-10.0,    200.0),
    "vibration":         (-5.0,     100.0),
    "altitude":          (-500.0,  10_000.0),
    "airspeed":          (-10.0,    200.0),
    "throttle":          (-5.0,     105.0),
    "ambient_temperature": (-100.0,  70.0),
    "ambient_pressure":  (50_000.0, 110_000.0),
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SimulatedAcquisitionConfig:
    """Knobs for :class:`SimulatedAcquisition`.

    ``now_ms`` is the wall time the simulator pretends the
    MCU's millisecond counter is at. ``mission_id`` /
    ``vehicle_id`` / ``engine_id`` are written into every
    sample.
    """

    mission_id: str = "simulated-mission"
    vehicle_id: str = "simulated-vehicle"
    engine_id: str = "simulated-engine"
    now_ms: int = 0
    # The MCU wall-time offset used to start ``now_ms``. When
    # the first sample is emitted, ``now_ms`` is replaced by
    # the actual perf_counter reading cast to milliseconds.
    use_wall_clock: bool = True


# ---------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------
class SimulatedAcquisition:
    """Default :class:`~.sensor.SensorSource` — no hardware required.

    Wraps a :class:`~backend.sensors.SensorBundle` and produces
    :class:`TelemetrySample` objects in the common wire format.
    The IMU is split into three orthogonal axes (``accel_x`` /
    ``accel_y`` / ``accel_z``); throttle is read from
    ``EngineState.throttle`` (0..1) and rescaled to 0..100 %.
    """

    def __init__(
        self,
        bundle: SensorBundle,
        config: Optional[SimulatedAcquisitionConfig] = None,
    ) -> None:
        self._bundle = bundle
        self._cfg = config or SimulatedAcquisitionConfig()
        # Wall-clock anchor — every sample's ``timestamp_ms`` is
        # the milliseconds elapsed since construction. Set
        # eagerly (not lazily) so the first sample has a
        # non-zero timestamp and downstream code that uses it
        # (e.g. ``is_due`` checks) works as expected.
        self._start_perf: float = _time.perf_counter()

    # ------------------------------------------------------------------
    # SensorSource protocol
    # ------------------------------------------------------------------
    @property
    def source_kind(self) -> str:
        return "simulated"

    @property
    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Step + read
    # ------------------------------------------------------------------
    def step(
        self,
        degradation_severity: float = 0.0,
        vibration_external: float = 0.0,
    ) -> SensorSample:
        """Advance the wrapped simulator by one step."""
        return self._bundle.tick(
            degradation_severity=degradation_severity,
            vibration_external=vibration_external,
        )

    def read(
        self,
        timeout_s: float = 0.0,
    ) -> Optional[TelemetrySample]:
        """Read the next sample.

        The call to :meth:`step` advances the underlying
        :class:`SensorBundle` and returns the corresponding
        :class:`TelemetrySample` in the common wire format.

        ``timeout_s`` is accepted for protocol compatibility
        with hardware-backed sources; the simulator returns
        immediately.
        """
        sensor_sample = self.step()
        return self._convert(sensor_sample)

    # ------------------------------------------------------------------
    # Wire-format conversion
    # ------------------------------------------------------------------
    def _convert(self, s: SensorSample) -> TelemetrySample:
        r = s.readings
        engine = s.engine
        env = s.env
        if engine is None or env is None:
            raise RuntimeError(
                "SimulatedAcquisition requires the underlying "
                "SensorBundle to carry engine + env state. Call "
                "step() (not run()) so tick() is used."
            )

        # Throttle: env.throttle is [0,1]; spec wants %.
        throttle_pct = float(env.throttle) * 100.0

        # IMU: the underlying simulator exposes a single vertical
        # axis (env.vertical_accel_mps2). Split into three
        # orthogonal components, with the X and Y axes derived
        # from engine vibration so the axes are correlated but
        # distinguishable. The split is deterministic — no RNG —
        # so the same engine state yields the same IMU every
        # time.
        ax = float(env.vertical_accel_mps2)
        ay = float(engine.vibration_rms_g) * 0.1
        az = float(engine.vibration_rms_g) * 0.07

        # Timestamp: prefer ``self._cfg.now_ms`` if the user
        # pinned it; otherwise use the wall clock delta from
        # when this source was constructed.
        if self._cfg.use_wall_clock:
            now_ms = int((_time.perf_counter() - self._start_perf) * 1000.0)
        else:
            now_ms = int(self._cfg.now_ms)

        # Sensor health: any dropout / stuck / fault flag rolls
        # up to ``SENSOR_FAULT``. ``OUT_OF_RANGE`` is added for
        # any channel whose value escapes its plausible range.
        health = SensorHealth.OK
        for ch_name, (lo, hi) in DEFAULT_PLAUSIBLE_RANGES.items():
            reading = r.get(ch_name)
            if reading is None or reading.value is None:
                continue
            v = float(reading.value)
            if v < lo or v > hi:
                health = health | SensorHealth.OUT_OF_RANGE
        for ch_name, reading in r.items():
            if reading is None:
                continue
            if reading.mode in (
                NoiseMode.STUCK,
                NoiseMode.DROPPED,
                NoiseMode.FAULT,
                NoiseMode.DRIFTING,
            ):
                health = health | SensorHealth.SENSOR_FAULT

        # Build the spec-defined 16-field wire payload.
        def _g(name: str) -> float:
            reading = r.get(name)
            if reading is None or reading.value is None:
                return 0.0
            return float(reading.value)

        return TelemetrySample(
            timestamp_ms=now_ms,
            rpm=_g("rpm"),
            egt=_g("egt"),
            cht=_g("cht"),
            oil_pressure=_g("oil_pressure"),
            oil_temperature=_g("oil_temperature"),
            fuel_flow=_g("fuel_flow"),
            vibration=_g("vibration"),
            accel_x=ax,
            accel_y=ay,
            accel_z=az,
            altitude=_g("altitude"),
            airspeed=_g("airspeed"),
            throttle=throttle_pct,
            ambient_temperature=_g("ambient_temperature"),
            ambient_pressure=_g("ambient_pressure"),
            sensor_health=health,
            mission_id=self._cfg.mission_id,
            vehicle_id=self._cfg.vehicle_id,
            engine_id=self._cfg.engine_id,
        )


__all__ = [
    "SimulatedAcquisition",
    "SimulatedAcquisitionConfig",
    "DEFAULT_PLAUSIBLE_RANGES",
]
