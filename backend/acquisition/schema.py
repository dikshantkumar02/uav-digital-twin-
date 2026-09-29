"""
Common telemetry schema (PHASE 18 — embedded telemetry acquisition).

A :class:`TelemetrySample` is the **wire format** the user spec
mandates for every MCU-emitted sample. The format is shared by
the simulated source, the serial source, and the CAN source, so
the digital twin and AI see the same data structure regardless
of where the bits came from.

The schema is intentionally simple and stable:

* a 16-field sensor payload (timestamp + 15 channels) matching
  the user's JSON example;
* a :class:`SensorHealth` bitfield with the flags the user's
  spec requires (``STALE``, ``OUT_OF_RANGE``,
  ``CALIBRATION_DUE``, ``CRC_ERROR``, plus ``SENSOR_FAULT`` for
  stuck / dropout / drift);
* a frozen dataclass with a deterministic ``to_dict`` /
  ``from_dict`` round-trip so it can be serialised on the bus
  and re-hydrated by a downstream consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------
TELEMETRY_SCHEMA_VERSION = "phase18-acq-1.0.0"


# ---------------------------------------------------------------------
# Field list — the canonical ordering used by the wire format
# ---------------------------------------------------------------------
TELEMETRY_FIELDS: tuple[str, ...] = (
    "timestamp",        # MCU wall time, ms since boot
    "rpm",              # engine speed
    "egt",              # exhaust-gas temperature, °C
    "cht",              # cylinder-head temperature, °C
    "oil_pressure",     # psi
    "oil_temperature",  # °C
    "fuel_flow",        # lph
    "vibration",        # RMS g
    "accel_x",          # m/s²
    "accel_y",          # m/s²
    "accel_z",          # m/s²
    "altitude",         # m
    "airspeed",         # m/s
    "throttle",         # % (0..100)
    "ambient_temperature",  # °C
    "ambient_pressure",     # Pa
)


# ---------------------------------------------------------------------
# Sensor health — bitfield
# ---------------------------------------------------------------------
class SensorHealth(IntFlag):
    """Bitfield of per-sample sensor health flags.

    Composable — multiple conditions can be OR'd together::

        h = SensorHealth.STALE | SensorHealth.CALIBRATION_DUE

    ``OK`` is the value 0 (the identity for ``|``) so a sample
    with no flags set is implicitly "healthy".
    """

    OK = 0
    STALE = 1                  # the bus is delivering late / cached data
    OUT_OF_RANGE = 2           # a channel value is outside its valid range
    CALIBRATION_DUE = 4        # the channel's calibration is past its due date
    CRC_ERROR = 8              # a serial frame failed CRC verification
    SENSOR_FAULT = 16          # the underlying sensor is faulted
                               # (stuck / dropout / drift)

    def is_ok(self) -> bool:
        return self == SensorHealth.OK

    def has(self, flag: "SensorHealth") -> bool:
        return (int(self) & int(flag)) == int(flag)


# ---------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetrySample:
    """One MCU-emitted sample in the common wire format.

    All sensor values are SI / engineering units as documented
    in :data:`TELEMETRY_FIELDS`. ``timestamp_ms`` is the MCU
    wall time (milliseconds since boot); the receiving edge
    uses its own wall-time at reception for latency
    measurement.

    The dataclass is frozen so it is hashable and safe to share
    across threads. ``to_dict`` / ``from_dict`` are the wire
    format — the JSON keys match :data:`TELEMETRY_FIELDS` for
    the sensor payload plus ``sensor_health`` (decimal bitfield),
    ``mission_id``, ``vehicle_id``, ``engine_id``, and
    ``model_version`` metadata.
    """

    timestamp_ms: int
    rpm: float
    egt: float
    cht: float
    oil_pressure: float
    oil_temperature: float
    fuel_flow: float
    vibration: float
    accel_x: float
    accel_y: float
    accel_z: float
    altitude: float
    airspeed: float
    throttle: float
    ambient_temperature: float
    ambient_pressure: float
    sensor_health: SensorHealth = SensorHealth.OK
    mission_id: str = "default-mission"
    vehicle_id: str = "default-vehicle"
    engine_id: str = "default-engine"
    model_version: str = TELEMETRY_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict matching the user spec."""
        return {
            "timestamp": int(self.timestamp_ms),
            "rpm": float(self.rpm),
            "egt": float(self.egt),
            "cht": float(self.cht),
            "oil_pressure": float(self.oil_pressure),
            "oil_temperature": float(self.oil_temperature),
            "fuel_flow": float(self.fuel_flow),
            "vibration": float(self.vibration),
            "accel_x": float(self.accel_x),
            "accel_y": float(self.accel_y),
            "accel_z": float(self.accel_z),
            "altitude": float(self.altitude),
            "airspeed": float(self.airspeed),
            "throttle": float(self.throttle),
            "ambient_temperature": float(self.ambient_temperature),
            "ambient_pressure": float(self.ambient_pressure),
            "sensor_health": int(self.sensor_health),
            "mission_id": self.mission_id,
            "vehicle_id": self.vehicle_id,
            "engine_id": self.engine_id,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetrySample":
        """Inverse of :meth:`to_dict` — round-trip safe."""
        return cls(
            timestamp_ms=int(d["timestamp"]),
            rpm=float(d["rpm"]),
            egt=float(d["egt"]),
            cht=float(d["cht"]),
            oil_pressure=float(d["oil_pressure"]),
            oil_temperature=float(d["oil_temperature"]),
            fuel_flow=float(d["fuel_flow"]),
            vibration=float(d["vibration"]),
            accel_x=float(d["accel_x"]),
            accel_y=float(d["accel_y"]),
            accel_z=float(d["accel_z"]),
            altitude=float(d["altitude"]),
            airspeed=float(d["airspeed"]),
            throttle=float(d["throttle"]),
            ambient_temperature=float(d["ambient_temperature"]),
            ambient_pressure=float(d["ambient_pressure"]),
            sensor_health=SensorHealth(int(d.get("sensor_health", 0))),
            mission_id=str(d.get("mission_id", "default-mission")),
            vehicle_id=str(d.get("vehicle_id", "default-vehicle")),
            engine_id=str(d.get("engine_id", "default-engine")),
            model_version=str(
                d.get("model_version", TELEMETRY_SCHEMA_VERSION)
            ),
        )


__all__ = [
    "TELEMETRY_FIELDS",
    "TELEMETRY_SCHEMA_VERSION",
    "SensorHealth",
    "TelemetrySample",
]
