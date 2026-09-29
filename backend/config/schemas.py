"""
Pydantic schemas for the configuration system.

Every numeric field on the engine maps carries a provenance label
(MEASURED / PUBLIC_DATA / PHYSICS_DERIVED / INTERPOLATED / SYNTHETIC /
USER_CONFIGURED) as required by the project specification.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


class Provenance(str, Enum):
    MEASURED = "MEASURED"
    PUBLIC_DATA = "PUBLIC_DATA"
    PHYSICS_DERIVED = "PHYSICS_DERIVED"
    INTERPOLATED = "INTERPOLATED"
    SYNTHETIC = "SYNTHETIC"
    USER_CONFIGURED = "USER_CONFIGURED"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
class EngineGeometry(_Base):
    cylinders: int = Field(gt=0, le=24)
    displacement_l: float = Field(gt=0)
    compression_ratio: float = Field(gt=1)
    bore_mm: float = Field(gt=0)
    stroke_mm: float = Field(gt=0)
    redline_rpm: float = Field(gt=0)
    idle_rpm: float = Field(gt=0)
    rated_rpm: float = Field(gt=0)
    rated_power_kw: float = Field(gt=0)


class _AxisTable(_Base):
    """2D performance map with axis metadata and a value matrix."""

    rpm_axis: list[float]
    map_axis: list[float]
    values: list[list[float]]

    @field_validator("values")
    @classmethod
    def _shape_matches_axes(cls, v: list[list[float]], info: Any) -> list[list[float]]:
        axes = info.data
        n_rpm = len(axes.get("rpm_axis", []))
        n_map = len(axes.get("map_axis", []))
        if len(v) != n_rpm:
            raise ValueError(
                f"values rows ({len(v)}) must match rpm_axis length ({n_rpm})"
            )
        for i, row in enumerate(v):
            if len(row) != n_map:
                raise ValueError(
                    f"values[{i}] has {len(row)} columns, expected {n_map}"
                )
        return v


class BsfcMap(_AxisTable):
    """Brake-specific fuel consumption map (g per kWh)."""


class EgtMap(_AxisTable):
    """Expected EGT map (deg C)."""


class ChtMap(_AxisTable):
    """Expected CHT map (deg C)."""


class ThrottleToMap(_Base):
    gain_inhg_per_unit: float = Field(gt=0)
    idle_map_inhg: float = Field(ge=0)
    max_map_inhg: float = Field(gt=0)
    dead_zone: float = Field(ge=0, le=0.5)


class EngineDynamics(_Base):
    rpm_time_constant_s: float = Field(gt=0)
    thermal_time_constant_s: float = Field(gt=0)
    oil_time_constant_s: float = Field(gt=0)


class EngineLimits(_Base):
    egt_max_c: float
    cht_max_c: float
    oil_temp_min_c: float
    oil_temp_max_c: float
    oil_pressure_min_psi: float
    oil_pressure_max_psi: float
    vibration_rms_max_g: float = Field(gt=0)


class Lubrication(_Base):
    oil_capacity_l: float = Field(gt=0)
    nominal_pressure_psi: float = Field(gt=0)
    pressure_rpm_slope: float = Field(ge=0)
    pressure_temp_drop_per_c: float = Field(ge=0)


class Degradation(_Base):
    base_wear_per_hour: float = Field(ge=0)
    wear_temp_factor_per_c: float = Field(ge=0)
    wear_rpm_factor_per_rpm: float = Field(ge=0)
    max_wear: float = Field(gt=0, le=10)


class EngineConfig(_Base):
    model_id: str
    description: str
    provenance_default: Provenance
    notes: list[str] = Field(default_factory=list)
    geometry: EngineGeometry
    performance_maps: dict[str, Any]  # validated below
    throttle_to_map: ThrottleToMap
    dynamics: EngineDynamics
    limits: EngineLimits
    lubrication: Lubrication
    degradation: Degradation

    @field_validator("performance_maps")
    @classmethod
    def _validate_maps(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "bsfc_g_per_kwh" not in v or "egt_map_c" not in v or "cht_map_c" not in v:
            raise ValueError(
                "performance_maps must contain bsfc_g_per_kwh, egt_map_c, cht_map_c"
            )
        BsfcMap.model_validate(v["bsfc_g_per_kwh"])
        EgtMap.model_validate(v["egt_map_c"])
        ChtMap.model_validate(v["cht_map_c"])
        return v


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
class SeaLevel(_Base):
    pressure_pa: float = Field(gt=0)
    temperature_k: float = Field(gt=0)
    density_kg_per_m3: float = Field(gt=0)
    speed_of_sound_mps: float = Field(gt=0)


class Atmosphere(_Base):
    model: str
    sea_level: SeaLevel
    lapse_rate_k_per_m: float = Field(ge=0)
    tropopause_m: float = Field(gt=0)


class Waypoint(_Base):
    t_s: float = Field(ge=0)
    altitude_m: float = Field(ge=0)
    airspeed_mps: float = Field(ge=0)
    throttle: float = Field(ge=0, le=1)


class Mission(_Base):
    profile: list[Waypoint] = Field(min_length=2)
    initial_seed: int


class TurbulenceCfg(_Base):
    enabled: bool
    intensity: float = Field(ge=0, le=2)
    correlation_time_s: float = Field(gt=0)
    seed: int


class GustsCfg(_Base):
    enabled: bool
    rate_per_hour: float = Field(ge=0)
    amplitude_mps: float = Field(gt=0)
    seed: int


class Disturbances(_Base):
    turbulence: TurbulenceCfg
    gusts: GustsCfg


class EnvironmentConfig(_Base):
    atmosphere: Atmosphere
    mission: Mission
    disturbances: Disturbances


# ---------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------
class SensorConfig(_Base):
    type: str
    unit: str
    sample_rate_hz: float = Field(gt=0)
    noise_std: float = Field(ge=0)
    bias: float
    drift_per_hour: float = Field(ge=0)
    dropout_prob: float = Field(ge=0, le=1)
    spike_prob: float = Field(ge=0, le=1)
    stuck_prob: float = Field(ge=0, le=1)
    stuck_value: float
    calibration_error: float = Field(ge=0, le=1)


class SensorsConfig(RootModel[dict[str, SensorConfig]]):
    """Sensors are keyed by channel name (e.g. ``rpm``, ``egt``)."""

    root: dict[str, SensorConfig] = Field(default_factory=dict)

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, key: str) -> SensorConfig:
        return self.root[key]

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()


# ---------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------
class FaultEntry(_Base):
    enabled: bool
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class FaultsConfig(RootModel[dict[str, FaultEntry]]):
    """Faults are keyed by class name (e.g. ``ENGINE_DEGRADATION``)."""

    root: dict[str, FaultEntry] = Field(default_factory=dict)

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, key: str) -> FaultEntry:
        return self.root[key]

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()


# ---------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------
class AppConfig(_Base):
    engine: EngineConfig
    environment: EnvironmentConfig
    sensors: SensorsConfig
    faults: FaultsConfig


# ---------------------------------------------------------------------
# Mission risk (PHASE 12)
# ---------------------------------------------------------------------
class RiskConfig(_Base):
    """Mission-risk-engine configuration (PHASE 12).

    All fields are operator-tunable. Weights are normalised to
    sum to 1.0 at load time, so changing individual values only
    requires keeping the sum reasonable.
    """

    provenance: Provenance = Provenance.USER_CONFIGURED
    notes: list[str] = Field(default_factory=list)

    # Per-signal weights in the deficit sum.
    weight_health: float = Field(ge=0, le=1, default=0.45)
    weight_rul: float = Field(ge=0, le=1, default=0.35)
    weight_anomaly: float = Field(ge=0, le=1, default=0.20)

    # Status / level thresholds on the final risk score [0, 1].
    threshold_caution: float = Field(ge=0, le=1, default=0.25)
    threshold_return_to_base: float = Field(ge=0, le=1, default=0.55)
    threshold_abort: float = Field(ge=0, le=1, default=0.80)

    # RUL deficit ramp parameters (hours).
    rul_deficit_floor_h: float = Field(gt=0, default=5.0)
    rul_deficit_ceiling_h: float = Field(gt=0, default=100.0)

    # Fault boost coefficient.
    fault_boost_coef: float = Field(ge=0, le=1, default=0.10)

    # Trend buffer parameters.
    trend_epsilon: float = Field(ge=0, le=1, default=0.03)
    trend_window: int = Field(gt=0, default=10)

    # Conservative floor on overall confidence.
    min_confidence_for_decision: float = Field(ge=0, le=1, default=0.20)

    @classmethod
    def defaults(cls) -> "RiskConfig":
        """Return the same defaults; the calculator can use this when
        no YAML file is loaded (matching the PHASE 11 "system is
        functional from day one" pattern)."""
        return cls()


# ---------------------------------------------------------------------
# Dashboard (PHASE 13)
# ---------------------------------------------------------------------
class DashboardConfig(_Base):
    """Real-time dashboard server configuration (PHASE 13).

    All fields are operator-tunable. The dashboard is runnable from a
    fresh checkout even if ``dashboard.yaml`` is absent — see
    :func:`load_dashboard_config` in :mod:`backend.config.loader`.
    """

    provenance: Provenance = Provenance.USER_CONFIGURED
    notes: list[str] = Field(default_factory=list)

    # Server bind.
    host: str = "127.0.0.1"
    port: int = Field(gt=0, le=65535, default=8000)

    # Production cadence.
    tick_rate_hz: float = Field(gt=0, le=200, default=10.0)

    # Ring buffer / queue depth for the latest snapshots.
    history_size: int = Field(gt=1, le=10_000, default=600)

    # Scenario registry.
    default_scenario: str = Field(
        default="engine_degradation_60s",
        description="Scenario name from backend.dashboard.scenarios.SCENARIO_REGISTRY",
    )

    # Static files.
    static_dir: str = "frontend"

    # Server-side knobs.
    log_level: str = "INFO"
    max_ws_clients: int = Field(gt=0, le=256, default=8)

    @classmethod
    def defaults(cls) -> "DashboardConfig":
        """Return the same defaults; the server can use this when
        no YAML file is loaded (matches the PHASE 12 "system is
        functional from day one" pattern)."""
        return cls()


# ---------------------------------------------------------------------
# Hardware interface (PHASE 14)
# ---------------------------------------------------------------------
class HardwareConfig(_Base):
    """Serial / UART hardware interface configuration (PHASE 14).

    All fields are operator-tunable. The transport is runnable from
    a fresh checkout with ``port: "loopback"`` so no /dev/tty* device
    is required for development or CI. See
    :mod:`backend.hardware.ports` for the loopback implementation.
    """

    provenance: Provenance = Provenance.USER_CONFIGURED
    notes: list[str] = Field(default_factory=list)

    # Port selection. "loopback" is the built-in in-process port pair
    # used by tests + smoke runs. Real values are e.g. "/dev/ttyUSB0"
    # (Linux) or "/dev/cu.usbserial-A50285BI" (macOS).
    port: str = Field(min_length=1, default="loopback")

    # Serial parameters. pyserial maps these 1:1 onto Serial().
    baudrate: int = Field(gt=0, le=10_000_000, default=115200)
    bytesize: int = Field(ge=5, le=8, default=8)
    parity: str = Field(default="N")  # N / E / O
    stopbits: float = Field(ge=1, le=2, default=1)

    # Read behaviour.
    read_timeout_s: float = Field(gt=0, le=60, default=0.5)
    read_chunk_bytes: int = Field(gt=0, le=1_048_576, default=4096)

    # Wire-protocol CRC policy. "ccitt" = CRC-16-CCITT/XMODEM
    # (4 hex digits). "none" disables CRC for debugging.
    crc: str = Field(default="ccitt")

    # Reconnect on transient errors (e.g. USB unplug/replug).
    reconnect_on_error: bool = True
    reconnect_backoff_s: float = Field(gt=0, le=60, default=1.0)

    # Optional channel-name allowlist. None = accept any channel the
    # sensor stack knows about (the SENSOR_CHANNELS set).
    expected_channels: Optional[list[str]] = None

    @field_validator("parity")
    @classmethod
    def _validate_parity(cls, v: str) -> str:
        v = v.upper()
        if v not in {"N", "E", "O", "M", "S"}:
            raise ValueError(f"parity must be one of N/E/O/M/S, got {v!r}")
        return v

    @field_validator("crc")
    @classmethod
    def _validate_crc(cls, v: str) -> str:
        v = v.lower()
        if v not in {"ccitt", "none"}:
            raise ValueError(f"crc must be 'ccitt' or 'none', got {v!r}")
        return v

    @classmethod
    def defaults(cls) -> "HardwareConfig":
        """Return the same defaults; the transport can use this when
        no YAML file is loaded (matches the PHASE 13 "system is
        functional from day one" pattern)."""
        return cls()


# ---------------------------------------------------------------------
# HIL validation harness (PHASE 15)
# ---------------------------------------------------------------------
class HilToleranceConfig(_Base):
    """Per-field tolerance bands for the HIL golden-frame comparator.

    Absolutes are in the natural unit of the field. Relatives are
    fractions of ``max(|expected|, 1)`` so a value near zero still
    has a meaningful bound.
    """

    health_abs: float = Field(ge=0, default=0.01)
    rul_abs: float = Field(ge=0, default=1.0)         # hours
    risk_abs: float = Field(ge=0, default=0.02)
    engine_state_rel: float = Field(ge=0, le=1, default=0.005)  # 0.5 %
    environment_rel: float = Field(ge=0, le=1, default=0.005)
    anomaly_abs: float = Field(ge=0, default=0.02)


class HilReplayConfig(_Base):
    """Replay-loop knobs for the HIL harness."""

    chunk_bytes: int = Field(gt=0, le=1_048_576, default=4096)
    # `None` = replay as fast as possible (test mode). Set to a
    # positive float to throttle to that rate in Hz.
    rate_hz: Optional[float] = Field(default=None)


class HilConfig(_Base):
    """HIL validation harness configuration (PHASE 15).

    All fields are operator-tunable. The harness is runnable from a
    fresh checkout with the in-process loopback transport (PHASE 14)
    and the committed ``data/golden/`` artefacts.
    """

    provenance: Provenance = Provenance.USER_CONFIGURED
    notes: list[str] = Field(default_factory=list)

    golden_root: str = "data/golden"
    default_scenario: str = "engine_degradation_60s"
    scenarios: list[str] = Field(
        default_factory=lambda: [
            "healthy_60s",
            "engine_degradation_60s",
            "overheating_60s",
            "sensor_fault_60s",
            "vibration_anomaly_60s",
        ]
    )
    tolerance: HilToleranceConfig = Field(default_factory=HilToleranceConfig)
    replay: HilReplayConfig = Field(default_factory=HilReplayConfig)

    @classmethod
    def defaults(cls) -> "HilConfig":
        """Return the same defaults; the harness can use this when
        no YAML file is loaded (matches the PHASE 13/14 patterns)."""
        return cls()
