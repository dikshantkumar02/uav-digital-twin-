"""
PHASE 21 — Fault taxonomy.

A *research-facing* catalog of fault types organised into three
categories (ENGINE, SENSOR, ENVIRONMENT) and mapped onto the
existing PHASE 7 :class:`~backend.faults.plan.FaultClass` so the
PHASE 7 :class:`~backend.faults.injector.FaultInjector` keeps
working unchanged.

The taxonomy is the single source of truth: every fault the
framework generates carries a :class:`FaultType` enum value,
and the registry :data:`FAULT_TAXONOMY` carries the per-type
metadata (mapped class, affected parameters, default
progression, default duration).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from backend.sensors import NoiseMode

from .plan import FaultClass


# ---------------------------------------------------------------------
# FaultCategory
# ---------------------------------------------------------------------
class FaultCategory(str, Enum):
    """The three high-level categories."""

    ENGINE = "ENGINE"
    SENSOR = "SENSOR"
    ENVIRONMENT = "ENVIRONMENT"


# ---------------------------------------------------------------------
# FaultType — the 17-element research catalog
# ---------------------------------------------------------------------
class FaultType(str, Enum):
    """The 17 research fault types.

    ENGINE (5):
        GRADUAL_PERFORMANCE_DEGRADATION,
        THERMAL_DEGRADATION,
        PRESSURE_RELATED_ANOMALY,
        VIBRATION_RELATED_ANOMALY,
        EFFICIENCY_LOSS.

    SENSOR (6):
        SENSOR_BIAS, SENSOR_DRIFT, SENSOR_DROPOUT,
        SENSOR_STUCK, SENSOR_SPIKE, SENSOR_NOISE_INCREASE.

    ENVIRONMENT (6):
        ENV_TURBULENCE, ENV_GUST, ENV_TEMP_SHIFT,
        ENV_ALTITUDE_TRANSITION, ENV_RAPID_THROTTLE,
        ENV_PRESSURE_SHIFT.
    """

    # ENGINE
    GRADUAL_PERFORMANCE_DEGRADATION = "GRADUAL_PERFORMANCE_DEGRADATION"
    THERMAL_DEGRADATION = "THERMAL_DEGRADATION"
    PRESSURE_RELATED_ANOMALY = "PRESSURE_RELATED_ANOMALY"
    VIBRATION_RELATED_ANOMALY = "VIBRATION_RELATED_ANOMALY"
    EFFICIENCY_LOSS = "EFFICIENCY_LOSS"

    # SENSOR
    SENSOR_BIAS = "SENSOR_BIAS"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    SENSOR_STUCK = "SENSOR_STUCK"
    SENSOR_SPIKE = "SENSOR_SPIKE"
    SENSOR_NOISE_INCREASE = "SENSOR_NOISE_INCREASE"

    # ENVIRONMENT
    ENV_TURBULENCE = "ENV_TURBULENCE"
    ENV_GUST = "ENV_GUST"
    ENV_TEMP_SHIFT = "ENV_TEMP_SHIFT"
    ENV_ALTITUDE_TRANSITION = "ENV_ALTITUDE_TRANSITION"
    ENV_RAPID_THROTTLE = "ENV_RAPID_THROTTLE"
    ENV_PRESSURE_SHIFT = "ENV_PRESSURE_SHIFT"


# ---------------------------------------------------------------------
# FaultTypeMeta
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaultTypeMeta:
    """Static metadata for a single :class:`FaultType`."""

    fault_type: FaultType
    category: FaultCategory
    mapped_class: FaultClass
    affected_parameters: Tuple[str, ...]
    default_progression: str
    default_duration_s: float
    # Only set for SENSOR_* types — the NoiseMode to use.
    sensor_mode: "NoiseMode | None" = None
    # The knob this type targets inside the env model.
    env_knob: "str | None" = None


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
def _meta(
    ft: FaultType,
    cat: FaultCategory,
    cls: FaultClass,
    affected: Tuple[str, ...],
    progression: str,
    duration_s: float,
    *,
    sensor_mode: "NoiseMode | None" = None,
    env_knob: "str | None" = None,
) -> FaultTypeMeta:
    return FaultTypeMeta(
        fault_type=ft,
        category=cat,
        mapped_class=cls,
        affected_parameters=affected,
        default_progression=progression,
        default_duration_s=float(duration_s),
        sensor_mode=sensor_mode,
        env_knob=env_knob,
    )


FAULT_TAXONOMY: Dict[FaultType, FaultTypeMeta] = {
    # ----- ENGINE (5) -----
    FaultType.GRADUAL_PERFORMANCE_DEGRADATION: _meta(
        FaultType.GRADUAL_PERFORMANCE_DEGRADATION,
        FaultCategory.ENGINE,
        FaultClass.ENGINE_DEGRADATION,
        ("rpm", "egt", "cht", "vibration", "bsfc"),
        "linear",
        300.0,
    ),
    FaultType.THERMAL_DEGRADATION: _meta(
        FaultType.THERMAL_DEGRADATION,
        FaultCategory.ENGINE,
        FaultClass.OVERHEATING,
        ("egt", "cht", "oil_temperature"),
        "linear",
        200.0,
    ),
    FaultType.PRESSURE_RELATED_ANOMALY: _meta(
        FaultType.PRESSURE_RELATED_ANOMALY,
        FaultCategory.ENGINE,
        FaultClass.LUBRICATION_PRESSURE_ANOMALY,
        ("oil_pressure", "oil_temperature"),
        "linear",
        200.0,
    ),
    FaultType.VIBRATION_RELATED_ANOMALY: _meta(
        FaultType.VIBRATION_RELATED_ANOMALY,
        FaultCategory.ENGINE,
        FaultClass.VIBRATION_ENGINE_ANOMALY,
        ("vibration", "bsfc", "rpm"),
        "linear",
        200.0,
    ),
    FaultType.EFFICIENCY_LOSS: _meta(
        FaultType.EFFICIENCY_LOSS,
        FaultCategory.ENGINE,
        FaultClass.PERFORMANCE_LOSS,
        ("rpm", "bsfc", "egt"),
        "linear",
        200.0,
    ),

    # ----- SENSOR (6) -----
    # SENSOR_FAULT requires target_channel + target_sensor_mode; the
    # framework fills these from the FaultTypeMeta when it builds
    # the PHASE 7 FaultScenario.
    FaultType.SENSOR_BIAS: _meta(
        FaultType.SENSOR_BIAS,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "step",
        300.0,
        sensor_mode=NoiseMode.FAULT,
    ),
    FaultType.SENSOR_DRIFT: _meta(
        FaultType.SENSOR_DRIFT,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "linear",
        300.0,
        sensor_mode=NoiseMode.DRIFTING,
    ),
    FaultType.SENSOR_DROPOUT: _meta(
        FaultType.SENSOR_DROPOUT,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "step",
        120.0,
        sensor_mode=NoiseMode.DROPPED,
    ),
    FaultType.SENSOR_STUCK: _meta(
        FaultType.SENSOR_STUCK,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "step",
        200.0,
        sensor_mode=NoiseMode.STUCK,
    ),
    FaultType.SENSOR_SPIKE: _meta(
        FaultType.SENSOR_SPIKE,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "pulse",
        200.0,
        sensor_mode=NoiseMode.SPIKE,
    ),
    FaultType.SENSOR_NOISE_INCREASE: _meta(
        FaultType.SENSOR_NOISE_INCREASE,
        FaultCategory.SENSOR,
        FaultClass.SENSOR_FAULT,
        ("target_channel",),
        "linear",
        200.0,
        sensor_mode=NoiseMode.FAULT,
    ),

    # ----- ENVIRONMENT (6) -----
    # All map to ENVIRONMENTAL_DISTURBANCE; the env_knob records
    # *which* knob this fault turns. The framework applies it
    # to the loaded environment config before launching the run.
    FaultType.ENV_TURBULENCE: _meta(
        FaultType.ENV_TURBULENCE,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("turbulence_intensity",),
        "linear",
        120.0,
        env_knob="turbulence_intensity",
    ),
    FaultType.ENV_GUST: _meta(
        FaultType.ENV_GUST,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("gust_amplitude_mps",),
        "linear",
        120.0,
        env_knob="gust_amplitude_mps",
    ),
    FaultType.ENV_TEMP_SHIFT: _meta(
        FaultType.ENV_TEMP_SHIFT,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("ambient_temperature_c",),
        "step",
        300.0,
        env_knob="ambient_temperature_c",
    ),
    FaultType.ENV_ALTITUDE_TRANSITION: _meta(
        FaultType.ENV_ALTITUDE_TRANSITION,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("altitude_m",),
        "exponential",
        300.0,
        env_knob="altitude_m",
    ),
    FaultType.ENV_RAPID_THROTTLE: _meta(
        FaultType.ENV_RAPID_THROTTLE,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("throttle",),
        "pulse",
        60.0,
        env_knob="throttle",
    ),
    FaultType.ENV_PRESSURE_SHIFT: _meta(
        FaultType.ENV_PRESSURE_SHIFT,
        FaultCategory.ENVIRONMENT,
        FaultClass.ENVIRONMENTAL_DISTURBANCE,
        ("ambient_pressure",),
        "step",
        300.0,
        env_knob="ambient_pressure",
    ),
}


# ---------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------
def get_meta(ft: FaultType) -> FaultTypeMeta:
    """Return the FaultTypeMeta for a FaultType. Raises KeyError if unknown."""
    return FAULT_TAXONOMY[ft]


def by_category(category: FaultCategory) -> Tuple[FaultType, ...]:
    """Return the FaultTypes in a given category, in declaration order."""
    return tuple(ft for ft, meta in FAULT_TAXONOMY.items() if meta.category is category)


def by_class(cls: FaultClass) -> Tuple[FaultType, ...]:
    """Return the FaultTypes that map to a given PHASE 7 FaultClass."""
    return tuple(ft for ft, meta in FAULT_TAXONOMY.items() if meta.mapped_class is cls)


__all__ = [
    "FAULT_TAXONOMY",
    "FaultCategory",
    "FaultType",
    "FaultTypeMeta",
    "by_category",
    "by_class",
    "get_meta",
]
