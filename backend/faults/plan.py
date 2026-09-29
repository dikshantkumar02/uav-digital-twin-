"""
Fault plan — what to inject, where, and when.

A :class:`FaultScenario` is a *declarative* description of a fault:
which class, how strong, when it starts, how it progresses, and
(optionally) which sensor channel it targets. A :class:`SeverityPlan`
turns that into a per-tick ``severity_at(t)`` function. A
:class:`SensorFaultPlan` is a side-channel event applied once to the
sensor bundle at onset.

The classes are *pure data* (no engine knowledge). The
:class:`~backend.faults.injector.FaultInjector` is what actually drives
the simulator / sensor bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.sensors import NoiseMode

from .progression import severity_at


class FaultClass(str, Enum):
    """Catalog of supported fault classes.

    The string values match the keys in ``config/faults.yaml`` so the
    enum can be constructed directly from the config.
    """

    HEALTHY = "HEALTHY"
    ENGINE_DEGRADATION = "ENGINE_DEGRADATION"
    OVERHEATING = "OVERHEATING"
    LUBRICATION_PRESSURE_ANOMALY = "LUBRICATION_PRESSURE_ANOMALY"
    VIBRATION_ENGINE_ANOMALY = "VIBRATION_ENGINE_ANOMALY"
    PERFORMANCE_LOSS = "PERFORMANCE_LOSS"
    SENSOR_FAULT = "SENSOR_FAULT"
    ENVIRONMENTAL_DISTURBANCE = "ENVIRONMENTAL_DISTURBANCE"
    UNKNOWN_INSUFFICIENT_EVIDENCE = "UNKNOWN_INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class FaultScenario:
    """A single declarative fault scenario."""

    fault_class: FaultClass
    severity: float = 0.5
    onset_time_s: float = 0.0
    duration_s: Optional[float] = None
    progression: str = "linear"
    seed: int = 0
    # Only for SENSOR_FAULT.
    target_channel: Optional[str] = None
    target_sensor_mode: Optional[NoiseMode] = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError(
                f"severity must be in [0, 1], got {self.severity!r}"
            )
        if self.onset_time_s < 0:
            raise ValueError(
                f"onset_time_s must be >= 0, got {self.onset_time_s!r}"
            )
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError(
                f"duration_s must be >= 0 when set, got {self.duration_s!r}"
            )
        if self.progression not in ("linear", "step", "exponential", "pulse"):
            raise ValueError(
                f"progression must be 'linear', 'step', 'exponential', "
                f"or 'pulse', got {self.progression!r}"
            )
        if self.fault_class is FaultClass.SENSOR_FAULT:
            if self.target_channel is None or self.target_sensor_mode is None:
                raise ValueError(
                    "SENSOR_FAULT requires both target_channel and target_sensor_mode"
                )

    @classmethod
    def from_config(
        cls,
        fault_class: FaultClass,
        severity: float = 0.5,
        onset_time_s: float = 0.0,
        duration_s: Optional[float] = None,
        seed: int = 0,
        target_channel: Optional[str] = None,
        target_sensor_mode: Optional[NoiseMode] = None,
        progression: str = "linear",
    ) -> "FaultScenario":
        return cls(
            fault_class=fault_class,
            severity=severity,
            onset_time_s=onset_time_s,
            duration_s=duration_s,
            progression=progression,
            seed=seed,
            target_channel=target_channel,
            target_sensor_mode=target_sensor_mode,
        )

    @property
    def is_engine_truth_layer(self) -> bool:
        """True if this fault lives in the engine's truth layer (PHASE 3)."""
        return self.fault_class in (
            FaultClass.HEALTHY,
            FaultClass.ENGINE_DEGRADATION,
            FaultClass.OVERHEATING,
            FaultClass.LUBRICATION_PRESSURE_ANOMALY,
            FaultClass.VIBRATION_ENGINE_ANOMALY,
            FaultClass.PERFORMANCE_LOSS,
            FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE,
        )

    @property
    def is_sensor_layer(self) -> bool:
        return self.fault_class is FaultClass.SENSOR_FAULT

    @property
    def is_environment_layer(self) -> bool:
        return self.fault_class is FaultClass.ENVIRONMENTAL_DISTURBANCE


@dataclass(frozen=True)
class SeverityPlan:
    """Time-indexed severity for a fault scenario.

    The plan is immutable; ``at(t)`` just calls the pure
    :func:`severity_at` from the progression module.
    """

    scenario: FaultScenario

    def at(self, t_s: float) -> float:
        return severity_at(
            t_s=float(t_s),
            onset_s=self.scenario.onset_time_s,
            duration_s=self.scenario.duration_s,
            peak=self.scenario.severity,
            model=self.scenario.progression,
        )

    def is_active(self, t_s: float) -> bool:
        return self.at(t_s) > 0.0


@dataclass(frozen=True)
class SensorFaultPlan:
    """One-shot sensor-side injection: which channel + which mode + when."""

    scenario: FaultScenario

    def __post_init__(self) -> None:
        if self.scenario.fault_class is not FaultClass.SENSOR_FAULT:
            raise ValueError(
                "SensorFaultPlan only applies to SENSOR_FAULT scenarios"
            )

    @property
    def channel(self) -> str:
        assert self.scenario.target_channel is not None
        return self.scenario.target_channel

    @property
    def mode(self) -> NoiseMode:
        assert self.scenario.target_sensor_mode is not None
        return self.scenario.target_sensor_mode

    @property
    def start_t(self) -> float:
        return self.scenario.onset_time_s

    @property
    def end_t(self) -> Optional[float]:
        if self.scenario.duration_s is None:
            return None
        return self.scenario.onset_time_s + self.scenario.duration_s


__all__ = [
    "FaultClass",
    "FaultScenario",
    "SensorFaultPlan",
    "SeverityPlan",
]
