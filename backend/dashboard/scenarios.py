"""
Scenario registry (PHASE 13).

A :class:`ScenarioSpec` is the dashboard's view of a fault scenario —
it is a *name + small bundle of metadata* that maps to a PHASE 7
:class:`~backend.faults.FaultScenario` (and optionally a PHASE 9
:class:`~backend.ml.FaultClassifier` configuration) at construction
time. The registry is the single source of truth for the
``/api/scenarios`` endpoint and the frontend's ``<select>`` dropdown.

The dashboard defaults to :data:`DEFAULT_SCENARIO_NAME` (a 60 s
engine-degradation mission) so a fresh page load is never boring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from backend.faults import FaultClass, FaultScenario
from backend.sensors import NoiseMode


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario the dashboard can run.

    Attributes
    ----------
    name:
        Stable identifier (matches the key in :data:`SCENARIO_REGISTRY`).
        The frontend uses this as the ``<option>`` value.
    fault_class:
        Which PHASE 7 fault class to inject. ``HEALTHY`` is the
        no-fault scenario.
    severity:
        Peak severity of the fault, in ``[0, 1]``. ``HEALTHY`` ignores
        this field.
    onset_time_s:
        When the fault starts (seconds, sim time). For ``HEALTHY``
        this is 0.
    duration_s:
        How long the fault stays active. ``None`` means the fault
        persists for the entire mission.
    seed:
        Deterministic seed for any randomness.
    progression:
        ``"linear"`` ramps the severity up over the active window;
        ``"step"`` jumps to the peak at onset and holds.
    run_classifier:
        Whether the optional PHASE 9 fault classifier is enabled for
        this scenario. Default ``False`` (the dashboard's healthy
        and degradation scenarios run faster without the windowed
        classifier and remain demonstrative).
    description:
        Human-readable description, shown in the dropdown.
    """

    name: str
    fault_class: FaultClass = FaultClass.HEALTHY
    severity: float = 0.0
    onset_time_s: float = 0.0
    duration_s: Optional[float] = None
    seed: int = 0
    progression: str = "linear"
    run_classifier: bool = False
    description: str = ""

    def to_fault_scenario(self) -> FaultScenario:
        """Convert to a PHASE 7 :class:`FaultScenario`."""
        kwargs = dict(
            fault_class=self.fault_class,
            severity=self.severity,
            onset_time_s=self.onset_time_s,
            duration_s=self.duration_s,
            seed=self.seed,
            progression=self.progression,
        )
        if self.fault_class is FaultClass.SENSOR_FAULT:
            kwargs["target_channel"] = "rpm"
            kwargs["target_sensor_mode"] = NoiseMode.STUCK
        return FaultScenario(**kwargs)

    def to_dict(self) -> dict:
        """Wire-format dict (matches the /api/scenarios list endpoint)."""
        return {
            "name": self.name,
            "fault_class": self.fault_class.value,
            "severity": float(self.severity),
            "onset_time_s": float(self.onset_time_s),
            "duration_s": None if self.duration_s is None else float(self.duration_s),
            "seed": int(self.seed),
            "progression": self.progression,
            "run_classifier": bool(self.run_classifier),
            "description": self.description,
        }


# ---------------------------------------------------------------------
# Classmethods — one builder per PHASE 7 fault class
# ---------------------------------------------------------------------
@classmethod  # type: ignore[arg-type]
def _healthy(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="healthy_60s",
        fault_class=FaultClass.HEALTHY,
        severity=0.0,
        onset_time_s=0.0,
        duration_s=None,
        seed=0,
        progression="linear",
        run_classifier=False,
        description="60 s mission, no faults. Baseline for comparison.",
    )


@classmethod  # type: ignore[arg-type]
def _engine_degradation(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="engine_degradation_60s",
        fault_class=FaultClass.ENGINE_DEGRADATION,
        # Severity bumped from 0.6 → 0.85 with a "step"
        # progression so the fault hits the diagnostic
        # system hard and the dashboard clearly surfaces a
        # warning. With a slow 0.6 / linear ramp the health
        # index and anomaly fusion are too conservative to
        # fire within a 60-s mission.
        severity=0.85,
        onset_time_s=10.0,
        duration_s=40.0,
        seed=1,
        progression="step",
        run_classifier=False,
        description="60 s mission, engine degradation at t=10 s. Demo default.",
    )


@classmethod  # type: ignore[arg-type]
def _overheating(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="overheating_60s",
        fault_class=FaultClass.OVERHEATING,
        # Bumped from 0.55 to 0.85 (step) so the diagnostic
        # system reaches CAUTION within the 60-s mission.
        # The 0.55 / linear ramp was too gentle to push
        # any of the alert paths over their threshold.
        severity=0.85,
        onset_time_s=15.0,
        duration_s=30.0,
        seed=2,
        progression="step",
        run_classifier=False,
        description="60 s mission, sustained overheating at t=15 s.",
    )


@classmethod  # type: ignore[arg-type]
def _lubrication(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="lubrication_pressure_60s",
        fault_class=FaultClass.LUBRICATION_PRESSURE_ANOMALY,
        severity=0.5,
        onset_time_s=8.0,
        duration_s=35.0,
        seed=3,
        progression="step",
        run_classifier=False,
        description="60 s mission, oil-pressure drop at t=8 s.",
    )


@classmethod  # type: ignore[arg-type]
def _vibration(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="vibration_anomaly_60s",
        fault_class=FaultClass.VIBRATION_ENGINE_ANOMALY,
        severity=0.7,
        onset_time_s=12.0,
        duration_s=30.0,
        seed=4,
        progression="linear",
        run_classifier=False,
        description="60 s mission, abnormal vibration at t=12 s.",
    )


@classmethod  # type: ignore[arg-type]
def _performance_loss(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="performance_loss_60s",
        fault_class=FaultClass.PERFORMANCE_LOSS,
        severity=0.5,
        onset_time_s=10.0,
        duration_s=30.0,
        seed=5,
        progression="linear",
        run_classifier=False,
        description="60 s mission, throttle/efficiency loss at t=10 s.",
    )


@classmethod  # type: ignore[arg-type]
def _environmental(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="environmental_disturbance_60s",
        fault_class=FaultClass.ENVIRONMENTAL_DISTURBANCE,
        severity=0.6,
        onset_time_s=5.0,
        duration_s=40.0,
        seed=6,
        progression="linear",
        run_classifier=False,
        description="60 s mission, heavy turbulence + gusts at t=5 s.",
    )


@classmethod  # type: ignore[arg-type]
def _sensor_fault(cls) -> ScenarioSpec:  # noqa: N805
    return cls(
        name="sensor_fault_60s",
        fault_class=FaultClass.SENSOR_FAULT,
        severity=1.0,
        onset_time_s=10.0,
        duration_s=30.0,
        seed=7,
        progression="step",
        run_classifier=False,
        description="60 s mission, stuck RPM sensor at t=10 s.",
    )


# Bind the classmethods to ScenarioSpec.
ScenarioSpec.healthy_60s = _healthy
ScenarioSpec.engine_degradation_60s = _engine_degradation
ScenarioSpec.overheating_60s = _overheating
ScenarioSpec.lubrication_pressure_60s = _lubrication
ScenarioSpec.vibration_anomaly_60s = _vibration
ScenarioSpec.performance_loss_60s = _performance_loss
ScenarioSpec.environmental_disturbance_60s = _environmental
ScenarioSpec.sensor_fault_60s = _sensor_fault


# ---------------------------------------------------------------------
# Registry + default
# ---------------------------------------------------------------------
DEFAULT_SCENARIO_NAME = "engine_degradation_60s"


def _build_registry() -> Dict[str, ScenarioSpec]:
    return {
        s.name: s
        for s in [
            ScenarioSpec.healthy_60s(),
            ScenarioSpec.engine_degradation_60s(),
            ScenarioSpec.overheating_60s(),
            ScenarioSpec.lubrication_pressure_60s(),
            ScenarioSpec.vibration_anomaly_60s(),
            ScenarioSpec.performance_loss_60s(),
            ScenarioSpec.environmental_disturbance_60s(),
            ScenarioSpec.sensor_fault_60s(),
        ]
    }


SCENARIO_REGISTRY: Dict[str, ScenarioSpec] = _build_registry()


def get_scenario(name: str) -> Optional[ScenarioSpec]:
    """Look up a scenario by name. Returns ``None`` if not found."""
    return SCENARIO_REGISTRY.get(name)


__all__ = [
    "DEFAULT_SCENARIO_NAME",
    "SCENARIO_REGISTRY",
    "ScenarioSpec",
    "get_scenario",
]
