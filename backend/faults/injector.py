"""
Fault injector — drives the simulator and sensor bundle from a scenario.

Given a :class:`~backend.faults.plan.FaultScenario`, the injector
turns the abstract plan into concrete values for the existing engine
inputs (``degradation_severity``, ``vibration_external``) and the
sensor bundle (``inject_fault(channel, mode, start_t)``) and the
environment model (``turbulence.set_intensity`` /
``gusts.set_amplitude``).

The injector does **not** modify the engine, the sensor bundle, or
the mission runner — it uses their public APIs only. This keeps
PHASE 7 a thin orchestrator on top of PHASES 3, 4, 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from backend.environment.disturbances import GustModel, TurbulenceModel
from backend.sensors import NoiseMode, SensorBundle
from backend.simulation import EngineInputs, EngineSimulator

from .plan import FaultClass, FaultScenario, SensorFaultPlan, SeverityPlan


# ---------------------------------------------------------------------
# Per-tick output of the injector.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaultTick:
    """What the injector wants the simulator to do at this tick."""

    degradation_severity: float
    vibration_external: float

    def to_engine_inputs_kwargs(self) -> dict:
        return {
            "degradation_severity": self.degradation_severity,
            "vibration_external": self.vibration_external,
        }


# ---------------------------------------------------------------------
# Class -> driver coefficients.
# ---------------------------------------------------------------------
# These constants are how each fault class affects the engine's truth
# layer via the existing EngineInputs hook. They are SYNTHETIC (see
# faults.yaml) and intentionally simple: a single scalar in [0, 1]
# that scales the chosen physical effect.
# ---------------------------------------------------------------------
_ENGINE_CLASS_SEVERITY_SCALE = {
    FaultClass.HEALTHY: 0.0,
    FaultClass.ENGINE_DEGRADATION: 1.0,
    FaultClass.OVERHEATING: 0.6,
    FaultClass.LUBRICATION_PRESSURE_ANOMALY: 0.5,
    # PERFORMANCE_LOSS does not use degradation_severity because the
    # engine's severity path also touches wear, which is a side-effect
    # we don't want. We drop RPM target differently — see _tick_engine.
    FaultClass.PERFORMANCE_LOSS: 0.0,
    FaultClass.VIBRATION_ENGINE_ANOMALY: 0.0,    # uses vibration_external
    FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE: 0.0,  # identity at truth layer
}

_VIBRATION_EXTERNAL_GAIN = {
    FaultClass.VIBRATION_ENGINE_ANOMALY: 5.0,    # g per unit severity
    # Environmental disturbance couples into airframe vibration
    # even when the engine is healthy. The gain is tuned to push
    # the *raw* vibration reading above the naive 3-sigma
    # threshold (the demo scenario's "naive threshold would
    # have flagged this" claim) while leaving every other
    # engine channel nominal — the multi-channel fusion is
    # what tells the engine from the airframe.
    FaultClass.ENVIRONMENTAL_DISTURBANCE: 2.5,    # g per unit severity
    FaultClass.HEALTHY: 0.0,
    FaultClass.ENGINE_DEGRADATION: 0.0,
    FaultClass.OVERHEATING: 0.0,
    FaultClass.LUBRICATION_PRESSURE_ANOMALY: 0.0,
    FaultClass.PERFORMANCE_LOSS: 0.0,
    FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE: 0.0,
}

# ENVIRONMENTAL_DISTURBANCE: how much to bump turbulence intensity and
# gust amplitude when the fault is active (severity scales these
# linearly between the configured baseline and the "fault" maximum).
_ENV_TURBULENCE_FAULT_INTENSITY = 0.4
_ENV_GUST_FAULT_AMPLITUDE_MPS = 12.0


class FaultInjector:
    """Drives the simulator and sensor bundle from a fault scenario.

    Usage::

        scenario = FaultScenario(FaultClass.ENGINE_DEGRADATION,
                                 severity=0.6, onset_time_s=20.0,
                                 duration_s=80.0, seed=1)
        inj = FaultInjector(scenario)
        inj.attach(sim, bundle)
        while running:
            tick = inj.tick(current_time_s)
            step = sim.step(degradation_severity=tick.degradation_severity,
                            vibration_external=tick.vibration_external)

    The injector can also be used in *advisory* mode (no
    ``attach``): just call :meth:`tick` to get the recommended values
    and pass them to the simulator yourself.
    """

    def __init__(self, scenario: FaultScenario) -> None:
        self._scenario = scenario
        self._severity_plan = SeverityPlan(scenario)
        # Sensor plan only meaningful for SENSOR_FAULT.
        self._sensor_plan: Optional[SensorFaultPlan] = None
        if scenario.fault_class is FaultClass.SENSOR_FAULT:
            self._sensor_plan = SensorFaultPlan(scenario)
        # Track whether the side-channel injection has fired.
        self._sensor_injected = False
        self._sensor_cleared = False
        # Cache the env models we mutate.
        self._sim: Optional[EngineSimulator] = None
        self._bundle: Optional[SensorBundle] = None
        self._baseline_turb_intensity: Optional[float] = None
        self._baseline_gust_amplitude: Optional[float] = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def scenario(self) -> FaultScenario:
        return self._scenario

    @property
    def severity_plan(self) -> SeverityPlan:
        return self._severity_plan

    @property
    def sensor_plan(self) -> Optional[SensorFaultPlan]:
        return self._sensor_plan

    # ------------------------------------------------------------------
    # Attach to a simulator + sensor bundle
    # ------------------------------------------------------------------
    def attach(self, sim: EngineSimulator, bundle: SensorBundle) -> None:
        """Attach to a running simulator + sensor bundle.

        Captures the *baseline* turbulence / gust settings so we can
        restore them when the fault ends or is cleared.
        """
        self._sim = sim
        self._bundle = bundle
        # Capture baselines from the live environment models.
        runner = sim.runner
        turb = runner.turbulence
        gust = runner.gusts
        if turb.enabled:
            # Recover the intensity-envelope: sigma / 3 (the default gain).
            self._baseline_turb_intensity = turb._sigma / 3.0
        else:
            self._baseline_turb_intensity = 0.0
        if gust.enabled:
            self._baseline_gust_amplitude = gust._max_amplitude
        else:
            self._baseline_gust_amplitude = 0.0

    def reset(self) -> None:
        """Reset the injector's internal state (e.g. between missions)."""
        self._sensor_injected = False
        self._sensor_cleared = False
        # If attached, restore the env baseline.
        if (
            self._sim is not None
            and self._scenario.fault_class is FaultClass.ENVIRONMENTAL_DISTURBANCE
        ):
            self._restore_env_baseline()

    # ------------------------------------------------------------------
    # Per-tick advisory
    # ------------------------------------------------------------------
    def tick(self, t_s: float) -> FaultTick:
        """Return the engine-input values to apply at time ``t_s``."""
        sev = self._severity_plan.at(t_s)
        if self._scenario.fault_class is FaultClass.HEALTHY:
            return FaultTick(0.0, 0.0)
        if self._scenario.fault_class is FaultClass.PERFORMANCE_LOSS:
            # Map severity -> rpm_efficiency_drop, then drive that via
            # the engine's existing severity pathway at a 0.8 scale.
            return FaultTick(0.8 * sev, 0.0)
        if self._scenario.fault_class is FaultClass.VIBRATION_ENGINE_ANOMALY:
            gain = _VIBRATION_EXTERNAL_GAIN[self._scenario.fault_class]
            return FaultTick(0.0, gain * sev)
        if self._scenario.fault_class is FaultClass.ENVIRONMENTAL_DISTURBANCE:
            # Airframe vibration couples through turbulence even
            # when the engine is healthy (the env layer's
            # turbulence_intensity / gust_amplitude is still
            # applied via apply_environment by the caller).
            gain = _VIBRATION_EXTERNAL_GAIN[self._scenario.fault_class]
            return FaultTick(0.0, gain * sev)
        if self._scenario.fault_class is FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE:
            return FaultTick(0.0, 0.0)
        if self._scenario.fault_class is FaultClass.SENSOR_FAULT:
            return FaultTick(0.0, 0.0)
        # Remaining: ENGINE_DEGRADATION, OVERHEATING, LUBRICATION_PRESSURE_ANOMALY.
        scale = _ENGINE_CLASS_SEVERITY_SCALE[self._scenario.fault_class]
        return FaultTick(scale * sev, 0.0)

    # ------------------------------------------------------------------
    # Side-channel: sensor fault injection
    # ------------------------------------------------------------------
    def apply_sensor(self, t_s: float) -> bool:
        """Inject or clear the configured sensor fault at time ``t_s``.

        Returns True if the state changed (injected or cleared). The
        caller is responsible for invoking this once per tick *after*
        the sensor bundle's ``tick()`` has produced a sample.
        """
        if self._sensor_plan is None or self._bundle is None:
            return False
        # Inject at onset.
        if (not self._sensor_injected) and t_s >= self._sensor_plan.start_t:
            self._bundle.inject_fault(
                self._sensor_plan.channel,
                self._sensor_plan.mode,
                start_t=self._sensor_plan.start_t,
            )
            self._sensor_injected = True
            return True
        # Clear at end (if a duration was set).
        if (
            self._sensor_injected
            and not self._sensor_cleared
            and self._sensor_plan.end_t is not None
            and t_s >= self._sensor_plan.end_t
        ):
            self._bundle.clear_fault(self._sensor_plan.channel)
            self._sensor_cleared = True
            return True
        return False

    # ------------------------------------------------------------------
    # Side-channel: environmental disturbance
    # ------------------------------------------------------------------
    def apply_environment(self, t_s: float) -> bool:
        """Apply the configured turbulence / gust settings for this tick.

        Returns True if the state changed. The injector only mutates
        the env models while the fault is active; on either side of
        the active window it restores the captured baseline.
        """
        if (
            self._sim is None
            or self._scenario.fault_class is not FaultClass.ENVIRONMENTAL_DISTURBANCE
        ):
            return False
        active = self._severity_plan.is_active(t_s)
        runner = self._sim.runner
        turb = runner.turbulence
        gust = runner.gusts
        if not active:
            if self._baseline_turb_intensity is not None and turb.enabled:
                turb.set_intensity(self._baseline_turb_intensity)
            if self._baseline_gust_amplitude is not None and gust.enabled:
                gust.set_amplitude(self._baseline_gust_amplitude)
            return False
        # Active: scale between baseline and fault-max.
        sev = float(self._severity_plan.at(t_s))
        if turb.enabled:
            base = self._baseline_turb_intensity or 0.0
            target = base + (_ENV_TURBULENCE_FAULT_INTENSITY - base) * sev
            turb.set_intensity(target)
        if gust.enabled:
            base = self._baseline_gust_amplitude or 0.0
            target = base + (_ENV_GUST_FAULT_AMPLITUDE_MPS - base) * sev
            gust.set_amplitude(target)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _restore_env_baseline(self) -> None:
        if self._sim is None:
            return
        runner = self._sim.runner
        if self._baseline_turb_intensity is not None and runner.turbulence.enabled:
            runner.turbulence.set_intensity(self._baseline_turb_intensity)
        if self._baseline_gust_amplitude is not None and runner.gusts.enabled:
            runner.gusts.set_amplitude(self._baseline_gust_amplitude)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def is_active(self, t_s: float) -> bool:
        return self._severity_plan.is_active(t_s)

    def make_engine_inputs(
        self, env, t_s: float
    ) -> "EngineInputs":
        """Return an EngineInputs populated for the given env + time."""
        tick = self.tick(t_s)
        return EngineInputs(
            env=env,
            degradation_severity=tick.degradation_severity,
            vibration_external=tick.vibration_external,
        )


__all__ = ["FaultInjector", "FaultTick"]
