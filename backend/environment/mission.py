"""
Mission profile + deterministic mission runner.

The mission is a list of timed waypoints (altitude, airspeed, throttle).
Linear interpolation between consecutive waypoints gives the commanded
flight state at any time ``t``. The :class:`MissionRunner` advances
time step by step, composes the atmosphere and disturbance models, and
emits an :class:`EnvironmentState` snapshot per step — the single
object the rest of the system consumes.

PHASE 20 (MALE-UAV mission profile generator) adds three
synchronised-output dataclasses at the bottom of this file:

* :class:`MissionTick` — the full per-tick bundle (env + engine +
  sensors + fault truth). The **offline researcher** reads this.
* :class:`ObservedTick` — the same bundle minus any fault label.
  This is what the **AI** sees during inference.
* :class:`MissionTrace` — the result of a full mission run: a
  parallel pair of lists (truth + observed) so the researcher can
  compute labels from ``truth`` without ever exposing them at
  inference time.

These three classes are pure data — no I/O, no side effects — so
they round-trip cleanly through ``to_dict()`` / JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from backend.config import EnvironmentConfig, Mission, Waypoint

from .atmosphere import Atmosphere, AtmosphericState
from .disturbances import GustModel, TurbulenceModel, compute_wind
from .state import WindState

# PHASE 20 — EngineState + sensor types are referenced in type hints
# only. The imports are deferred to a TYPE_CHECKING block so we don't
# pull `backend.simulation` or `backend.sensors` (both of which depend
# on EnvironmentState) at module load time. That keeps the
# `environment <-> simulation` and `environment <-> sensors` import
# cycles from breaking.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.simulation import EngineState
    from backend.sensors import SensorReading, SensorSample


# ---------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class _InterpolatedPoint:
    t_s: float
    altitude_m: float
    airspeed_mps: float
    throttle: float


class MissionProfile:
    """Sorted, validated, linearly-interpolated mission profile."""

    def __init__(self, waypoints: Iterable[Waypoint]) -> None:
        wps = sorted(waypoints, key=lambda w: w.t_s)
        if len(wps) < 2:
            raise ValueError("mission profile requires at least 2 waypoints")
        for a, b in zip(wps, wps[1:]):
            if b.t_s <= a.t_s:
                raise ValueError("mission waypoint times must be strictly increasing")
        self._wps: List[Waypoint] = wps

    @classmethod
    def from_config(cls, cfg: Mission) -> "MissionProfile":
        return cls(cfg.profile)

    @property
    def duration_s(self) -> float:
        return self._wps[-1].t_s

    @property
    def waypoints(self) -> List[Waypoint]:
        return list(self._wps)

    def at(self, t_s: float) -> _InterpolatedPoint:
        """Return the commanded flight state at time ``t_s`` (linear interp)."""
        t = max(0.0, float(t_s))
        if t <= self._wps[0].t_s:
            w = self._wps[0]
            return _InterpolatedPoint(w.t_s, w.altitude_m, w.airspeed_mps, w.throttle)
        if t >= self._wps[-1].t_s:
            w = self._wps[-1]
            return _InterpolatedPoint(w.t_s, w.altitude_m, w.airspeed_mps, w.throttle)
        # Find bracketing waypoints.
        lo, hi = self._wps[0], self._wps[1]
        for a, b in zip(self._wps, self._wps[1:]):
            if a.t_s <= t <= b.t_s:
                lo, hi = a, b
                break
        span = hi.t_s - lo.t_s
        u = 0.0 if span <= 0.0 else (t - lo.t_s) / span
        return _InterpolatedPoint(
            t_s=t,
            altitude_m=lo.altitude_m + u * (hi.altitude_m - lo.altitude_m),
            airspeed_mps=lo.airspeed_mps + u * (hi.airspeed_mps - lo.airspeed_mps),
            throttle=lo.throttle + u * (hi.throttle - lo.throttle),
        )


# ---------------------------------------------------------------------
# Composite environment state (what the rest of the system sees)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EnvironmentState:
    """Single tick of the environment seen by the engine + diagnostics."""

    time_s: float
    altitude_m: float
    airspeed_mps: float
    throttle: float          # commanded, [0, 1]
    atmosphere: AtmosphericState
    wind: WindState
    vertical_accel_mps2: float   # rough proxy: d(w)/dt (used to drive IMU later)

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "altitude_m": self.altitude_m,
            "airspeed_mps": self.airspeed_mps,
            "throttle": self.throttle,
            "vertical_accel_mps2": self.vertical_accel_mps2,
            **self.atmosphere.to_dict(),
            **self.wind.to_dict(),
        }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
class MissionRunner:
    """Deterministic, reproducible step-by-step mission runner."""

    def __init__(self, cfg: EnvironmentConfig, dt_s: float = 0.1) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self._cfg = cfg
        self._dt = float(dt_s)
        self._atmos = Atmosphere(cfg.atmosphere)
        self._profile = MissionProfile.from_config(cfg.mission)
        self._turb = TurbulenceModel(cfg.disturbances.turbulence)
        self._gust = GustModel(cfg.disturbances.gusts)
        self._t = 0.0
        self._prev_w = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def dt_s(self) -> float:
        return self._dt

    @property
    def duration_s(self) -> float:
        return self._profile.duration_s

    @property
    def atmosphere(self) -> Atmosphere:
        return self._atmos

    @property
    def profile(self) -> MissionProfile:
        return self._profile

    @property
    def turbulence(self) -> "TurbulenceModel":
        """The active turbulence model (exposed for the fault injector)."""
        return self._turb

    @property
    def gusts(self) -> "GustModel":
        """The active gust model (exposed for the fault injector)."""
        return self._gust

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._t = 0.0
        self._prev_w = 0.0
        self._turb.reset()
        self._gust.reset()

    def step(self) -> EnvironmentState:
        """Advance the mission by one fixed step and return environment state."""
        cmd = self._profile.at(self._t)
        wind = compute_wind(self._t, self._dt, self._turb, self._gust)
        atmos = self._atmos.state(cmd.altitude_m)
        # Vertical acceleration proxy: simple difference of total wind.
        # Sufficient as a disturbance-driven IMU input for PHASE 2.
        v_accel = (wind.total_w_mps - self._prev_w) / self._dt if self._dt > 0 else 0.0
        self._prev_w = wind.total_w_mps
        state = EnvironmentState(
            time_s=self._t,
            altitude_m=cmd.altitude_m,
            airspeed_mps=cmd.airspeed_mps,
            throttle=cmd.throttle,
            atmosphere=atmos,
            wind=wind,
            vertical_accel_mps2=v_accel,
        )
        self._t += self._dt
        return state

    def run(self) -> List[EnvironmentState]:
        """Run the entire mission to completion and return the full trace."""
        import math as _math

        self.reset()
        states: List[EnvironmentState] = []
        n_steps = int(_math.ceil(self._profile.duration_s / self._dt))
        for _ in range(n_steps):
            states.append(self.step())
        return states

    def run_up_to(self, t_end_s: float) -> List[EnvironmentState]:
        """Run from the current time up to ``t_end_s`` and return new states."""
        if t_end_s < self._t:
            raise ValueError(
                f"t_end_s ({t_end_s}) is before current time ({self._t})"
            )
        states: List[EnvironmentState] = []
        while self._t + 1e-9 < t_end_s:
            states.append(self.step())
        return states


__all__ = [
    "EnvironmentState",
    "FaultTruthRecord",
    "MissionProfile",
    "MissionRunner",
    "MissionTick",
    "MissionTrace",
    "ObservedTick",
]


# ---------------------------------------------------------------------
# PHASE 20 — synchronised per-tick output
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaultTruthRecord:
    """The ground-truth fault state at one tick.

    The fields are what the :class:`~backend.faults.FaultInjector`
    decided to do at this tick. The class is intentionally minimal
    — it captures the four pieces of information the offline
    researcher needs to label a sample for supervised training.

    This record lives **only** on :class:`MissionTick` (the
    researcher-facing side). The AI-facing :class:`ObservedTick`
    has no equivalent field — that is how the constraint "the AI
    must never receive the ground-truth fault label during
    inference" is enforced structurally.
    """

    fault_class: str          # FaultClass.value; "HEALTHY" when no fault
    severity: float           # [0, 1] at this tick (linear progression)
    sensor_mode: Optional[str]   # NoiseMode.value if a sensor fault is
                                # active at this tick; None otherwise
    target_channel: Optional[str]   # channel name for a sensor fault,
                                    # None otherwise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_class": self.fault_class,
            "severity": float(self.severity),
            "sensor_mode": self.sensor_mode,
            "target_channel": self.target_channel,
        }


@dataclass(frozen=True)
class MissionTick:
    """One synchronised per-tick bundle — the **researcher's** view.

    Bundles every layer the MALE-UAV mission generator produces
    into a single frozen record:

    * ``env`` — the PHASE 2 :class:`EnvironmentState`
      (altitude, airspeed, throttle, atmosphere, wind, …).
    * ``engine`` — the PHASE 3 :class:`EngineState` (RPM, MAP,
      EGT, CHT, oil, vibration, wear, …). This is the *truth*
      engine state — see :class:`ObservedTick` for the AI view.
    * ``sensors`` — the per-channel :class:`SensorReading` map
      (the observed values + their :class:`NoiseMode`).
    * ``fault_truth`` — the :class:`FaultTruthRecord` for this
      tick. **Only present here.** The AI-facing
      :class:`ObservedTick` does not have this field — that
      is how the "AI never sees the ground-truth fault label"
      rule is enforced.
    """

    time_s: float
    env: EnvironmentState
    engine: EngineState
    sensors: Dict[str, SensorReading]
    fault_truth: FaultTruthRecord

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict.

        Includes the fault truth so the offline researcher can
        build labelled training sets. The AI-facing
        :meth:`ObservedTick.to_dict` deliberately omits every
        fault field.
        """
        return {
            "time_s": float(self.time_s),
            "env": self.env.to_dict(),
            "engine": self.engine.to_dict(),
            "sensors": {
                ch: {"value": r.value, "mode": r.mode.value}
                for ch, r in self.sensors.items()
            },
            "fault_truth": self.fault_truth.to_dict(),
        }


@dataclass(frozen=True)
class ObservedTick:
    """The **AI's** per-tick view — no fault label.

    Structurally identical to :class:`MissionTick` minus
    ``fault_truth``. The dataclass has no fault fields at all;
    the AI cannot see them even by accident. A test asserts
    that the JSON of :meth:`to_dict` contains none of the keys
    ``fault_class``, ``severity``, ``sensor_mode``,
    ``target_channel``.
    """

    time_s: float
    env: EnvironmentState
    engine: EngineState
    sensors: Dict[str, SensorReading]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict. **Never** includes fault fields."""
        return {
            "time_s": float(self.time_s),
            "env": self.env.to_dict(),
            "engine": self.engine.to_dict(),
            "sensors": {
                ch: {"value": r.value, "mode": r.mode.value}
                for ch, r in self.sensors.items()
            },
        }


@dataclass(frozen=True)
class MissionTrace:
    """The full output of :class:`ProfileGenerator.run`.

    Two parallel lists so the researcher can compute labels from
    ``truth`` after the fact while the AI consumer can iterate
    ``observed`` during inference. The two lists are guaranteed
    to be the same length and the same time axis.
    """

    truth: Tuple[MissionTick, ...]
    observed: Tuple[ObservedTick, ...]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.truth)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "truth": [t.to_dict() for t in self.truth],
            "observed": [t.to_dict() for t in self.observed],
        }
