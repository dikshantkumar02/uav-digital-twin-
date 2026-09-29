"""
PHASE 20 — Profile generator: compiles a phase list into a
synchronised mission trace.

The :class:`ProfileGenerator` is the orchestrator that turns a
list of :class:`MissionPhaseSpec` (or a built-in
:class:`MissionTemplate`) into a :class:`MissionTrace` — a pair
of per-tick arrays:

* ``truth``  — full state including the ground-truth fault label
  (:class:`MissionTick`). For the **offline researcher**.
* ``observed`` — same as truth but with the fault field removed
  (:class:`ObservedTick`). For the **AI at inference time**.

The generator composes the existing PHASE 2 ``MissionRunner``,
PHASE 3 ``Engine`` (via the ``EngineSimulator``), PHASE 4
``SensorBundle``, and PHASE 7 ``FaultInjector``. The only additive
changes to the data layer are:

* ``EngineInputs.engine_load`` (PHASE 20, default 0.0) — biases
  vibration / BSFC / fuel flow when the mission profile commands
  a non-zero engine load.
* ``WindState.steady_wind_mps`` (PHASE 20, default 0.0) — carries
  the per-phase steady wind bias.

Continuous transitions
----------------------

Each :class:`MissionPhaseSpec` is compiled into a pair of
``Waypoint`` entries so the existing linear interpolation between
consecutive waypoints gives a smooth altitude / airspeed /
throttle ramp. The engine's existing first-order lags
(``tau_rpm``, ``tau_thermal``, ``tau_oil``) further smooth the
*truth* state, so the observed values are C¹-continuous even
when the commanded throttle is C⁰.

Random seed
-----------

The :class:`ProfileGeneratorConfig` carries a ``seed`` that
seeds:

* ``numpy``'s global default RNG — consumed by the
  :class:`TurbulenceModel` and :class:`GustModel` RNGs via the
  config (the generator patches the env config's seeds before
  building the runner).
* The optional :class:`FaultScenario`'s ``seed``.

Two runs of the same template + same seed produce bit-identical
``MissionTrace.truth`` and ``.observed`` lists.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from backend.config import (
    EnvironmentConfig,
    Mission,
    Waypoint,
)
from backend.environment.atmosphere import Atmosphere
from backend.environment.disturbances import compute_wind
from backend.environment.mission import (
    EnvironmentState,
    FaultTruthRecord,
    MissionProfile,
    MissionRunner,
    MissionTick,
    MissionTrace,
    ObservedTick,
)
from backend.environment.phases import MissionPhaseSpec, PhaseKind
from backend.environment.state import WindState
from backend.simulation import EngineInputs, EngineSimulator

# The fault / sensor imports below are deferred to avoid the package-level
# cycle: backend.sensors.bundle → backend.environment → profile_generator
# → backend.faults.injector → backend.sensors.*. They are imported lazily
# inside the methods that actually need them.
from .templates import MISSION_TEMPLATES, MissionTemplate


# ---------------------------------------------------------------------
# Generator configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileGeneratorConfig:
    """How the generator is run.

    Attributes
    ----------
    seed:
        Master random seed. Used to (a) seed the env config's
        turbulence / gust RNGs, (b) seed the optional fault
        scenario. Two runs with the same seed produce bit-identical
        traces.
    dt_s:
        Time step for the underlying runner. Defaults to 0.1 s,
        matching the existing PHASE 2 default.
    transition_default_s:
        Default outgoing-transition length when a phase does not
        override it. Individual phases may override (the
        ``RAPID_THROTTLE`` template shortens it to 1 s).
    fault_plan:
        Optional single fault to overlay on the mission. Defaults
        to a healthy scenario (no fault).
    """

    seed: int = 42
    dt_s: float = 0.1
    transition_default_s: float = 5.0
    fault_plan: Optional[FaultScenario] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_phase_start_times(
    phases: Sequence[MissionPhaseSpec],
    transition_default_s: float,
) -> List[MissionPhaseSpec]:
    """Resolve ``start_t_s`` for each phase by walking the duration list.

    Mutates *no* input — returns new dataclasses with the same
    field values and the start times filled in. The first phase
    starts at 0.0; each subsequent phase starts at the previous
    one's ``end_t_s``.
    """
    if not phases:
        return []
    resolved: List[MissionPhaseSpec] = []
    cursor = 0.0
    for ph in phases:
        # Re-construct the dataclass with start_t_s set to cursor.
        # If the phase did not specify a custom transition_s we
        # use the generator's default.
        ts = (
            float(ph.transition_s)
            if ph.transition_s is not None
            else float(transition_default_s)
        )
        new_ph = MissionPhaseSpec(
            kind=ph.kind,
            start_t_s=float(cursor),
            duration_s=float(ph.duration_s),
            target_altitude_m=float(ph.target_altitude_m),
            target_airspeed_mps=float(ph.target_airspeed_mps),
            target_throttle=float(ph.target_throttle),
            engine_load=float(ph.engine_load),
            ambient_temperature_c=ph.ambient_temperature_c,
            pressure_pa=ph.pressure_pa,
            wind_mps=float(ph.wind_mps),
            turbulence_intensity=float(ph.turbulence_intensity),
            transition_s=float(ts),
        )
        resolved.append(new_ph)
        cursor = float(new_ph.end_t_s)
    return resolved


def _phase_at(t_s: float, phases: Sequence[MissionPhaseSpec]) -> MissionPhaseSpec:
    """Return the phase active at time ``t_s``.

    Linear scan is O(N) but N=10 per mission, so this is fine.
    """
    if not phases:
        raise ValueError("phase list is empty")
    # Clamp negative times to the first phase.
    if t_s <= phases[0].start_t_s:
        return phases[0]
    for ph in phases:
        if ph.start_t_s <= t_s < ph.end_t_s:
            return ph
    return phases[-1]


def _phase_lerp_value(
    t_s: float,
    phases: Sequence[MissionPhaseSpec],
    attr: str,
    fallback: float,
) -> float:
    """Linearly interpolate a per-phase scalar across the active
    transition window into the next phase.

    Within a phase, the value is held constant. Over the last
    ``transition_s`` of the phase, it ramps to the *next* phase's
    value. For the last phase the ramp is held at the same value
    (no successor to ramp to).

    The first phase's value is also held at the start (no
    predecessor). The two pieces — the constant value of phase N
    and the constant value of phase N+1, joined by a linear ramp
    over the transition window — give a piecewise linear profile
    that the existing first-order engine lags will further smooth.
    """
    if t_s <= phases[0].start_t_s:
        return float(getattr(phases[0], attr))
    for i, ph in enumerate(phases):
        if ph.start_t_s <= t_s < ph.end_t_s:
            v0 = float(getattr(ph, attr))
            if i + 1 >= len(phases):
                return v0
            next_ph = phases[i + 1]
            v1 = float(getattr(next_ph, attr))
            t_in_phase = t_s - ph.start_t_s
            # The transition window sits at the END of the phase.
            if t_in_phase <= ph.duration_s - ph.transition_s:
                return v0
            t_in_trans = t_in_phase - (ph.duration_s - ph.transition_s)
            if ph.transition_s <= 0.0:
                return v1
            u = max(0.0, min(1.0, t_in_trans / ph.transition_s))
            return v0 + u * (v1 - v0)
    # Past the last phase — clamp to last phase's value.
    return float(getattr(phases[-1], attr))


def _phase_lerp_optional(
    t_s: float,
    phases: Sequence[MissionPhaseSpec],
    attr: str,
) -> Optional[float]:
    """Same as :func:`_phase_lerp_value` but for ``Optional[float]``."""
    if t_s <= phases[0].start_t_s:
        return phases[0].__dict__.get(attr)
    for i, ph in enumerate(phases):
        if ph.start_t_s <= t_s < ph.end_t_s:
            v0 = ph.__dict__.get(attr)
            if i + 1 >= len(phases):
                return v0
            next_ph = phases[i + 1]
            v1 = next_ph.__dict__.get(attr)
            t_in_phase = t_s - ph.start_t_s
            if t_in_phase <= ph.duration_s - ph.transition_s:
                return v0
            t_in_trans = t_in_phase - (ph.duration_s - ph.transition_s)
            if ph.transition_s <= 0.0:
                return v1
            u = max(0.0, min(1.0, t_in_trans / ph.transition_s))
            if v0 is None and v1 is None:
                return None
            if v0 is None:
                return v1
            if v1 is None:
                return v0
            return v0 + u * (v1 - v0)
    return phases[-1].__dict__.get(attr)


# ---------------------------------------------------------------------
# ProfileGenerator
# ---------------------------------------------------------------------
class ProfileGenerator:
    """Compile a phase list into a synchronised mission trace.

    Two construction paths:

    * ``ProfileGenerator(template, config)`` — use a built-in
      :class:`MissionTemplate` (NORMAL, HIGH_ALTITUDE, …).
    * ``ProfileGenerator(phases=..., config=...)`` — supply a
      custom list of :class:`MissionPhaseSpec`.

    Methods
    -------
    * :meth:`compile_profile` — build the :class:`MissionProfile`
      (the existing PHASE 2 waypoint list).
    * :meth:`iter_truth` — yield :class:`MissionTick` for the
      full mission (researcher view; includes ``fault_truth``).
    * :meth:`iter_observed` — yield :class:`ObservedTick` (AI
      view; no fault label).
    * :meth:`run` — run the whole mission and return a
      :class:`MissionTrace` (two parallel lists).
    """

    def __init__(
        self,
        template: Optional[MissionTemplate] = None,
        config: Optional[ProfileGeneratorConfig] = None,
        *,
        phases: Optional[Sequence[MissionPhaseSpec]] = None,
    ) -> None:
        if phases is not None:
            raw_phases = list(phases)
        elif template is not None:
            raw_phases = list(MISSION_TEMPLATES[template])
        else:
            raise ValueError(
                "ProfileGenerator requires either a template or a phases list"
            )
        if not raw_phases:
            raise ValueError("phase list is empty")
        self._config = config or ProfileGeneratorConfig()
        self._raw_phases = raw_phases
        self._phases: List[MissionPhaseSpec] = _resolve_phase_start_times(
            raw_phases, self._config.transition_default_s
        )
        # Sanity check: phases are in canonical order (startup → shutdown).
        kinds = [p.kind for p in self._phases]
        if kinds != list(PhaseKind.canonical_order()):
            raise ValueError(
                f"phases must be in canonical order, got: "
                f"{[k.value for k in kinds]}"
            )
        # Cache the total duration.
        self._duration_s = self._phases[-1].end_t_s

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def config(self) -> ProfileGeneratorConfig:
        return self._config

    @property
    def phases(self) -> List[MissionPhaseSpec]:
        return list(self._phases)

    @property
    def duration_s(self) -> float:
        return float(self._duration_s)

    # ------------------------------------------------------------------
    # Phase → MissionProfile
    # ------------------------------------------------------------------
    def compile_profile(self) -> MissionProfile:
        """Build the :class:`MissionProfile` (waypoint list) the
        existing :class:`MissionRunner` consumes.

        Each phase ``i`` contributes one waypoint at
        ``t = phase_i.end_t_s`` carrying the phase's *target*
        altitude / airspeed / throttle. The existing linear
        interpolation between consecutive waypoints then gives
        a smooth commanded ramp that spans each phase's full
        body — a 300 s climb phase, for instance, distributes
        a 5000 m climb over its 300 s body, well within the
        spec's 20 m/s continuity bound.

        The phase's ``transition_s`` field is *not* used to
        inject extra waypoints: the per-phase scalars that
        need a finer-grained ramp (engine_load, wind_mps,
        turbulence_intensity, ambient_temperature_c,
        pressure_pa) are interpolated separately by
        :func:`_phase_lerp_value` over the same window. The
        waypoint list stays simple and the mission is fully
        C⁰-continuous.

        The engine's existing first-order lags further smooth
        the *truth* state into a C¹-continuous observed value.
        """
        waypoints: List[Waypoint] = []
        # First waypoint: phase 0's target at its start.
        first = self._phases[0]
        waypoints.append(
            Waypoint(
                t_s=float(first.start_t_s),
                altitude_m=float(first.target_altitude_m),
                airspeed_mps=float(first.target_airspeed_mps),
                throttle=float(first.target_throttle),
            )
        )
        for ph in self._phases:
            waypoints.append(
                Waypoint(
                    t_s=float(ph.end_t_s),
                    altitude_m=float(ph.target_altitude_m),
                    airspeed_mps=float(ph.target_airspeed_mps),
                    throttle=float(ph.target_throttle),
                )
            )
        # MissionProfile needs at least 2 waypoints.
        if len(waypoints) < 2:
            raise ValueError("phase list produced fewer than 2 waypoints")
        # Build the MissionProfile directly from the waypoint list.
        # No need to construct a full ``Mission`` (which requires
        # ``initial_seed``); the profile is consumed by
        # MissionRunner, which only looks at the waypoints.
        return MissionProfile(waypoints)

    # ------------------------------------------------------------------
    # Internal: per-phase lookups
    # ------------------------------------------------------------------
    def _engine_load_at(self, t_s: float) -> float:
        return _phase_lerp_value(
            t_s, self._phases, "engine_load", fallback=0.0
        )

    def _wind_at(self, t_s: float) -> float:
        return _phase_lerp_value(
            t_s, self._phases, "wind_mps", fallback=0.0
        )

    def _turbulence_at(self, t_s: float) -> float:
        return _phase_lerp_value(
            t_s, self._phases, "turbulence_intensity", fallback=0.0
        )

    def _ambient_temp_at(self, t_s: float) -> Optional[float]:
        return _phase_lerp_optional(
            t_s, self._phases, "ambient_temperature_c"
        )

    def _pressure_at(self, t_s: float) -> Optional[float]:
        return _phase_lerp_optional(
            t_s, self._phases, "pressure_pa"
        )

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------
    def _build_env_config(self) -> EnvironmentConfig:
        """Return a deep-copied env config with the RNG seeds set
        to ``self._config.seed`` and the mission profile replaced
        by the phase-derived waypoint list.
        """
        # Lazy import: config layer is a sibling of environment.
        from backend.config import load_config

        loaded = load_config()
        # The engine/simulator/sensor layers want the inner
        # EnvironmentConfig, not the LoadedConfig wrapper.
        cfg = copy.deepcopy(loaded.environment)
        seed = int(self._config.seed)
        # Turbulence seed.
        cfg.disturbances.turbulence.seed = seed
        # Gust seed (offset to decorrelate from turbulence).
        cfg.disturbances.gusts.seed = seed + 1
        # Mission profile (overwrite) with the phase-derived
        # waypoint list. ``initial_seed`` is consumed by other
        # paths in the system; we re-use the master seed so the
        # mission is fully reproducible.
        cfg.mission = Mission(
            profile=list(
                Waypoint(
                    t_s=float(wp.t_s),
                    altitude_m=float(wp.altitude_m),
                    airspeed_mps=float(wp.airspeed_mps),
                    throttle=float(wp.throttle),
                )
                for wp in self.compile_profile().waypoints
            ),
            initial_seed=seed,
        )
        return cfg

    def _build_simulator(
        self,
        env_cfg: EnvironmentConfig,
    ) -> EngineSimulator:
        # EngineSimulator takes the engine config (from
        # LoadedConfig) and the env config separately. Load the
        # full config so we can hand the right pieces off.
        from backend.config import load_config
        from backend.faults import FaultClass, FaultInjector, FaultScenario
        from backend.sensors import SensorBundle

        loaded = load_config()
        sim = EngineSimulator(loaded.engine, env_cfg, dt_s=self._config.dt_s)
        # Build the sensor bundle around the simulator. The bundle
        # is the only place sensor noise / drift / dropout live.
        self._bundle = SensorBundle(loaded.sensors, sim, master_seed=self._config.seed)
        # Build the fault injector (healthy by default).
        if self._config.fault_plan is not None:
            base = self._config.fault_plan
            scenario = FaultScenario(
                fault_class=base.fault_class,
                severity=base.severity,
                onset_time_s=base.onset_time_s,
                duration_s=base.duration_s,
                progression=base.progression,
                seed=int(self._config.seed),
                target_channel=base.target_channel,
                target_sensor_mode=base.target_sensor_mode,
            )
        else:
            scenario = FaultScenario(
                fault_class=FaultClass.HEALTHY,
                seed=int(self._config.seed),
            )
        self._injector = FaultInjector(scenario)
        self._injector.attach(sim, self._bundle)
        return sim

    # ------------------------------------------------------------------
    # Step driver: produce one (env, engine, sensors, fault_truth) tuple
    # ------------------------------------------------------------------
    def _tick(
        self,
        sim: EngineSimulator,
        bundle: SensorBundle,
        t_s: float,
    ) -> Tuple[EnvironmentState, Any, Dict[str, SensorReading], Any, Any]:
        """Advance the simulation by one tick.

        Returns ``(env, engine_state, readings, fault_tick, fault_truth)``.

        * ``env`` is the modified :class:`EnvironmentState` with
          per-phase ``engine_load`` / ``wind`` / ``ambient`` /
          ``turbulence`` applied. The linear ``MissionProfile``
          interpolation already gives the right altitude /
          airspeed / throttle.
        * ``engine_state`` is the PHASE 3 :class:`EngineState`.
        * ``readings`` is the per-channel :class:`SensorReading`
          dict (observed).
        * ``fault_tick`` is the :class:`FaultTick` the injector
          wants the engine inputs to carry.
        * ``fault_truth`` is the :class:`FaultTruthRecord` the
          researcher reads.
        """
        # Lazy import — see note at the top of the file.
        from backend.sensors import SensorReading

        dt = sim.dt_s
        # 1) Per-phase scalars for this tick.
        engine_load = self._engine_load_at(t_s)
        steady_wind = self._wind_at(t_s)
        turb_intensity = self._turbulence_at(t_s)
        ambient_t = self._ambient_temp_at(t_s)
        pressure_pa = self._pressure_at(t_s)

        # 2) Mutate the runner's turbulence / gust models for the
        # active phase. The TYPING is a thin runtime knob — the
        # underlying RNGs are unchanged so determinism is preserved.
        runner = sim.runner
        if runner.turbulence.enabled:
            runner.turbulence.set_intensity(turb_intensity)
        if runner.gusts.enabled:
            # Steady-wind bias is a constant; we just set the gust
            # amplitude to 1.0 m/s baseline plus the steady bias.
            runner.gusts.set_amplitude(max(0.0, 1.0 + abs(steady_wind)))

        # 3) Build the wind state for this tick.
        wind = compute_wind(
            t_s, dt, runner.turbulence, runner.gusts,
            steady_wind_mps=steady_wind,
        )

        # 4) Atmosphere state. Use the per-phase ambient / pressure
        # if specified, otherwise the runner's atmosphere at the
        # commanded altitude.
        cmd_alt = sim.runner.profile.at(t_s).altitude_m
        if ambient_t is not None or pressure_pa is not None:
            # Build a one-off atmosphere state at the commanded alt
            # then patch in the per-phase overrides.
            atmos = runner.atmosphere.state(cmd_alt)
            if ambient_t is not None:
                # Recompute pressure from the ISA gradient at the
                # given temperature (ideal gas).
                # We keep the per-phase T_c override but stay
                # self-consistent: pressure is recomputed so the
                # gas law holds.
                from backend.environment.atmosphere import AtmosphericState
                t_k = float(ambient_t) + 273.15
                rho = atmos.pressure_pa / (287.05287 * t_k) if t_k > 0 else atmos.density_kg_per_m3
                speed = (1.4 * 287.05287 * max(t_k, 1.0)) ** 0.5
                atmos = AtmosphericState(
                    altitude_m=atmos.altitude_m,
                    pressure_pa=atmos.pressure_pa,
                    temperature_k=t_k,
                    temperature_c=float(ambient_t),
                    density_kg_per_m3=rho,
                    speed_of_sound_mps=speed,
                )
            if pressure_pa is not None:
                # Recompute density and temperature so the gas law
                # holds.
                from backend.environment.atmosphere import AtmosphericState
                t_k = atmos.temperature_k if atmos.temperature_k > 0 else 288.15
                rho = float(pressure_pa) / (287.05287 * t_k)
                speed = (1.4 * 287.05287 * t_k) ** 0.5
                atmos = AtmosphericState(
                    altitude_m=atmos.altitude_m,
                    pressure_pa=float(pressure_pa),
                    temperature_k=t_k,
                    temperature_c=atmos.temperature_c,
                    density_kg_per_m3=rho,
                    speed_of_sound_mps=speed,
                )
        else:
            atmos = runner.atmosphere.state(cmd_alt)

        # 5) Build the EnvironmentState the engine sees.
        cmd = sim.runner.profile.at(t_s)
        # Vertical accel proxy: difference of total wind, as in PHASE 2.
        # (Stored on the previous wind state — we approximate by
        # using wind.total_w_mps / dt, which is what MissionRunner
        # does internally.)
        env = EnvironmentState(
            time_s=float(t_s),
            altitude_m=float(cmd.altitude_m),
            airspeed_mps=float(cmd.airspeed_mps),
            throttle=float(cmd.throttle),
            atmosphere=atmos,
            wind=wind,
            vertical_accel_mps2=wind.total_w_mps / dt if dt > 0 else 0.0,
        )

        # 6) Engine inputs: fault layer (per FaultInjector) + per-phase
        # engine_load. The fault tick is the *engine-input* layer;
        # engine_load is the *mission* layer — they compose.
        fault_tick = self._injector.tick(t_s)
        engine_inputs = EngineInputs(
            env=env,
            degradation_severity=fault_tick.degradation_severity,
            vibration_external=fault_tick.vibration_external,
            engine_load=engine_load,
        )

        # 7) Step the engine directly (the simulator's step helper
        # uses its own internal env-step which would double-step the
        # runner). We bypass the simulator's step and call the
        # engine directly with our hand-built env.
        engine_state = sim.engine.step(engine_inputs)

        # 8) Read every sensor channel. The bundle holds the channels
        # but its ``tick()`` method also calls ``sim.step()``; we
        # therefore drive the channels directly to avoid double-step.
        readings: Dict[str, SensorReading] = {}
        for name, ch in bundle.channels.items():
            value, mode = ch.read(engine_state, env, t_s, dt)
            readings[name] = SensorReading(value=value, mode=mode)

        # 9) Apply sensor-side fault injection if the scenario says so.
        self._injector.apply_sensor(t_s)
        # Apply environment-layer fault.
        self._injector.apply_environment(t_s)

        # 10) Build the FaultTruthRecord the researcher reads.
        # Sensor fault details, if any.
        sensor_mode_value: Optional[str] = None
        target_channel: Optional[str] = None
        if self._injector.scenario.fault_class.value == "SENSOR_FAULT":
            plan = self._injector.sensor_plan
            if plan is not None:
                sensor_mode_value = plan.mode.value
                target_channel = plan.channel
        truth = FaultTruthRecord(
            fault_class=self._injector.scenario.fault_class.value,
            severity=float(self._injector.severity_plan.at(t_s)),
            sensor_mode=sensor_mode_value,
            target_channel=target_channel,
        )

        return env, engine_state, readings, fault_tick, truth

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------
    def _run_full(self) -> List[
        Tuple[EnvironmentState, Any, Dict[str, SensorReading], Any, Any]
    ]:
        env_cfg = self._build_env_config()
        sim = self._build_simulator(env_cfg)
        # Reset state.
        sim.reset()
        self._bundle.reset()
        self._injector.reset()
        out: List[
            Tuple[EnvironmentState, Any, Dict[str, SensorReading], Any, Any]
        ] = []
        # Walk the mission in fixed dt steps.
        n_steps = int(math.ceil(self._duration_s / self._config.dt_s))
        for i in range(n_steps):
            t_s = i * self._config.dt_s
            out.append(self._tick(sim, self._bundle, t_s))
        return out

    def iter_truth(self) -> Iterator[MissionTick]:
        """Yield :class:`MissionTick` for the full mission (researcher view)."""
        # Run-once memoization — both iter_truth and iter_observed
        # share the same underlying state.
        if not hasattr(self, "_cache"):
            self._cache = self._run_full()
        for env, engine_state, readings, _fault_tick, truth in self._cache:
            yield MissionTick(
                time_s=env.time_s,
                env=env,
                engine=engine_state,
                sensors=readings,
                fault_truth=truth,
            )

    def iter_observed(self) -> Iterator[ObservedTick]:
        """Yield :class:`ObservedTick` (AI view; no fault label)."""
        if not hasattr(self, "_cache"):
            self._cache = self._run_full()
        for env, engine_state, readings, _fault_tick, _truth in self._cache:
            yield ObservedTick(
                time_s=env.time_s,
                env=env,
                engine=engine_state,
                sensors=readings,
            )

    def run(self) -> MissionTrace:
        """Run the whole mission and return a :class:`MissionTrace`."""
        if not hasattr(self, "_cache"):
            self._cache = self._run_full()
        truth_list: List[MissionTick] = []
        observed_list: List[ObservedTick] = []
        for env, engine_state, readings, _fault_tick, truth in self._cache:
            truth_list.append(
                MissionTick(
                    time_s=env.time_s,
                    env=env,
                    engine=engine_state,
                    sensors=readings,
                    fault_truth=truth,
                )
            )
            observed_list.append(
                ObservedTick(
                    time_s=env.time_s,
                    env=env,
                    engine=engine_state,
                    sensors=readings,
                )
            )
        return MissionTrace(
            truth=tuple(truth_list),
            observed=tuple(observed_list),
        )


__all__ = [
    "ProfileGenerator",
    "ProfileGeneratorConfig",
]
