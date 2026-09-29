"""
Representative aero-piston engine simulator.

This is the **truth model** the Digital Twin (PHASE 6) will compare
against. It uses the same performance maps as the twin but also
integrates:

* Throttle -> MAP transfer (with idle floor and dead zone)
* First-order lag on RPM, EGT, CHT, oil temperature, oil pressure
* A wear accumulator that biases EGT, CHT, vibration and BSFC
* Vibration baseline as a function of RPM (engine-only component)
* Optional environment disturbance coupling (vertical accel)

The engine has no notion of sensor noise — that's the job of PHASE 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from backend.config import EngineConfig
from backend.environment import EnvironmentState

from .interp import map_evaluator


# ---------------------------------------------------------------------
# Public state containers
# ---------------------------------------------------------------------
@dataclass
class EngineInputs:
    """External inputs consumed by the engine each step."""

    env: EnvironmentState
    degradation_severity: float = 0.0   # externally driven (0..1); see PHASE 7
    vibration_external: float = 0.0     # additive, environment-driven
    # PHASE 20 — mission-level engine load in [0, 1]. A scalar
    # representing commanded accessory / generator / external-store
    # load. Biases vibration, BSFC, and fuel flow with small gains
    # so the existing test suite is unaffected when left at 0.0.
    engine_load: float = 0.0


@dataclass
class DegradationState:
    """Internal wear / health state."""

    wear: float = 0.0            # normalised, 0 = new, 1 = end-of-life
    accumulated_hours: float = 0.0
    last_step_dt: float = 0.0


@dataclass
class EngineState:
    """Snapshot of the engine at one time step."""

    time_s: float
    rpm: float
    manifold_pressure_inhg: float
    egt_c: float
    cht_c: float
    oil_pressure_psi: float
    oil_temperature_c: float
    fuel_flow_lph: float
    vibration_rms_g: float
    bsfc_g_per_kwh: float
    brake_power_kw: float
    wear: float
    throttle: float
    altitude_m: float
    airspeed_mps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "time_s": self.time_s,
            "rpm": self.rpm,
            "manifold_pressure_inhg": self.manifold_pressure_inhg,
            "egt_c": self.egt_c,
            "cht_c": self.cht_c,
            "oil_pressure_psi": self.oil_pressure_psi,
            "oil_temperature_c": self.oil_temperature_c,
            "fuel_flow_lph": self.fuel_flow_lph,
            "vibration_rms_g": self.vibration_rms_g,
            "bsfc_g_per_kwh": self.bsfc_g_per_kwh,
            "brake_power_kw": self.brake_power_kw,
            "wear": self.wear,
            "throttle": self.throttle,
            "altitude_m": self.altitude_m,
            "airspeed_mps": self.airspeed_mps,
        }


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
class Engine:
    """Deterministic, reproducible representative aero-piston engine."""

    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg
        # Compiled evaluators for the 2D performance maps.
        pm = cfg.performance_maps
        self._bsfc_eval = map_evaluator(
            pm["bsfc_g_per_kwh"]["rpm_axis"],
            pm["bsfc_g_per_kwh"]["map_axis"],
            pm["bsfc_g_per_kwh"]["values"],
        )
        self._egt_eval = map_evaluator(
            pm["egt_map_c"]["rpm_axis"],
            pm["egt_map_c"]["map_axis"],
            pm["egt_map_c"]["values"],
        )
        self._cht_eval = map_evaluator(
            pm["cht_map_c"]["rpm_axis"],
            pm["cht_map_c"]["map_axis"],
            pm["cht_map_c"]["values"],
        )

        geom = cfg.geometry
        self._idle_rpm = geom.idle_rpm
        self._redline_rpm = geom.redline_rpm
        self._rated_rpm = geom.rated_rpm
        self._rated_power_kw = geom.rated_power_kw
        self._max_torque_rpm = self._max_torque_rpm_estimate()

        # Throttle -> MAP.
        tm = cfg.throttle_to_map
        self._map_gain = tm.gain_inhg_per_unit
        self._map_idle = tm.idle_map_inhg
        self._map_max = tm.max_map_inhg
        self._map_dead = tm.dead_zone

        # First-order lags.
        self._tau_rpm = cfg.dynamics.rpm_time_constant_s
        self._tau_thermal = cfg.dynamics.thermal_time_constant_s
        self._tau_oil = cfg.dynamics.oil_time_constant_s

        # Lubrication.
        lub = cfg.lubrication
        self._oil_nominal = lub.nominal_pressure_psi
        self._oil_rpm_slope = lub.pressure_rpm_slope
        self._oil_temp_drop_per_c = lub.pressure_temp_drop_per_c

        # Degradation (internal).
        deg = cfg.degradation
        self._deg_base = deg.base_wear_per_hour
        self._deg_temp_factor = deg.wear_temp_factor_per_c
        self._deg_rpm_factor = deg.wear_rpm_factor_per_rpm
        self._deg_max = deg.max_wear

        # Vibration baseline parameters (SYNTHETIC, prototype envelope).
        # Vibration rises with RPM and gets a small bump near max torque.
        self._vib_idle_g = 0.4
        self._vib_full_g = 3.0

        # State.
        self._t = 0.0
        self._dt = 0.1
        self._rpm = self._idle_rpm
        self._map = self._map_idle
        self._egt = self._egt_eval(self._idle_rpm, self._map_idle)
        self._cht = self._cht_eval(self._idle_rpm, self._map_idle)
        self._oil_t = 60.0
        self._oil_p = self._oil_nominal
        self._wear = DegradationState()
        self._last_state: Optional[EngineState] = None
        # Sea-level reference density, supplied by the caller (it lives in
        # the environment config, not the engine config). Until set, the
        # altitude correction is a no-op.
        self._ref_density = 1.225

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------
    @property
    def wear(self) -> DegradationState:
        return self._wear

    @property
    def last_state(self) -> Optional[EngineState]:
        return self._last_state

    @property
    def dt_s(self) -> float:
        return self._dt

    def set_dt(self, dt_s: float) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self._dt = float(dt_s)

    def set_reference_density(self, rho_kg_per_m3: float) -> None:
        """Provide a sea-level reference density for altitude corrections.

        The engine has no opinion on the atmosphere; the caller (the
        mission runner) supplies it once at start-up. A value <= 0
        disables the altitude correction.
        """
        if rho_kg_per_m3 <= 0:
            self._ref_density = 0.0
        else:
            self._ref_density = float(rho_kg_per_m3)

    def reset(self) -> None:
        self._t = 0.0
        self._rpm = self._idle_rpm
        self._map = self._map_idle
        self._egt = self._egt_eval(self._idle_rpm, self._map_idle)
        self._cht = self._cht_eval(self._idle_rpm, self._map_idle)
        self._oil_t = 60.0
        self._oil_p = self._oil_nominal
        self._wear = DegradationState()
        self._last_state = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _max_torque_rpm_estimate(self) -> float:
        """Approximate RPM of peak torque from rated/max RPM.

        Most aero pistons peak torque somewhere between idle and rated.
        0.75 * rated is a reasonable SYNTHETIC default; the absolute
        value is not used as a published engine characteristic.
        """
        return 0.75 * self._rated_rpm

    def _throttle_to_map(self, throttle: float) -> float:
        """Static throttle -> manifold pressure (inHg) transfer.

        Below the dead-zone the throttle reads as idle. Above the dead
        zone, the response is a linear ramp from ``idle_map_inhg`` to
        ``max_map_inhg`` scaled by ``gain_inhg_per_unit``.
        """
        thr = max(0.0, min(1.0, float(throttle)))
        if thr < self._map_dead:
            return self._map_idle
        u = (thr - self._map_dead) / max(1e-9, 1.0 - self._map_dead)
        span = max(0.0, self._map_max - self._map_idle)
        return self._map_idle + u * self._map_gain * span

    def _rpm_target(self, throttle: float) -> float:
        """Steady-state RPM for a given throttle (idle .. rated)."""
        thr = max(0.0, min(1.0, float(throttle)))
        return self._idle_rpm + thr * (self._rated_rpm - self._idle_rpm)

    def _vibration_baseline(self, rpm: float) -> float:
        """Engine-only vibration (g_rms), SYNTHETIC model."""
        x = max(0.0, min(1.0, (rpm - self._idle_rpm) / max(1.0, self._redline_rpm - self._idle_rpm)))
        # Smooth ramp with a small bump near max torque.
        torque_x = (rpm - self._max_torque_rpm) / max(1.0, self._redline_rpm - self._idle_rpm)
        bump = 0.15 * math.exp(-(torque_x ** 2) * 12.0)
        return self._vib_idle_g + (self._vib_full_g - self._vib_idle_g) * x + bump

    def _step_degradation(
        self,
        dt: float,
        egt: float,
        cht: float,
        rpm: float,
        severity: float = 0.0,
    ) -> None:
        """Advance the wear accumulator using Arrhenius-like temperature factor."""
        # Use average of EGT and CHT as a proxy for thermal stress.
        t_heat = max(0.0, 0.5 * (egt + cht))
        d_wear = self._deg_base * (
            1.0
            + self._deg_temp_factor * t_heat
            + self._deg_rpm_factor * rpm
        )
        # External severity (from PHASE 7 fault injection) accelerates
        # wear. Even if a high-severity engine runs cooler due to lost
        # power, the wear rate itself must still increase.
        d_wear *= 1.0 + 5.0 * float(severity)
        d_wear = d_wear * dt / 3600.0     # base_wear_per_hour -> per second
        # Cap the result by the configurable max wear.
        self._wear.wear = min(self._deg_max, self._wear.wear + d_wear)
        self._wear.accumulated_hours += dt / 3600.0
        self._wear.last_step_dt = dt

    def _step_lubrication(
        self, dt: float, rpm: float, oil_t: float, cht: float
    ) -> tuple[float, float]:
        """Update oil pressure and temperature with first-order dynamics."""
        # Pressure: rises with RPM, drops with oil temperature (oil thins).
        target_p = (
            self._oil_nominal
            + self._oil_rpm_slope * (rpm - self._idle_rpm)
            - self._oil_temp_drop_per_c * max(0.0, oil_t - 60.0)
        )
        target_p = max(0.0, target_p)
        alpha_p = 1.0 - math.exp(-dt / max(1e-3, 5.0))  # fast pressure response
        new_p = self._oil_p + alpha_p * (target_p - self._oil_p)

        # Oil temperature tracks CHT with a long lag.
        target_t = max(20.0, cht - 60.0)
        alpha_t = 1.0 - math.exp(-dt / self._tau_oil)
        new_t = self._oil_t + alpha_t * (target_t - self._oil_t)

        return new_p, new_t

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, inputs: EngineInputs) -> EngineState:
        env = inputs.env
        dt = self._dt
        self._t = env.time_s

        # 1) Throttle -> MAP (instantaneous; mechanical actuator).
        map_target = self._throttle_to_map(env.throttle)

        # 2) RPM first-order lag toward the throttle target.
        rpm_target = self._rpm_target(env.throttle)
        alpha_rpm = 1.0 - math.exp(-dt / self._tau_rpm)
        # Reduce effective RPM by degradation severity ("rpm_efficiency_drop"
        # in faults.yaml).
        rpm_efficiency = 1.0 - 0.20 * inputs.degradation_severity
        self._rpm = self._rpm + alpha_rpm * (rpm_target * rpm_efficiency - self._rpm)
        self._rpm = max(self._idle_rpm, min(self._redline_rpm, self._rpm))

        # 3) MAP follows RPM + throttle demand (small load-dependent drop).
        self._map = max(0.0, map_target - 0.05 * (rpm_target - self._rpm))

        # 4) EGT / CHT from maps, biased by wear and degradation severity.
        egt_target = self._egt_eval(self._rpm, self._map)
        cht_target = self._cht_eval(self._rpm, self._map)
        wear_bias = 20.0 * self._wear.wear + 60.0 * inputs.degradation_severity
        cht_bias = 5.0 * self._wear.wear + 30.0 * inputs.degradation_severity
        egt_target += wear_bias
        cht_target += cht_bias

        alpha_th = 1.0 - math.exp(-dt / self._tau_thermal)
        self._egt = self._egt + alpha_th * (egt_target - self._egt)
        self._cht = self._cht + alpha_th * (cht_target - self._cht)

        # 5) Lubrication.
        self._oil_p, self._oil_t = self._step_lubrication(
            dt, self._rpm, self._oil_t, self._cht
        )

        # 6) BSFC and brake power.
        bsfc_target = self._bsfc_eval(self._rpm, self._map)
        bsfc_target *= 1.0 + 0.15 * self._wear.wear + 0.05 * inputs.degradation_severity
        # PHASE 20 — engine load: small extra BSFC bias.
        bsfc_target *= 1.0 + 0.10 * inputs.engine_load
        # Air density correction: lower density at altitude -> richer mixture.
        # (Very rough — kept SYNTHETIC and bounded.)
        if self._ref_density > 0:
            rho_ratio = env.atmosphere.density_kg_per_m3 / self._ref_density
            bsfc_target *= max(0.8, min(1.3, 1.0 / rho_ratio ** 0.2))

        # Brake power: fraction of rated, scaled by MAP and RPM relative to rated.
        map_frac = max(0.0, min(1.0, (self._map - self._map_idle) / max(1.0, self._map_max - self._map_idle)))
        rpm_frac = max(0.0, min(1.0, (self._rpm - self._idle_rpm) / max(1.0, self._rated_rpm - self._idle_rpm)))
        power_frac = 0.7 * map_frac + 0.3 * rpm_frac
        brake_power_kw = self._rated_power_kw * power_frac

        # Fuel flow: BSFC (g/kWh) * power (kW) = g/h, convert to L/h.
        fuel_flow_gph = bsfc_target * max(0.0, brake_power_kw)
        # PHASE 20 — engine load: small extra fuel-flow bias.
        fuel_flow_gph *= 1.0 + 0.05 * inputs.engine_load
        fuel_density_g_per_l = 720.0   # avgas approx (PUBLIC_DATA)
        fuel_flow_lph = fuel_flow_gph / fuel_density_g_per_l

        # 7) Vibration: engine baseline + wear + external disturbance.
        vibration = self._vibration_baseline(self._rpm)
        vibration *= 1.0 + 0.6 * self._wear.wear + 0.3 * inputs.degradation_severity
        # PHASE 20 — engine load: small extra vibration bias.
        vibration *= 1.0 + 0.4 * inputs.engine_load
        vibration += inputs.vibration_external

        # 8) Degradation integrator.
        self._step_degradation(dt, self._egt, self._cht, self._rpm, inputs.degradation_severity)

        state = EngineState(
            time_s=self._t,
            rpm=self._rpm,
            manifold_pressure_inhg=self._map,
            egt_c=self._egt,
            cht_c=self._cht,
            oil_pressure_psi=self._oil_p,
            oil_temperature_c=self._oil_t,
            fuel_flow_lph=fuel_flow_lph,
            vibration_rms_g=vibration,
            bsfc_g_per_kwh=bsfc_target,
            brake_power_kw=brake_power_kw,
            wear=self._wear.wear,
            throttle=env.throttle,
            altitude_m=env.altitude_m,
            airspeed_mps=env.airspeed_mps,
        )
        self._last_state = state
        return state

    # ------------------------------------------------------------------
    # Convenience: run a list of environment states
    # ------------------------------------------------------------------
    def run(
        self,
        env_states: List[EnvironmentState],
        degradation_severity: float = 0.0,
        vibration_external: Optional[List[float]] = None,
    ) -> List[EngineState]:
        """Run the engine over a list of environment states."""
        if vibration_external is not None and len(vibration_external) != len(env_states):
            raise ValueError("vibration_external length must match env_states")
        self.reset()
        out: List[EngineState] = []
        for i, env in enumerate(env_states):
            ve = 0.0 if vibration_external is None else float(vibration_external[i])
            out.append(
                self.step(EngineInputs(env=env, degradation_severity=degradation_severity, vibration_external=ve))
            )
        return out


__all__ = ["DegradationState", "Engine", "EngineInputs", "EngineState"]
