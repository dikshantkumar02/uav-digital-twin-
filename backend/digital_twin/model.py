"""
Digital Twin — predicted engine state per tick.

The twin is a *physics-driven estimator* that mirrors the truth model
(PHASE 3) using:

* the same performance maps (BSFC, EGT, CHT) for first-principles prediction
* the same throttle -> MAP transfer
* the same first-order dynamics on RPM, EGT, CHT, oil temperature,
  oil pressure
* a closed-loop correction: each tick, observed values (when valid)
  are blended into the predicted state with a small Kalman-like gain
  so the twin *tracks* the engine without copying sensor noise

The twin does **not** see:

* the truth model's wear accumulator
* the externally-driven ``degradation_severity`` input
* the externally-supplied ``vibration_external`` term

That is the point: whatever deviation remains between twin and truth
is a *residual* — the diagnostic engine (PHASE 8+) interprets it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.config import EngineConfig
from backend.environment import EnvironmentState
from backend.simulation import EngineState
from backend.simulation.interp import map_evaluator

from .residual import CHANNEL_TO_STATE, ChannelResidual, ResidualFrame


# ---------------------------------------------------------------------
# Predicted state
# ---------------------------------------------------------------------
@dataclass
class TwinState:
    """Predicted engine state at one tick (no ground truth)."""

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
    altitude_m: float
    airspeed_mps: float
    ambient_temperature_c: float
    ambient_pressure_pa: float
    confidence: float = 1.0

    def to_dict(self) -> dict:
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
            "altitude_m": self.altitude_m,
            "airspeed_mps": self.airspeed_mps,
            "ambient_temperature_c": self.ambient_temperature_c,
            "ambient_pressure_pa": self.ambient_pressure_pa,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------
# Default expected per-channel noise levels (sigma).
# These are used to compute z-scores. They are intentionally close to
# the sensor noise_std from config so that a healthy engine + nominal
# noise yields |z| < 1.
# ---------------------------------------------------------------------
DEFAULT_EXPECTED_SIGMA: Dict[str, float] = {
    "rpm": 30.0,
    "egt": 25.0,
    "cht": 15.0,
    "oil_pressure": 3.0,
    "oil_temperature": 3.0,
    "fuel_flow": 5.0,
    "vibration": 0.5,
    "altitude": 10.0,
    "airspeed": 3.0,
    "ambient_temperature": 2.0,
    "ambient_pressure": 600.0,
}


# ---------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------
class DigitalTwin:
    """The Digital Twin itself."""

    def __init__(
        self,
        cfg: EngineConfig,
        dt_s: float = 0.1,
        closed_loop_gain: float = 0.15,
        ref_density: float = 1.225,
    ) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if not 0.0 <= closed_loop_gain <= 1.0:
            raise ValueError("closed_loop_gain must be in [0, 1]")
        self._cfg = cfg
        self._dt = float(dt_s)
        self._gain = float(closed_loop_gain)
        self._ref_density = float(ref_density)

        # Map evaluators (same as the truth model).
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
        self._max_torque_rpm = 0.75 * self._rated_rpm

        tm = cfg.throttle_to_map
        self._map_gain = tm.gain_inhg_per_unit
        self._map_idle = tm.idle_map_inhg
        self._map_max = tm.max_map_inhg
        self._map_dead = tm.dead_zone

        self._tau_rpm = cfg.dynamics.rpm_time_constant_s
        self._tau_thermal = cfg.dynamics.thermal_time_constant_s
        self._tau_oil = cfg.dynamics.oil_time_constant_s

        lub = cfg.lubrication
        self._oil_nominal = lub.nominal_pressure_psi
        self._oil_rpm_slope = lub.pressure_rpm_slope
        self._oil_temp_drop_per_c = lub.pressure_temp_drop_per_c

        self._vib_idle_g = 0.4
        self._vib_full_g = 3.0

        # Initial state.
        self._t = 0.0
        self._rpm = self._idle_rpm
        self._map = self._map_idle
        self._egt = self._egt_eval(self._idle_rpm, self._map_idle)
        self._cht = self._cht_eval(self._idle_rpm, self._map_idle)
        self._oil_t = 60.0
        self._oil_p = self._oil_nominal
        self._last_state: Optional[TwinState] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def dt_s(self) -> float:
        return self._dt

    @property
    def last_state(self) -> Optional[TwinState]:
        return self._last_state

    def set_reference_density(self, rho_kg_per_m3: float) -> None:
        if rho_kg_per_m3 > 0:
            self._ref_density = float(rho_kg_per_m3)

    def reset(self) -> None:
        self._t = 0.0
        self._rpm = self._idle_rpm
        self._map = self._map_idle
        self._egt = self._egt_eval(self._idle_rpm, self._map_idle)
        self._cht = self._cht_eval(self._idle_rpm, self._map_idle)
        self._oil_t = 60.0
        self._oil_p = self._oil_nominal
        self._last_state = None

    # ------------------------------------------------------------------
    # Internal helpers (mirror the truth model, but no wear / severity)
    # ------------------------------------------------------------------
    def _throttle_to_map(self, throttle: float) -> float:
        thr = max(0.0, min(1.0, float(throttle)))
        if thr < self._map_dead:
            return self._map_idle
        u = (thr - self._map_dead) / max(1e-9, 1.0 - self._map_dead)
        span = max(0.0, self._map_max - self._map_idle)
        return self._map_idle + u * self._map_gain * span

    def _rpm_target(self, throttle: float) -> float:
        thr = max(0.0, min(1.0, float(throttle)))
        return self._idle_rpm + thr * (self._rated_rpm - self._idle_rpm)

    def _vibration_baseline(self, rpm: float) -> float:
        x = max(0.0, min(1.0, (rpm - self._idle_rpm) / max(1.0, self._redline_rpm - self._idle_rpm)))
        torque_x = (rpm - self._max_torque_rpm) / max(1.0, self._redline_rpm - self._idle_rpm)
        bump = 0.15 * math.exp(-(torque_x ** 2) * 12.0)
        return self._vib_idle_g + (self._vib_full_g - self._vib_idle_g) * x + bump

    def _step_lubrication(self, dt: float, rpm: float, oil_t: float, cht: float) -> tuple[float, float]:
        target_p = (
            self._oil_nominal
            + self._oil_rpm_slope * (rpm - self._idle_rpm)
            - self._oil_temp_drop_per_c * max(0.0, oil_t - 60.0)
        )
        target_p = max(0.0, target_p)
        alpha_p = 1.0 - math.exp(-dt / max(1e-3, 5.0))
        new_p = self._oil_p + alpha_p * (target_p - self._oil_p)
        target_t = max(20.0, cht - 60.0)
        alpha_t = 1.0 - math.exp(-dt / self._tau_oil)
        new_t = self._oil_t + alpha_t * (target_t - self._oil_t)
        return new_p, new_t

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def predict(self, env: EnvironmentState) -> TwinState:
        """One open-loop step. No observations are used.

        This is the *pure* prediction; useful for "what should the
        engine be doing given this command?" questions. The closed-loop
        :meth:`step` method additionally nudges state toward observations.
        """
        dt = self._dt
        self._t = env.time_s

        map_target = self._throttle_to_map(env.throttle)
        rpm_target = self._rpm_target(env.throttle)
        alpha_rpm = 1.0 - math.exp(-dt / self._tau_rpm)
        self._rpm = self._rpm + alpha_rpm * (rpm_target - self._rpm)
        self._rpm = max(self._idle_rpm, min(self._redline_rpm, self._rpm))
        self._map = max(0.0, map_target - 0.05 * (rpm_target - self._rpm))

        egt_target = self._egt_eval(self._rpm, self._map)
        cht_target = self._cht_eval(self._rpm, self._map)
        alpha_th = 1.0 - math.exp(-dt / self._tau_thermal)
        self._egt = self._egt + alpha_th * (egt_target - self._egt)
        self._cht = self._cht + alpha_th * (cht_target - self._cht)

        self._oil_p, self._oil_t = self._step_lubrication(
            dt, self._rpm, self._oil_t, self._cht
        )

        bsfc = self._bsfc_eval(self._rpm, self._map)
        if self._ref_density > 0:
            rho_ratio = env.atmosphere.density_kg_per_m3 / self._ref_density
            bsfc *= max(0.8, min(1.3, 1.0 / rho_ratio ** 0.2))

        map_frac = max(0.0, min(1.0, (self._map - self._map_idle) / max(1.0, self._map_max - self._map_idle)))
        rpm_frac = max(0.0, min(1.0, (self._rpm - self._idle_rpm) / max(1.0, self._rated_rpm - self._idle_rpm)))
        power_frac = 0.7 * map_frac + 0.3 * rpm_frac
        brake_power_kw = self._rated_power_kw * power_frac
        fuel_flow_lph = bsfc * max(0.0, brake_power_kw) / 720.0  # avgas density

        vibration = self._vibration_baseline(self._rpm)

        state = TwinState(
            time_s=self._t,
            rpm=self._rpm,
            manifold_pressure_inhg=self._map,
            egt_c=self._egt,
            cht_c=self._cht,
            oil_pressure_psi=self._oil_p,
            oil_temperature_c=self._oil_t,
            fuel_flow_lph=fuel_flow_lph,
            vibration_rms_g=vibration,
            bsfc_g_per_kwh=bsfc,
            brake_power_kw=brake_power_kw,
            altitude_m=env.altitude_m,
            airspeed_mps=env.airspeed_mps,
            ambient_temperature_c=env.atmosphere.temperature_c,
            ambient_pressure_pa=env.atmosphere.pressure_pa,
            confidence=1.0,
        )
        self._last_state = state
        return state

    def step(
        self,
        env: EnvironmentState,
        observations: Optional[Dict[str, float]] = None,
    ) -> TwinState:
        """One closed-loop step.

        ``observations`` is an optional mapping channel -> observed value.
        For each observation that is present, the twin's internal state
        is nudged by ``closed_loop_gain`` toward the observation. The
        residual is therefore the *un-modelled* part — the part the
        twin could not have predicted from physics + command alone.
        """
        pred = self.predict(env)
        if observations:
            # Apply gentle closed-loop correction.
            self._apply_observations(pred, observations)
            # Reduce confidence slightly per corrected channel.
            n_obs = sum(1 for k in observations if k in CHANNEL_TO_STATE)
            pred.confidence = max(0.0, 1.0 - 0.05 * n_obs)
        return pred

    def _apply_observations(
        self,
        state: TwinState,
        observations: Dict[str, float],
    ) -> None:
        """Nudge internal state toward observed values.

        Only the *slow* physical channels (RPM, EGT, CHT, oil_p, oil_t,
        fuel_flow, vibration, altitude, airspeed, ambient_*) are
        absorbed. The twin's internal _rpm / _egt / _cht etc. are
        adjusted so the next prediction starts from the corrected
        state, not the original predicted state.
        """
        g = self._gain
        for channel, value in observations.items():
            field = CHANNEL_TO_STATE.get(channel)
            if field is None:
                continue
            # Nudge the corresponding internal state by gain * (obs - pred).
            current = getattr(state, field, None)
            if current is None:
                continue
            corrected = (1.0 - g) * float(current) + g * float(value)
            setattr(state, field, corrected)
            # Also nudge the underlying internal state where it makes sense.
            if field == "rpm" and self._idle_rpm <= corrected <= self._redline_rpm:
                self._rpm = corrected
            elif field == "egt_c":
                self._egt = corrected
            elif field == "cht_c":
                self._cht = corrected
            elif field == "oil_pressure_psi":
                self._oil_p = corrected
            elif field == "oil_temperature_c":
                self._oil_t = corrected
            elif field == "altitude_m":
                # Twin doesn't carry altitude as an internal state.
                pass
            elif field == "airspeed_mps":
                pass

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------
    def residual(
        self,
        state: TwinState,
        observations: Dict[str, float],
        expected_sigma: Optional[Dict[str, float]] = None,
    ) -> ResidualFrame:
        """Compute a residual frame for the given twin state + observations."""
        sigmas = {**DEFAULT_EXPECTED_SIGMA, **(expected_sigma or {})}
        frame = ResidualFrame(time_s=state.time_s)
        conf_sum = 0.0
        n = 0
        for channel, field_name in CHANNEL_TO_STATE.items():
            obs = observations.get(channel)
            pred = getattr(state, field_name, None)
            if obs is None or pred is None:
                continue
            sigma = sigmas.get(channel, 1.0)
            res = float(obs) - float(pred)
            z = res / sigma if sigma > 0 else 0.0
            # Confidence: shrinks with |z| but never collapses to zero.
            conf = max(0.0, 1.0 - abs(z) / 6.0)
            in_bounds = abs(z) <= 3.0
            frame.residuals[channel] = ChannelResidual(
                channel=channel,
                observed=float(obs),
                predicted=float(pred),
                residual=res,
                z_score=z,
                confidence=conf,
                in_bounds=in_bounds,
            )
            conf_sum += conf
            n += 1
        if n:
            frame.overall_confidence = conf_sum / n
        return frame

    # ------------------------------------------------------------------
    # Convenience: compare against a known engine state (for testing)
    # ------------------------------------------------------------------
    def twin_vs_truth_mae(
        self,
        twin_states: List[TwinState],
        truth_states: List[EngineState],
    ) -> Dict[str, float]:
        """Mean absolute error per channel across a list of (twin, truth) pairs."""
        if len(twin_states) != len(truth_states) or not twin_states:
            return {}
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        truth_by_field = {
            "rpm": "rpm",
            "egt_c": "egt_c",
            "cht_c": "cht_c",
            "oil_pressure_psi": "oil_pressure_psi",
            "oil_temperature_c": "oil_temperature_c",
            "fuel_flow_lph": "fuel_flow_lph",
            "vibration_rms_g": "vibration_rms_g",
            "altitude_m": "altitude_m",
            "airspeed_mps": "airspeed_mps",
        }
        for ts, es in zip(twin_states, truth_states):
            for twin_field, truth_field in truth_by_field.items():
                tv = getattr(ts, twin_field, None)
                ev = getattr(es, truth_field, None)
                if tv is None or ev is None:
                    continue
                sums[truth_field] = sums.get(truth_field, 0.0) + abs(float(tv) - float(ev))
                counts[truth_field] = counts.get(truth_field, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}


__all__ = [
    "DEFAULT_EXPECTED_SIGMA",
    "DigitalTwin",
    "TwinState",
]
