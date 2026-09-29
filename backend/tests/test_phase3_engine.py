"""PHASE 3 tests — representative piston engine model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from backend.config import EngineConfig, EnvironmentConfig, load_config
from backend.environment import EnvironmentState
from backend.simulation import (
    DegradationState,
    Engine,
    EngineInputs,
    EngineSimulator,
    EngineState,
    SimulatorStep,
    bilinear_interp,
    interp_axis,
    map_evaluator,
)

pytestmark = pytest.mark.phase3

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _engine_cfg() -> EngineConfig:
    return load_config(CONFIG_DIR).engine


def _env_cfg() -> EnvironmentConfig:
    return load_config(CONFIG_DIR).environment


def _env_state(t: float = 0.0, alt: float = 0.0, airspeed: float = 0.0, throttle: float = 0.0) -> EnvironmentState:
    """Build a minimal EnvironmentState directly — no runner needed.

    The engine only reads (altitude, airspeed, throttle, atmosphere, wind)
    from the environment. The atmosphere is derived from altitude via the
    ISA1976 model, so we can construct a fully self-consistent state
    without stepping the mission runner.
    """
    from backend.environment import Atmosphere
    from backend.environment.state import WindState

    atmos = Atmosphere(_env_cfg().atmosphere).state(alt)
    wind = WindState(
        time_s=t,
        turbulence_w_mps=0.0,
        gust_w_mps=0.0,
        total_w_mps=0.0,
        is_gust_active=False,
    )
    return EnvironmentState(
        time_s=t,
        altitude_m=alt,
        airspeed_mps=airspeed,
        throttle=throttle,
        atmosphere=atmos,
        wind=wind,
        vertical_accel_mps2=0.0,
    )


# ---------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------
class TestInterpolation:
    def test_axis_clamp_below(self) -> None:
        i, j, u = interp_axis([0.0, 1.0, 2.0], -0.5)
        assert (i, j, u) == (0, 0, 0.0)

    def test_axis_clamp_above(self) -> None:
        i, j, u = interp_axis([0.0, 1.0, 2.0], 5.0)
        assert (i, j, u) == (2, 2, 0.0)

    def test_axis_midpoint(self) -> None:
        i, j, u = interp_axis([0.0, 1.0, 2.0], 0.5)
        assert (i, j) == (0, 1)
        assert u == pytest.approx(0.5)

    def test_bilinear_corners_match_table(self) -> None:
        table = [[10.0, 20.0], [30.0, 40.0]]
        assert bilinear_interp([0, 1], [0, 1], table, 0, 0) == 10.0
        assert bilinear_interp([0, 1], [0, 1], table, 0, 1) == 20.0
        assert bilinear_interp([0, 1], [0, 1], table, 1, 0) == 30.0
        assert bilinear_interp([0, 1], [0, 1], table, 1, 1) == 40.0

    def test_bilinear_midpoint_is_average(self) -> None:
        table = [[0.0, 1.0], [2.0, 3.0]]
        # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 ; midpoint = 1.5
        assert bilinear_interp([0, 1], [0, 1], table, 0.5, 0.5) == pytest.approx(1.5)

    def test_evaluator_matches_direct_call(self) -> None:
        table = [[10.0, 20.0], [30.0, 40.0]]
        f = map_evaluator([0, 1], [0, 1], table)
        assert f(0.5, 0.5) == bilinear_interp([0, 1], [0, 1], table, 0.5, 0.5)


# ---------------------------------------------------------------------
# Throttle / MAP / RPM
# ---------------------------------------------------------------------
class TestThrottleToMAP:
    def test_below_deadzone_is_idle(self) -> None:
        engine = Engine(_engine_cfg())
        # dead_zone default 0.02; 0.0 should map to idle
        assert engine._throttle_to_map(0.0) == pytest.approx(engine._map_idle)

    def test_full_throttle_approaches_max_map(self) -> None:
        engine = Engine(_engine_cfg())
        m = engine._throttle_to_map(1.0)
        # At full throttle (above dead zone, gain=1.0, max=29.92)
        # the MAP should be at the configured maximum.
        assert m == pytest.approx(29.92, abs=1e-6)

    def test_monotonic(self) -> None:
        engine = Engine(_engine_cfg())
        prev = -1.0
        for thr in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            m = engine._throttle_to_map(thr)
            assert m >= prev
            prev = m


class TestRPMDynamics:
    def test_rpm_rises_toward_target(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        env = _env_state(t=0.0, throttle=0.7)
        s0 = engine.step(EngineInputs(env=env, degradation_severity=0.0))
        env2 = _env_state(t=0.1, throttle=0.7)
        s1 = engine.step(EngineInputs(env=env2, degradation_severity=0.0))
        assert s0.rpm < s1.rpm

    def test_rpm_converges_to_idle_at_zero_throttle(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        # First, run at full throttle for a few steps to get above idle.
        for i in range(20):
            env = _env_state(t=i * 0.1, throttle=1.0)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        high = engine.last_state.rpm
        # Now drop to zero throttle for a long time.
        for i in range(300):
            env = _env_state(t=(20 + i) * 0.1, throttle=0.0)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        low = engine.last_state.rpm
        assert high > low
        # Should be at idle exactly (within a small numerical tolerance).
        assert abs(low - engine._idle_rpm) < 5.0


# ---------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------
class TestThermal:
    def test_egt_within_engine_limits(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        for i in range(200):
            env = _env_state(t=i * 0.1, throttle=0.7)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        s = engine.last_state
        assert s.egt_c < engine._cfg.limits.egt_max_c

    def test_cht_within_engine_limits(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        for i in range(200):
            env = _env_state(t=i * 0.1, throttle=0.7)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        s = engine.last_state
        assert s.cht_c < engine._cfg.limits.cht_max_c


# ---------------------------------------------------------------------
# Lubrication
# ---------------------------------------------------------------------
class TestLubrication:
    def test_oil_pressure_positive(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        for i in range(100):
            env = _env_state(t=i * 0.1, throttle=0.6)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        s = engine.last_state
        assert s.oil_pressure_psi > 0.0

    def test_oil_pressure_drops_with_high_oil_temp(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        # Force hot oil
        engine._oil_t = 130.0
        for i in range(30):
            env = _env_state(t=i * 0.1, throttle=0.6)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        hot_p = engine.last_state.oil_pressure_psi

        # Reset and run at normal oil temp
        engine.reset()
        engine._oil_t = 60.0
        for i in range(30):
            env = _env_state(t=i * 0.1, throttle=0.6)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        cool_p = engine.last_state.oil_pressure_psi
        assert hot_p < cool_p


# ---------------------------------------------------------------------
# Vibration
# ---------------------------------------------------------------------
class TestVibration:
    def test_vibration_rises_with_rpm(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        for i in range(200):
            env = _env_state(t=i * 0.1, throttle=1.0)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        high = engine.last_state.vibration_rms_g
        engine.reset()
        for i in range(200):
            env = _env_state(t=i * 0.1, throttle=0.2)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        low = engine.last_state.vibration_rms_g
        assert high > low

    def test_external_vibration_adds(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        env0 = _env_state(t=0.0, throttle=0.5)
        s0 = engine.step(EngineInputs(env=env0, vibration_external=0.0))
        env1 = _env_state(t=0.1, throttle=0.5)
        s1 = engine.step(EngineInputs(env=env1, vibration_external=2.0))
        assert s1.vibration_rms_g > s0.vibration_rms_g


# ---------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------
class TestDegradation:
    def test_wear_grows_with_time(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        for i in range(500):
            env = _env_state(t=i * 0.1, throttle=0.7)
            engine.step(EngineInputs(env=env, degradation_severity=0.0))
        assert engine.wear.wear > 0.0

    def test_severity_accelerates_wear(self) -> None:
        engine_a = Engine(_engine_cfg())
        engine_a.set_dt(0.1)
        engine_b = Engine(_engine_cfg())
        engine_b.set_dt(0.1)
        n = 500
        for i in range(n):
            env = _env_state(t=i * 0.1, throttle=0.7)
            engine_a.step(EngineInputs(env=env, degradation_severity=0.0))
            engine_b.step(EngineInputs(env=env, degradation_severity=0.5))
        assert engine_b.wear.wear > engine_a.wear.wear

    def test_wear_capped_at_max(self) -> None:
        engine = Engine(_engine_cfg())
        engine.set_dt(0.1)
        # 5 000 steps is enough for the wear accumulator to saturate at
        # the configured cap under hot, high-severity operation.
        for i in range(5_000):
            env = _env_state(t=i * 0.1, throttle=1.0)
            engine.step(EngineInputs(env=env, degradation_severity=1.0))
        assert engine.wear.wear <= engine._deg_max + 1e-12


# ---------------------------------------------------------------------
# EngineSimulator + scenarios
# ---------------------------------------------------------------------
class TestEngineSimulator:
    def test_run_is_deterministic(self) -> None:
        cfg_e = _engine_cfg()
        cfg_env = _env_cfg()
        a = EngineSimulator(cfg_e, cfg_env, dt_s=0.1)
        b = EngineSimulator(cfg_e, cfg_env, dt_s=0.1)
        seq_a = [s.engine.to_dict() for s in a.run()]
        seq_b = [s.engine.to_dict() for s in b.run()]
        assert seq_a == seq_b

    def test_state_count_matches_mission(self) -> None:
        sim = EngineSimulator(_engine_cfg(), _env_cfg(), dt_s=0.1)
        steps = sim.run()
        expected = math.ceil(_env_cfg().mission.profile[-1].t_s / 0.1)
        assert len(steps) == expected

    def test_rpm_never_exceeds_redline(self) -> None:
        sim = EngineSimulator(_engine_cfg(), _env_cfg(), dt_s=0.1)
        for s in sim.run():
            assert s.engine.rpm <= sim.engine._redline_rpm + 1e-6

    def test_fuel_flow_positive_when_throttle_above_deadzone(self) -> None:
        sim = EngineSimulator(_engine_cfg(), _env_cfg(), dt_s=0.1)
        for s in sim.run():
            if s.env.throttle > 0.05:
                assert s.engine.fuel_flow_lph >= 0.0


class TestMandatoryScenarios:
    """Engine-level slices of the mandatory scenarios."""

    def test_s1_healthy_full_mission_within_limits(self) -> None:
        sim = EngineSimulator(_engine_cfg(), _env_cfg(), dt_s=0.1)
        steps = sim.run()
        # All engine signals must stay within configured limits.
        for s in steps:
            e = s.engine
            assert 0 < e.rpm <= sim.engine._redline_rpm + 1e-6
            assert 0 < e.egt_c < sim.engine._cfg.limits.egt_max_c
            assert 0 < e.cht_c < sim.engine._cfg.limits.cht_max_c
            assert sim.engine._cfg.limits.oil_pressure_min_psi <= e.oil_pressure_psi <= sim.engine._cfg.limits.oil_pressure_max_psi
            assert 0 < e.vibration_rms_g < sim.engine._cfg.limits.vibration_rms_max_g
            assert e.fuel_flow_lph >= 0.0

    def test_s7_throttle_transient_does_not_diverge(self) -> None:
        """Rapid throttle transitions are transient: RPM should follow
        the commanded target with first-order lag, not run away."""
        sim = EngineSimulator(_engine_cfg(), _env_cfg(), dt_s=0.1)
        env_states = sim.runner.run()
        # Force alternating throttle steps by patching the env_states
        # (MissionProfile is read-only; we synthesise new states).
        mutated: list[EnvironmentState] = []
        for i, env in enumerate(env_states):
            thr = 0.9 if (i // 5) % 2 == 0 else 0.1   # 0.5 s on/off
            mutated.append(EnvironmentState(
                time_s=env.time_s,
                altitude_m=env.altitude_m,
                airspeed_mps=env.airspeed_mps,
                throttle=thr,
                atmosphere=env.atmosphere,
                wind=env.wind,
                vertical_accel_mps2=env.vertical_accel_mps2,
            ))
        # RPM must remain between idle and redline despite aggressive toggling.
        sim.reset()
        max_rpm = 0.0
        min_rpm = 1e9
        for env in mutated:
            s = sim.engine.step(EngineInputs(env=env, degradation_severity=0.0))
            max_rpm = max(max_rpm, s.rpm)
            min_rpm = min(min_rpm, s.rpm)
        assert min_rpm >= sim.engine._idle_rpm - 1e-6
        assert max_rpm <= sim.engine._redline_rpm + 1e-6
