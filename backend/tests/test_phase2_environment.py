"""PHASE 2 tests — environment + mission simulator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from backend.config import (
    GustsCfg,
    Mission,
    TurbulenceCfg,
    Waypoint,
    load_config,
)
from backend.environment import (
    Atmosphere,
    EnvironmentState,
    GustModel,
    MissionProfile,
    MissionRunner,
    TurbulenceModel,
)

pytestmark = pytest.mark.phase2

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _atmos_cfg():
    cfg = load_config(CONFIG_DIR)
    return cfg.environment


# ---------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------
class TestAtmosphere:
    def test_sea_level_matches_config(self) -> None:
        env = _atmos_cfg()
        atmos = Atmosphere(env.atmosphere)
        s = atmos.state(0.0)
        assert s.altitude_m == 0.0
        assert s.pressure_pa == pytest.approx(env.atmosphere.sea_level.pressure_pa, rel=1e-3)
        assert s.temperature_c == pytest.approx(
            env.atmosphere.sea_level.temperature_k - 273.15, rel=1e-3
        )
        assert s.density_kg_per_m3 == pytest.approx(
            env.atmosphere.sea_level.density_kg_per_m3, rel=1e-3
        )

    def test_pressure_decreases_with_altitude(self) -> None:
        atmos = Atmosphere(_atmos_cfg().atmosphere)
        p0 = atmos.pressure_pa(0.0)
        p1k = atmos.pressure_pa(1000.0)
        p3k = atmos.pressure_pa(3000.0)
        p5k = atmos.pressure_pa(5000.0)
        assert p0 > p1k > p3k > p5k

    def test_temperature_decreases_in_troposphere(self) -> None:
        atmos = Atmosphere(_atmos_cfg().atmosphere)
        t0 = atmos.temperature_c(0.0)
        t5k = atmos.temperature_c(5000.0)
        # ~6.5 K/km => ~32.5 C drop at 5 km.
        assert t0 - t5k == pytest.approx(32.5, abs=0.5)

    def test_isa_known_points(self) -> None:
        """Cross-check against well-known ISA altitudes."""
        atmos = Atmosphere(_atmos_cfg().atmosphere)
        # 11 km tropopause: T = -56.5 C (ISA); P ~ 22632 Pa.
        s = atmos.state(11000.0)
        assert s.temperature_c == pytest.approx(-56.5, abs=0.1)
        assert s.pressure_pa == pytest.approx(22632.0, rel=1e-2)
        # Speed of sound at sea level: ~340.3 m/s.
        a0 = atmos.state(0.0).speed_of_sound_mps
        assert a0 == pytest.approx(340.3, abs=0.5)

    def test_negative_altitude_clamps_to_sea_level(self) -> None:
        atmos = Atmosphere(_atmos_cfg().atmosphere)
        s = atmos.state(-100.0)
        assert s.altitude_m == 0.0
        assert s.pressure_pa == pytest.approx(
            _atmos_cfg().atmosphere.sea_level.pressure_pa, rel=1e-3
        )

    def test_stratosphere_isothermal(self) -> None:
        atmos = Atmosphere(_atmos_cfg().atmosphere)
        t1 = atmos.temperature_c(12000.0)
        t2 = atmos.temperature_c(18000.0)
        assert t1 == pytest.approx(t2, abs=0.1)


# ---------------------------------------------------------------------
# Mission profile
# ---------------------------------------------------------------------
class TestMissionProfile:
    def test_linear_interpolation_midpoint(self) -> None:
        wps = [
            Waypoint(t_s=0, altitude_m=0, airspeed_mps=0, throttle=0.0),
            Waypoint(t_s=100, altitude_m=1000, airspeed_mps=50, throttle=0.8),
        ]
        p = MissionProfile(wps)
        mid = p.at(50)
        assert mid.altitude_m == pytest.approx(500)
        assert mid.airspeed_mps == pytest.approx(25)
        assert mid.throttle == pytest.approx(0.4)

    def test_clamps_before_first_and_after_last(self) -> None:
        wps = [
            Waypoint(t_s=10, altitude_m=100, airspeed_mps=10, throttle=0.1),
            Waypoint(t_s=20, altitude_m=200, airspeed_mps=20, throttle=0.2),
        ]
        p = MissionProfile(wps)
        before = p.at(0.0)
        after = p.at(100.0)
        assert before.altitude_m == 100
        assert before.airspeed_mps == 10
        assert after.altitude_m == 200
        assert after.airspeed_mps == 20

    def test_rejects_single_waypoint(self) -> None:
        with pytest.raises(ValueError):
            MissionProfile([Waypoint(t_s=0, altitude_m=0, airspeed_mps=0, throttle=0.0)])

    def test_rejects_non_monotonic(self) -> None:
        wps = [
            Waypoint(t_s=10, altitude_m=0, airspeed_mps=0, throttle=0.0),
            Waypoint(t_s=10, altitude_m=0, airspeed_mps=0, throttle=0.0),  # dup
            Waypoint(t_s=20, altitude_m=0, airspeed_mps=0, throttle=0.0),
        ]
        with pytest.raises(ValueError):
            MissionProfile(wps)

    def test_duration_is_last_waypoint_time(self) -> None:
        wps = [
            Waypoint(t_s=0, altitude_m=0, airspeed_mps=0, throttle=0.0),
            Waypoint(t_s=42, altitude_m=0, airspeed_mps=0, throttle=0.0),
        ]
        assert MissionProfile(wps).duration_s == 42


# ---------------------------------------------------------------------
# Turbulence
# ---------------------------------------------------------------------
class TestTurbulence:
    def test_disabled_returns_zero(self) -> None:
        turb = TurbulenceModel(
            TurbulenceCfg(enabled=False, intensity=0.0, correlation_time_s=1.0, seed=1)
        )
        for _ in range(10):
            assert turb.step(0.1) == 0.0

    def test_steady_state_variance_within_tolerance(self) -> None:
        turb = TurbulenceModel(
            TurbulenceCfg(enabled=True, intensity=0.10, correlation_time_s=5.0, seed=123)
        )
        # sigma = 0.10 * 3.0 = 0.3 m/s
        n = 200_000
        dt = 0.05
        samples = np.array([turb.step(dt) for _ in range(n)])
        var = samples.var()
        sigma2 = 0.3 ** 2
        # Allow 20% tolerance — first-order Euler has O(dt) bias.
        assert var == pytest.approx(sigma2, rel=0.20)

    def test_deterministic_with_fixed_seed(self) -> None:
        a = TurbulenceModel(
            TurbulenceCfg(enabled=True, intensity=0.10, correlation_time_s=5.0, seed=42)
        )
        b = TurbulenceModel(
            TurbulenceCfg(enabled=True, intensity=0.10, correlation_time_s=5.0, seed=42)
        )
        a_seq = [a.step(0.1) for _ in range(50)]
        b_seq = [b.step(0.1) for _ in range(50)]
        assert a_seq == b_seq

    def test_different_seeds_diverge(self) -> None:
        a = TurbulenceModel(
            TurbulenceCfg(enabled=True, intensity=0.10, correlation_time_s=5.0, seed=1)
        )
        b = TurbulenceModel(
            TurbulenceCfg(enabled=True, intensity=0.10, correlation_time_s=5.0, seed=2)
        )
        a_seq = [a.step(0.1) for _ in range(50)]
        b_seq = [b.step(0.1) for _ in range(50)]
        assert a_seq != b_seq


# ---------------------------------------------------------------------
# Gusts
# ---------------------------------------------------------------------
class TestGusts:
    def test_disabled_returns_zero(self) -> None:
        g = GustModel(GustsCfg(enabled=False, rate_per_hour=0.0, amplitude_mps=5.0, seed=1))
        for t in (0.1, 1.0, 10.0):
            assert g.step(t, 0.1) == 0.0

    def test_amplitude_bounded(self) -> None:
        g = GustModel(GustsCfg(enabled=True, rate_per_hour=4.0, amplitude_mps=6.0, seed=9))
        for i in range(20_000):
            v = g.step(i * 0.1, 0.1)
            assert -6.0 - 1e-9 <= v <= 6.0 + 1e-9

    def test_arrival_rate_approximately_configured(self) -> None:
        rate = 360.0  # per hour => 0.1 per second; many arrivals
        g = GustModel(
            GustsCfg(enabled=True, rate_per_hour=rate, amplitude_mps=6.0, seed=11)
        )
        T = 3600.0
        dt = 0.05
        n = int(T / dt)
        n_steps = 0
        for i in range(n):
            v = g.step(i * dt, dt)
            if abs(v) > 1e-9:
                n_steps += 1
        # Expected fraction of time spent in a gust: rate * duration.
        # 0.1 / s * 8 s = 0.8; the active flag is a sticky envelope so
        # the total active seconds is bounded by (rate * duration) * T.
        # 0.1 * 8 * 3600 = 2880 active-seconds expected, so 2880 / 3600 ≈ 0.8
        # fraction of steps with non-zero gust.
        frac = n_steps / n
        assert 0.55 < frac < 0.99, f"unexpected active fraction {frac}"


# ---------------------------------------------------------------------
# Mission runner + composite wind
# ---------------------------------------------------------------------
class TestMissionRunner:
    def test_runner_is_deterministic(self) -> None:
        env = _atmos_cfg()
        env_run_a = MissionRunner(env, dt_s=0.1)
        env_run_b = MissionRunner(env, dt_s=0.1)
        run_a = [s.to_dict() for s in env_run_a.run()]
        run_b = [s.to_dict() for s in env_run_b.run()]
        assert run_a == run_b

    def test_run_full_default_mission(self) -> None:
        env = _atmos_cfg()
        runner = MissionRunner(env, dt_s=0.1)
        states = runner.run()
        assert len(states) > 0
        first, _ = states[0], states[-1]
        # First waypoint: t=0, alt=0, throttle=0.0
        assert first.time_s == 0.0
        assert first.altitude_m == 0.0
        assert first.throttle == 0.0
        # Profile.end matches the last waypoint exactly.
        p = runner.profile
        end = p.at(p.duration_s)
        assert end.altitude_m == 0.0
        assert end.throttle == pytest.approx(0.2)

    def test_atmospheric_state_tracks_altitude(self) -> None:
        env = _atmos_cfg()
        runner = MissionRunner(env, dt_s=0.1)
        for s in runner.run():
            assert s.atmosphere.altitude_m == pytest.approx(s.altitude_m, abs=1e-6)

    def test_pressure_versus_altitude_monotonic_in_full_run(self) -> None:
        env = _atmos_cfg()
        runner = MissionRunner(env, dt_s=0.1)
        states = runner.run()
        for a, b in zip(states, states[1:]):
            # When altitude strictly increases, pressure must strictly decrease.
            if b.altitude_m - a.altitude_m > 1e-3:
                assert b.atmosphere.pressure_pa < a.atmosphere.pressure_pa

    def test_run_up_to_advances_partially(self) -> None:
        env = _atmos_cfg()
        runner = MissionRunner(env, dt_s=0.1)
        runner.run_up_to(50.0)
        assert runner.dt_s == 0.1
        # The runner's internal time should be near 50s now.
        # Run one more step and check the time delta.
        s = runner.step()
        assert 50.0 <= s.time_s <= 50.5

    def test_reset_returns_to_start(self) -> None:
        env = _atmos_cfg()
        runner = MissionRunner(env, dt_s=0.1)
        runner.run()
        runner.reset()
        s = runner.step()
        assert s.time_s == pytest.approx(0.0, abs=1e-6)
        assert s.altitude_m == 0.0


# ---------------------------------------------------------------------
# Mandatory scenarios (env-level slice)
# ---------------------------------------------------------------------
class TestMandatoryScenarios:
    """S1 (Healthy) and S2 (Turbulence) — environment-level evidence only.
    The engine-level classifications arrive in later phases."""

    def test_scenario_1_healthy_no_disturbance(self) -> None:
        env = _atmos_cfg()
        env.disturbances.turbulence.enabled = False
        env.disturbances.gusts.enabled = False
        runner = MissionRunner(env, dt_s=0.1)
        # Wind should be identically zero throughout the mission.
        for s in runner.run():
            assert s.wind.total_w_mps == 0.0
            assert s.wind.is_gust_active is False
            assert s.vertical_accel_mps2 == 0.0

    def test_scenario_2_turbulence_not_engine_fault(self) -> None:
        """Env-only check: strong turbulence must produce large wind
        and vertical-accel signatures, but the engine status here is
        not yet defined — that's PHASE 3+. The test asserts only that
        the environment model exposes the disturbance cleanly.
        """
        env = _atmos_cfg()
        env.disturbances.turbulence.intensity = 0.40
        env.disturbances.turbulence.seed = 7
        env.disturbances.gusts.enabled = False
        runner = MissionRunner(env, dt_s=0.1)
        states = runner.run()
        # Some wind values must be non-zero (turbulence is active).
        max_w = max(abs(s.wind.turbulence_w_mps) for s in states)
        assert max_w > 0.05
        # Vertical accel is non-zero at some point.
        max_va = max(abs(s.vertical_accel_mps2) for s in states)
        assert max_va > 0.0
