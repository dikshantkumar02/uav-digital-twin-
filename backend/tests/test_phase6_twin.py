"""PHASE 6 tests — Digital Twin (physics-driven state estimator)."""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, List

import pytest

from backend.config import EngineConfig, EnvironmentConfig, load_config
from backend.digital_twin import (
    CHANNEL_TO_STATE,
    ChannelResidual,
    DigitalTwin,
    ResidualFrame,
    TwinState,
)
from backend.environment import Atmosphere, EnvironmentState
from backend.environment.state import WindState
from backend.simulation import EngineSimulator

pytestmark = pytest.mark.phase6

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------
def _engine_cfg() -> EngineConfig:
    return load_config(CONFIG_DIR).engine


def _env_cfg() -> EnvironmentConfig:
    return load_config(CONFIG_DIR).environment


def _env_state(
    t: float = 0.0,
    alt: float = 0.0,
    airspeed: float = 0.0,
    throttle: float = 0.0,
) -> EnvironmentState:
    """Build a minimal, fully self-consistent EnvironmentState."""
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


def _make_twin(**kw) -> DigitalTwin:
    return DigitalTwin(_engine_cfg(), **kw)


def _drive_twin(env_states: List[EnvironmentState], gain: float = 0.15):
    twin = _make_twin(dt_s=0.1, closed_loop_gain=gain)
    out: List[TwinState] = []
    for env in env_states:
        out.append(twin.step(env))
    return twin, out


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------
class TestConstruction:
    def test_init_defaults(self) -> None:
        t = _make_twin()
        assert t.dt_s == pytest.approx(0.1)
        assert t.last_state is None

    def test_invalid_dt(self) -> None:
        with pytest.raises(ValueError):
            DigitalTwin(_engine_cfg(), dt_s=0.0)
        with pytest.raises(ValueError):
            DigitalTwin(_engine_cfg(), dt_s=-1.0)

    def test_invalid_gain(self) -> None:
        with pytest.raises(ValueError):
            DigitalTwin(_engine_cfg(), closed_loop_gain=-0.1)
        with pytest.raises(ValueError):
            DigitalTwin(_engine_cfg(), closed_loop_gain=1.1)

    def test_set_reference_density(self) -> None:
        t = _make_twin()
        t.set_reference_density(1.0)
        t.set_reference_density(0.0)  # ignored
        t.set_reference_density(-1.0)  # ignored

    def test_reset(self) -> None:
        t = _make_twin()
        t.step(_env_state(throttle=0.5, alt=1000.0))
        assert t.last_state is not None
        t.reset()
        assert t.last_state is None


# ---------------------------------------------------------------------
# TwinState
# ---------------------------------------------------------------------
class TestTwinState:
    def test_fields_match_truth(self) -> None:
        """TwinState must mirror EngineState's non-wear fields."""
        s = TwinState(
            time_s=1.0,
            rpm=2000.0,
            manifold_pressure_inhg=20.0,
            egt_c=700.0,
            cht_c=200.0,
            oil_pressure_psi=60.0,
            oil_temperature_c=100.0,
            fuel_flow_lph=40.0,
            vibration_rms_g=1.5,
            bsfc_g_per_kwh=350.0,
            brake_power_kw=100.0,
            altitude_m=1000.0,
            airspeed_mps=60.0,
            ambient_temperature_c=10.0,
            ambient_pressure_pa=90000.0,
        )
        d = s.to_dict()
        for k in (
            "time_s", "rpm", "egt_c", "cht_c", "oil_pressure_psi",
            "oil_temperature_c", "fuel_flow_lph", "vibration_rms_g",
            "bsfc_g_per_kwh", "brake_power_kw", "altitude_m", "airspeed_mps",
            "ambient_temperature_c", "ambient_pressure_pa", "confidence",
        ):
            assert k in d
        assert d["confidence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Channel / residual model
# ---------------------------------------------------------------------
class TestResidual:
    def test_channel_to_state_covers_engine_channels(self) -> None:
        """The map must cover every channel the sensor bundle exposes."""
        for ch in (
            "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
            "fuel_flow", "vibration", "altitude", "airspeed",
            "ambient_temperature", "ambient_pressure",
        ):
            assert ch in CHANNEL_TO_STATE

    def test_channel_residual_is_outlier(self) -> None:
        cr = ChannelResidual(
            channel="rpm",
            observed=3000.0, predicted=2000.0, residual=1000.0,
            z_score=10.0, confidence=0.0, in_bounds=False,
        )
        assert cr.is_outlier is True
        cr2 = ChannelResidual(
            channel="rpm",
            observed=2005.0, predicted=2000.0, residual=5.0,
            z_score=0.1, confidence=0.98, in_bounds=True,
        )
        assert cr2.is_outlier is False

    def test_residual_frame_dict_round_trip(self) -> None:
        cr = ChannelResidual(
            channel="rpm", observed=2100.0, predicted=2000.0,
            residual=100.0, z_score=3.3, confidence=0.45, in_bounds=False,
        )
        rf = ResidualFrame(time_s=1.0)
        rf.residuals["rpm"] = cr
        rf.overall_confidence = 0.45
        d = rf.to_dict()
        assert d["time_s"] == pytest.approx(1.0)
        assert d["overall_confidence"] == pytest.approx(0.45)
        assert d["residual.rpm.observed"] == pytest.approx(2100.0)
        assert d["residual.rpm.z_score"] == pytest.approx(3.3)
        assert rf.outlier_channels() == ["rpm"]

    def test_residual_frame_getitem(self) -> None:
        rf = ResidualFrame(time_s=0.0)
        rf.residuals["egt"] = ChannelResidual(
            channel="egt", observed=700.0, predicted=650.0,
            residual=50.0, z_score=2.0, confidence=0.66, in_bounds=True,
        )
        assert rf["egt"].channel == "egt"


# ---------------------------------------------------------------------
# Open-loop prediction
# ---------------------------------------------------------------------
class TestPredict:
    def test_idle_start(self) -> None:
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.0, alt=0.0))
        geom = _engine_cfg().geometry
        assert s.rpm == pytest.approx(geom.idle_rpm, abs=1.0)
        tm = _engine_cfg().throttle_to_map
        assert s.manifold_pressure_inhg == pytest.approx(tm.idle_map_inhg, abs=0.5)

    def test_throttle_increases_rpm(self) -> None:
        t = _make_twin()
        # Throttle up; let dynamics settle.
        for _ in range(200):
            t.predict(_env_state(throttle=0.7))
        s = t.last_state
        geom = _engine_cfg().geometry
        assert s.rpm > 0.5 * geom.rated_rpm

    def test_rpm_clamped_to_redline(self) -> None:
        t = _make_twin()
        for _ in range(500):
            t.predict(_env_state(throttle=1.0))
        geom = _engine_cfg().geometry
        assert t.last_state.rpm <= geom.redline_rpm + 1e-6

    def test_egt_rises_with_load(self) -> None:
        t = _make_twin()
        for _ in range(300):
            t.predict(_env_state(throttle=0.8, alt=1000.0))
        # EGT should be well above the idle EGT.
        assert t.last_state.egt_c > 500.0

    def test_confidence_full_when_open_loop(self) -> None:
        t = _make_twin()
        s = t.step(_env_state(throttle=0.5))
        assert s.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Closed-loop step
# ---------------------------------------------------------------------
class TestClosedLoop:
    def test_observation_nudges_state(self) -> None:
        t = _make_twin(closed_loop_gain=0.5)
        # Let the twin reach a steady-state prediction.
        for _ in range(50):
            t.predict(_env_state(throttle=0.5))
        s0 = t.predict(_env_state(throttle=0.5))  # baseline prediction
        target = 3000.0
        # step() runs predict() again (which advances _rpm), then applies
        # the closed-loop correction. The expected rpm is therefore
        # (1-g) * advanced_prediction + g * observation.
        s1 = t.step(_env_state(throttle=0.5), observations={"rpm": target})
        # s1.rpm should lie between s0.rpm and target (nudged by gain).
        assert s0.rpm < s1.rpm < target

    def test_confidence_decreases_with_observations(self) -> None:
        t = _make_twin()
        # 5 observations across 5 channels.
        s = t.step(
            _env_state(throttle=0.5),
            observations={
                "rpm": 2000.0, "egt": 650.0, "cht": 200.0,
                "oil_pressure": 60.0, "oil_temperature": 90.0,
            },
        )
        # 0.05 * 5 = 0.25 penalty
        assert s.confidence == pytest.approx(0.75)

    def test_unknown_channel_ignored(self) -> None:
        t = _make_twin()
        s = t.step(_env_state(throttle=0.5), observations={"made_up": 42.0})
        assert s.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------
class TestResidualComputation:
    def test_healthy_residuals_small(self) -> None:
        """Twin tracking itself yields zero residual."""
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.5, alt=500.0))
        # Feed the twin's own predictions back in as observations.
        obs = {
            "rpm": s.rpm,
            "egt": s.egt_c,
            "cht": s.cht_c,
            "oil_pressure": s.oil_pressure_psi,
            "oil_temperature": s.oil_temperature_c,
        }
        rf = t.residual(s, obs)
        for ch, r in rf.residuals.items():
            assert r.z_score == pytest.approx(0.0, abs=1e-6)

    def test_unmodelled_offset_raises_residual(self) -> None:
        """An un-modelled +50 C bias in EGT should yield a positive residual."""
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.5, alt=500.0))
        obs = {"egt": s.egt_c + 50.0}
        rf = t.residual(s, obs)
        assert rf["egt"].residual == pytest.approx(50.0)
        # Default sigma for egt is 25 → z = 2.0.
        assert rf["egt"].z_score == pytest.approx(2.0)
        assert rf["egt"].in_bounds is True

    def test_huge_offset_is_outlier(self) -> None:
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.5))
        obs = {"rpm": s.rpm + 5.0 * 30.0}  # 5σ
        rf = t.residual(s, obs)
        assert rf["rpm"].z_score == pytest.approx(5.0)
        assert rf["rpm"].in_bounds is False
        assert "rpm" in rf.outlier_channels()

    def test_missing_observation_skipped(self) -> None:
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.5))
        rf = t.residual(s, {"rpm": s.rpm})  # only rpm
        assert "rpm" in rf.residuals
        assert "egt" not in rf.residuals
        assert rf.overall_confidence == pytest.approx(1.0)

    def test_overall_confidence_average(self) -> None:
        t = _make_twin()
        s = t.predict(_env_state(throttle=0.5))
        # Two channels, both with z=1 → conf = max(0, 1 - 1/6) ≈ 0.833
        obs = {"rpm": s.rpm + 30.0, "cht": s.cht_c + 15.0}
        rf = t.residual(s, obs)
        assert rf.overall_confidence == pytest.approx(1.0 - 1.0 / 6.0)


# ---------------------------------------------------------------------
# Determinism + tracking
# ---------------------------------------------------------------------
class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        envs = [_env_state(t=i * 0.1, throttle=0.5, alt=500.0) for i in range(50)]
        t1, out1 = _drive_twin(envs)
        t2, out2 = _drive_twin(envs)
        for a, b in zip(out1, out2):
            assert a.rpm == pytest.approx(b.rpm)
            assert a.egt_c == pytest.approx(b.egt_c)
            assert a.cht_c == pytest.approx(b.cht_c)

    def test_zero_severity_tracks_truth(self) -> None:
        """Healthy mission: twin should track the truth model closely."""
        cfg_e, cfg_env = _engine_cfg(), _env_cfg()
        sim = EngineSimulator(cfg_e, cfg_env, dt_s=0.1)
        sim.reset()
        twin = DigitalTwin(cfg_e, dt_s=0.1, closed_loop_gain=0.2)
        twin_states: List[TwinState] = []
        truth_states: List = []
        for _ in range(200):
            step = sim.step()  # severity=0.0
            ts = twin.step(step.env, observations={
                "rpm": step.engine.rpm,
                "egt": step.engine.egt_c,
                "cht": step.engine.cht_c,
            })
            twin_states.append(ts)
            truth_states.append(step.engine)
        mae = twin.twin_vs_truth_mae(twin_states, truth_states)
        # After warm-up, the closed-loop twin should be within a few
        # units on RPM and a couple of degrees on EGT/CHT.
        assert mae["rpm"] < 10.0
        assert mae["egt_c"] < 10.0
        assert mae["cht_c"] < 5.0


# ---------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------
class TestLatency:
    def test_single_step_under_target(self) -> None:
        """A single step must complete in well under the 100 ms tick budget."""
        t = _make_twin()
        envs = [_env_state(t=i * 0.1, throttle=0.5, alt=500.0) for i in range(50)]
        # Warm up the caches.
        for e in envs[:5]:
            t.step(e)
        # Measure.
        t0 = _time.perf_counter()
        n = 200
        for e in envs:
            t.step(e, observations={"rpm": 2000.0, "egt": 650.0})
        elapsed = _time.perf_counter() - t0
        per_step_ms = 1000.0 * elapsed / n
        assert per_step_ms < 5.0  # generous; this is pure-Python


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------
class TestScenarios:
    def test_s1_healthy_residuals_small(self) -> None:
        """S1 (healthy): residuals should stay small across a full mission."""
        cfg_e, cfg_env = _engine_cfg(), _env_cfg()
        sim = EngineSimulator(cfg_e, cfg_env, dt_s=0.1)
        sim.reset()
        twin = DigitalTwin(cfg_e, dt_s=0.1, closed_loop_gain=0.2)
        max_z = 0.0
        for _ in range(200):
            step = sim.step()  # severity=0.0
            obs = {
                "rpm": step.engine.rpm,
                "egt": step.engine.egt_c,
                "cht": step.engine.cht_c,
                "oil_pressure": step.engine.oil_pressure_psi,
                "oil_temperature": step.engine.oil_temperature_c,
                "fuel_flow": step.engine.fuel_flow_lph,
            }
            ts = twin.step(step.env, observations=obs)
            rf = twin.residual(ts, obs)
            for r in rf.residuals.values():
                if r.z_score is not None:
                    max_z = max(max_z, abs(r.z_score))
        # Under a healthy mission with closed-loop correction,
        # |z| should be very small.
        assert max_z < 0.5

    def test_s4_degradation_residuals_grow(self) -> None:
        """S4 (degradation): under severity=1.0 the truth model loses
        power / RPM faster than the twin's un-modelled prediction. We
        run the twin open-loop (no observations) on the same env and
        confirm the per-tick |residual| grows above the healthy floor."""
        cfg_e, cfg_env = _engine_cfg(), _env_cfg()
        sim = EngineSimulator(cfg_e, cfg_env, dt_s=0.1)
        twin = DigitalTwin(cfg_e, dt_s=0.1, closed_loop_gain=0.2)
        # First, establish a healthy baseline residual.
        sim.reset()
        for _ in range(200):
            twin.predict(sim.step().env)
        sim.reset()
        healthy_abs: List[float] = []
        for _ in range(200):
            step = sim.step()
            ts = twin.predict(step.env)
            healthy_abs.append(abs(step.engine.rpm - ts.rpm))
        healthy_mean = sum(healthy_abs) / len(healthy_abs)
        # Now run with severity=1.0; the truth RPM drops below the twin's
        # nominal target, so the residual grows.
        sim.reset()
        degraded_abs: List[float] = []
        for _ in range(200):
            step = sim.step(degradation_severity=1.0)
            ts = twin.predict(step.env)
            degraded_abs.append(abs(step.engine.rpm - ts.rpm))
        degraded_mean = sum(degraded_abs) / len(degraded_abs)
        assert degraded_mean > healthy_mean
        assert degraded_mean > 20.0  # a clear, un-modelled gap

    def test_s7_rapid_throttle_keeps_twin_bounded(self) -> None:
        """S7 (rapid throttle): aggressive toggling must not diverge."""
        twin = _make_twin()
        for i in range(500):
            thr = 0.0 if (i // 2) % 2 == 0 else 1.0
            twin.step(_env_state(throttle=thr, t=i * 0.1))
        geom = _engine_cfg().geometry
        s = twin.last_state
        assert geom.idle_rpm - 1.0 <= s.rpm <= geom.redline_rpm + 1.0
        # EGT stays within plausible range.
        assert 300.0 <= s.egt_c <= 900.0
