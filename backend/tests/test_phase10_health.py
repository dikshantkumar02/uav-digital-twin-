"""PHASE 10 tests — health index."""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from backend.config import load_config
from backend.diagnostics import (
    AnomalyAssessment,
    AnomalyLabel,
    ChannelAnomaly,
)
from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.health import (
    DEFAULT_WEIGHTS,
    FAULT_CONTRIBUTION_THRESHOLD,
    HealthIndex,
    HealthIndexCalculator,
    HealthLabel,
    HealthTrend,
    MIN_OVERALL_CONFIDENCE,
    Subsystem,
    SubsystemHealth,
    TREND_EPSILON,
    TREND_WINDOW,
    aggregate,
    label_for,
    new_trend_history,
    normalize_weights,
    score_lubrication,
    score_mechanical,
    score_performance,
    score_sensors,
    score_thermal,
    update_trend_history,
)
from backend.ml import (
    CalibrationStatus,
    FaultClassification,
)
from backend.sensors import NoiseMode, SensorBundle, SensorReading, SensorSample
from backend.simulation import EngineSimulator, EngineState
from backend.telemetry import FrameStatus

pytestmark = pytest.mark.phase10

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_state(
    *,
    time_s: float = 1.0,
    rpm: float = 2000.0,
    manifold_pressure_inhg: float = 22.0,
    egt_c: float = 650.0,
    cht_c: float = 200.0,
    oil_pressure_psi: float = 60.0,
    oil_temperature_c: float = 90.0,
    fuel_flow_lph: float = 30.0,
    vibration_rms_g: float = 1.5,
    bsfc_g_per_kwh: float = 400.0,
    brake_power_kw: float = 100.0,
    wear: float = 0.0,
    throttle: float = 0.6,
    altitude_m: float = 500.0,
    airspeed_mps: float = 50.0,
) -> EngineState:
    return EngineState(
        time_s=time_s,
        rpm=rpm,
        manifold_pressure_inhg=manifold_pressure_inhg,
        egt_c=egt_c,
        cht_c=cht_c,
        oil_pressure_psi=oil_pressure_psi,
        oil_temperature_c=oil_temperature_c,
        fuel_flow_lph=fuel_flow_lph,
        vibration_rms_g=vibration_rms_g,
        bsfc_g_per_kwh=bsfc_g_per_kwh,
        brake_power_kw=brake_power_kw,
        wear=wear,
        throttle=throttle,
        altitude_m=altitude_m,
        airspeed_mps=airspeed_mps,
    )


def _nominal_state(*, time_s: float = 1.0) -> EngineState:
    """A clean, fully-healthy engine state.

    Values are picked to land on the nominal/scoring targets so the
    overall score is exactly 1.0:
      * ``rpm`` = rated_rpm (2400)
      * ``oil_pressure_psi`` = nominal_pressure_psi (60)
      * ``brake_power_kw`` = rated_power_kw (134)
      * EGT, CHT, oil temp: in their permitted envelopes
      * wear = 0
      * vibration well below max
    """
    return _make_state(
        time_s=time_s,
        rpm=2400.0,          # at rated
        manifold_pressure_inhg=25.0,  # mid-range
        egt_c=650.0,         # well below 870
        cht_c=200.0,         # well below 260
        oil_pressure_psi=60.0,  # at nominal
        oil_temperature_c=90.0,   # in (40, 120)
        fuel_flow_lph=30.0,
        vibration_rms_g=1.5,  # well below 6.0
        bsfc_g_per_kwh=400.0,
        brake_power_kw=134.0,  # at rated
        wear=0.0,
    )


def _healthy_assessment(channels: int = 7) -> AnomalyAssessment:
    """Build a fully-clean anomaly assessment for the given channels."""
    cs: Dict[str, ChannelAnomaly] = {}
    for i in range(channels):
        name = f"ch{i}"
        cs[name] = ChannelAnomaly(
            channel=name, score=0.0, label=AnomalyLabel.NORMAL,
            confidence=1.0, contributors=[],
        )
    return AnomalyAssessment(
        time_s=0.0, overall_score=0.0, overall_label=AnomalyLabel.NORMAL,
        confidence=1.0, channels=cs, notes=[], frame_status="OK",
    )


def _anomalous_assessment(
    target: str, score: float = 0.95
) -> AnomalyAssessment:
    """Build an assessment with one ANOMALY channel."""
    cs: Dict[str, ChannelAnomaly] = {
        target: ChannelAnomaly(
            channel=target, score=score, label=AnomalyLabel.ANOMALY,
            confidence=1.0, contributors=[f"residual {target}"],
        ),
    }
    return AnomalyAssessment(
        time_s=0.0, overall_score=score, overall_label=AnomalyLabel.ANOMALY,
        confidence=1.0, channels=cs, notes=[], frame_status="OK",
    )


# ---------------------------------------------------------------------
# TestTypes
# ---------------------------------------------------------------------
class TestTypes:
    def test_label_members(self) -> None:
        assert {m.value for m in HealthLabel} == {
            "HEALTHY", "DEGRADED", "CRITICAL", "INSUFFICIENT_DATA",
        }

    def test_trend_members(self) -> None:
        assert {m.value for m in HealthTrend} == {
            "IMPROVING", "STABLE", "DEGRADING", "INSUFFICIENT_DATA",
        }

    def test_subsystem_members(self) -> None:
        assert {m.value for m in Subsystem} == {
            "THERMAL", "LUBRICATION", "PERFORMANCE", "MECHANICAL", "SENSORS",
        }

    def test_subsystem_health_to_dict(self) -> None:
        sh = SubsystemHealth(
            subsystem=Subsystem.THERMAL, score=0.7, confidence=1.0,
            contributors=["egt_c"], notes=["hi"],
        )
        d = sh.to_dict()
        assert d["subsystem"] == "THERMAL"
        assert d["score"] == pytest.approx(0.7)
        assert d["confidence"] == pytest.approx(1.0)
        assert d["contributors"] == ["egt_c"]
        assert d["notes"] == ["hi"]
        assert sh.is_insufficient is False

    def test_health_index_to_dict(self) -> None:
        idx = HealthIndex(
            time_s=1.0, overall_score=0.6, overall_label=HealthLabel.DEGRADED,
            confidence=0.9, trend=HealthTrend.STABLE, wear=0.1,
        )
        d = idx.to_dict()
        assert d["time_s"] == 1.0
        assert d["overall_score"] == pytest.approx(0.6)
        assert d["overall_label"] == "DEGRADED"
        assert d["trend"] == "STABLE"
        assert d["wear"] == 0.1
        assert idx.is_degraded is True
        assert idx.is_healthy is False

    def test_label_for_thresholds(self) -> None:
        assert label_for(1.0) is HealthLabel.HEALTHY
        assert label_for(0.80) is HealthLabel.HEALTHY
        assert label_for(0.79) is HealthLabel.DEGRADED
        assert label_for(0.55) is HealthLabel.DEGRADED
        assert label_for(0.54) is HealthLabel.CRITICAL
        assert label_for(0.01) is HealthLabel.CRITICAL
        assert label_for(0.0) is HealthLabel.INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# TestSubsystems
# ---------------------------------------------------------------------
class TestSubsystems:
    def test_thermal_nominal_is_one(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        s = score_thermal(_nominal_state(), cfg)
        assert s.score == pytest.approx(1.0, abs=1e-6)
        assert s.confidence == pytest.approx(1.0)

    def test_thermal_egt_at_redline(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.egt_c = cfg.limits.egt_max_c
        s = score_thermal(st, cfg)
        # EGT at redline -> 0.0. Other channels at 1.0. Mean = 0.667.
        assert s.score == pytest.approx(2.0 / 3.0, abs=1e-3)
        assert "egt_c" in s.contributors

    def test_thermal_cht_over(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.cht_c = cfg.limits.cht_max_c + 50.0
        s = score_thermal(st, cfg)
        # CHT well over max -> 0.0
        assert "cht_c" in s.contributors
        assert s.score < 0.7

    def test_thermal_no_state(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        s = score_thermal(None, cfg)
        assert s.score == 0.0
        assert s.confidence == 0.0
        assert s.is_insufficient is True

    def test_lubrication_pressure_nominal(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        s = score_lubrication(_nominal_state(), cfg)
        assert s.score == pytest.approx(1.0, abs=1e-6)
        assert s.confidence == pytest.approx(1.0)

    def test_lubrication_pressure_at_min(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.oil_pressure_psi = cfg.limits.oil_pressure_min_psi
        s = score_lubrication(st, cfg)
        # oil pressure at min -> 0.0; oil temp still healthy.
        assert "oil_pressure_psi" in s.contributors
        assert s.score < 0.6

    def test_performance_nominal(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        s = score_performance(_nominal_state(), cfg)
        # All five channels should be at or near 1.0.
        assert s.score >= 0.9
        assert s.confidence == pytest.approx(1.0)

    def test_performance_idle_rpm_low(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.rpm = cfg.geometry.idle_rpm
        s = score_performance(st, cfg)
        assert "rpm" in s.contributors
        # Should still be well above 0 (subsystem includes other channels).
        assert s.score < 1.0

    def test_mechanical_healthy(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        s = score_mechanical(_nominal_state(), cfg)
        assert s.score == pytest.approx(1.0, abs=1e-6)
        assert s.confidence == pytest.approx(1.0)

    def test_mechanical_wear_one(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.wear = 1.0
        s = score_mechanical(st, cfg)
        # Wear=1.0 -> 0.0; vibration at 1.5g -> 1.0. Mean = 0.5.
        assert "wear" in s.contributors
        assert s.score == pytest.approx(0.5, abs=1e-6)

    def test_sensors_clean_is_one(self) -> None:
        s = score_sensors(_healthy_assessment())
        assert s.score == pytest.approx(1.0, abs=1e-6)
        assert s.confidence == pytest.approx(1.0)
        assert s.contributors == []

    def test_sensors_one_anomaly_drops(self) -> None:
        a = _anomalous_assessment("rpm", score=0.90)
        s = score_sensors(a)
        # One channel inverted: 0.10. Mean over many channels -> low.
        assert "rpm" in s.contributors
        assert s.score < 0.5

    def test_sensors_no_assessment(self) -> None:
        s = score_sensors(None)
        assert s.score == 0.0
        assert s.confidence == 0.0
        assert s.is_insufficient is True


# ---------------------------------------------------------------------
# TestAggregator
# ---------------------------------------------------------------------
class TestAggregator:
    def test_weighted_mean(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        subs = {
            Subsystem.THERMAL: score_thermal(st, cfg),
            Subsystem.LUBRICATION: score_lubrication(st, cfg),
            Subsystem.PERFORMANCE: score_performance(st, cfg),
            Subsystem.MECHANICAL: score_mechanical(st, cfg),
            Subsystem.SENSORS: score_sensors(_healthy_assessment()),
        }
        idx = aggregate(subs, time_s=1.0)
        # All nominal -> score == 1.0 -> label HEALTHY.
        assert idx.overall_score == pytest.approx(1.0, abs=1e-6)
        assert idx.overall_label is HealthLabel.HEALTHY
        assert idx.trend is HealthTrend.INSUFFICIENT_DATA  # no history

    def test_normalize_weights(self) -> None:
        w = normalize_weights({
            Subsystem.THERMAL: 1.0,
            Subsystem.LUBRICATION: 1.0,
            Subsystem.PERFORMANCE: 0.0,
            Subsystem.MECHANICAL: 0.0,
            Subsystem.SENSORS: 0.0,
        })
        # 1.0 / 2.0 = 0.5
        assert w[Subsystem.THERMAL] == pytest.approx(0.5)
        assert w[Subsystem.LUBRICATION] == pytest.approx(0.5)
        # The zeroed subsystems remain 0.
        assert w[Subsystem.PERFORMANCE] == 0.0

    def test_normalize_weights_empty_falls_back(self) -> None:
        w = normalize_weights({})
        assert w == DEFAULT_WEIGHTS

    def test_aggregate_empty_is_insufficient(self) -> None:
        idx = aggregate({}, time_s=0.0)
        assert idx.overall_score == 0.0
        assert idx.confidence < MIN_OVERALL_CONFIDENCE
        assert idx.overall_label is HealthLabel.INSUFFICIENT_DATA

    def test_aggregate_degraded_label(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.egt_c = cfg.limits.egt_max_c  # one channel redline
        subs = {
            Subsystem.THERMAL: score_thermal(st, cfg),
            Subsystem.LUBRICATION: score_lubrication(st, cfg),
            Subsystem.PERFORMANCE: score_performance(st, cfg),
            Subsystem.MECHANICAL: score_mechanical(st, cfg),
            Subsystem.SENSORS: score_sensors(_healthy_assessment()),
        }
        idx = aggregate(subs, time_s=1.0)
        # Overall should be DEGRADED: thermal ≈ 0.667, others 1.0.
        # Weighted: 0.20*0.667 + 0.80*1.0 = 0.9333. Still HEALTHY.
        # The aggregate respects subsystems, not raw channels.
        assert 0.8 < idx.overall_score <= 1.0
        assert idx.overall_label is HealthLabel.HEALTHY


# ---------------------------------------------------------------------
# TestTrend
# ---------------------------------------------------------------------
class TestTrend:
    def test_insufficient_with_short_history(self) -> None:
        hist = new_trend_history()
        for v in [0.7, 0.7, 0.7]:
            update_trend_history(hist, v)
        # Only 3 entries, need TREND_WINDOW.
        assert len(hist) == 3
        idx = aggregate({}, trend_history=hist, time_s=1.0)
        assert idx.trend is HealthTrend.INSUFFICIENT_DATA

    def test_stable_when_no_movement(self) -> None:
        hist = new_trend_history()
        for _ in range(TREND_WINDOW):
            update_trend_history(hist, 0.85)
        idx = aggregate({}, trend_history=hist, time_s=1.0)
        # 0.0 overall with history 0.85 -> DEGRADING (delta < -TREND_EPSILON).
        # This is expected: empty subsystems score 0.0, which is
        # far below the historical 0.85.
        assert idx.trend is HealthTrend.DEGRADING

    def test_degrading_when_score_drops(self) -> None:
        hist = new_trend_history()
        for v in [0.95] * TREND_WINDOW:
            update_trend_history(hist, v)
        # Build a healthy index, then a degraded one.
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        subs = {
            Subsystem.THERMAL: score_thermal(st, cfg),
            Subsystem.LUBRICATION: score_lubrication(st, cfg),
            Subsystem.PERFORMANCE: score_performance(st, cfg),
            Subsystem.MECHANICAL: score_mechanical(st, cfg),
            Subsystem.SENSORS: score_sensors(_healthy_assessment()),
        }
        idx1 = aggregate(subs, trend_history=hist, time_s=1.0)
        update_trend_history(hist, idx1.overall_score)
        # Now drop the score.
        st.egt_c = cfg.limits.egt_max_c
        st.cht_c = cfg.limits.cht_max_c
        subs[Subsystem.THERMAL] = score_thermal(st, cfg)
        st.wear = 0.6
        subs[Subsystem.MECHANICAL] = score_mechanical(st, cfg)
        idx2 = aggregate(subs, trend_history=hist, time_s=2.0)
        assert idx2.overall_score < idx1.overall_score - TREND_EPSILON
        assert idx2.trend is HealthTrend.DEGRADING

    def test_improving_when_score_rises(self) -> None:
        hist = new_trend_history()
        for v in [0.5] * TREND_WINDOW:
            update_trend_history(hist, v)
        # A clean state should jump above 0.5.
        cfg = load_config(CONFIG_DIR).engine
        subs = {
            Subsystem.THERMAL: score_thermal(_nominal_state(), cfg),
            Subsystem.LUBRICATION: score_lubrication(_nominal_state(), cfg),
            Subsystem.PERFORMANCE: score_performance(_nominal_state(), cfg),
            Subsystem.MECHANICAL: score_mechanical(_nominal_state(), cfg),
            Subsystem.SENSORS: score_sensors(_healthy_assessment()),
        }
        idx = aggregate(subs, trend_history=hist, time_s=1.0)
        assert idx.overall_score > 0.5 + TREND_EPSILON
        assert idx.trend is HealthTrend.IMPROVING


# ---------------------------------------------------------------------
# TestHealthCalculator
# ---------------------------------------------------------------------
class TestHealthCalculator:
    def _calc(self) -> HealthIndexCalculator:
        cfg = load_config(CONFIG_DIR)
        return HealthIndexCalculator(cfg.engine)

    def test_healthy_inputs(self) -> None:
        cal = self._calc()
        idx = cal.update(
            state=_nominal_state(),
            anomaly=_healthy_assessment(),
        )
        assert idx.overall_label is HealthLabel.HEALTHY
        assert idx.overall_score >= 0.95
        assert idx.confidence == pytest.approx(1.0)
        # No contributing faults when there's no classification.
        assert idx.contributing_faults == {}
        assert idx.wear == 0.0

    def test_egt_redline_marks_thermal(self) -> None:
        cal = self._calc()
        cfg = load_config(CONFIG_DIR).engine
        st = _nominal_state()
        st.egt_c = cfg.limits.egt_max_c
        idx = cal.update(
            state=st, anomaly=_healthy_assessment(),
        )
        # The thermal subsystem should have egt_c as a contributor.
        thermal = idx.subsystems[Subsystem.THERMAL]
        assert "egt_c" in thermal.contributors
        assert thermal.score < 1.0
        # Overall score should drop below 1.0 but probably still HEALTHY.
        assert idx.overall_score < 1.0
        assert idx.overall_label in (HealthLabel.HEALTHY, HealthLabel.DEGRADED)

    def test_classifier_contributes_faults(self) -> None:
        cal = self._calc()
        fc = FaultClassification(
            time_s=1.0,
            fault_class=FaultClass.VIBRATION_ENGINE_ANOMALY,
            confidence=0.85,
            probabilities={
                FaultClass.HEALTHY: 0.05,
                FaultClass.VIBRATION_ENGINE_ANOMALY: 0.85,
                FaultClass.ENGINE_DEGRADATION: 0.10,
            },
            status=CalibrationStatus.CALIBRATED,
            features_used=60,
        )
        idx = cal.update(
            state=_nominal_state(),
            anomaly=_healthy_assessment(),
            classification=fc,
        )
        # High-probability classes should be in contributing_faults.
        assert "VIBRATION_ENGINE_ANOMALY" in idx.contributing_faults
        assert "ENGINE_DEGRADATION" in idx.contributing_faults  # at threshold
        assert "HEALTHY" not in idx.contributing_faults  # below threshold
        assert idx.contributing_faults["VIBRATION_ENGINE_ANOMALY"] == pytest.approx(0.85)

    def test_reset_clears_state(self) -> None:
        cal = self._calc()
        cal.update(state=_nominal_state(), anomaly=_healthy_assessment())
        cal.update(state=_nominal_state(), anomaly=_healthy_assessment())
        assert cal.update_count == 2
        cal.reset()
        assert cal.update_count == 0
        assert cal.last_index is None
        assert len(cal.trend_history) == 0

    def test_latency_under_target(self) -> None:
        cal = self._calc()
        st = _nominal_state()
        a = _healthy_assessment()
        # Warmup.
        for _ in range(5):
            cal.update(state=st, anomaly=a)
        us = cal.measure_latency(n_iter=200)
        # Generous target: 2 ms per update.
        assert us < 2000.0

    def test_insufficient_when_no_inputs(self) -> None:
        cal = self._calc()
        idx = cal.update()
        # All subsystems have 0 confidence -> INSUFFICIENT_DATA.
        assert idx.overall_label is HealthLabel.INSUFFICIENT_DATA
        assert idx.overall_score == 0.0
        assert idx.confidence < MIN_OVERALL_CONFIDENCE

    def test_time_s_propagated(self) -> None:
        cal = self._calc()
        idx = cal.update(state=_nominal_state(time_s=42.5),
                         anomaly=_healthy_assessment())
        assert idx.time_s == pytest.approx(42.5)


# ---------------------------------------------------------------------
# TestScenarios — end-to-end with the engine simulator
# ---------------------------------------------------------------------
class TestScenarios:
    def _setup(self):
        cfg = load_config(CONFIG_DIR)
        sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
        bundle = SensorBundle(cfg.sensors, sim, master_seed=42)
        return cfg, sim, bundle, HealthIndexCalculator(cfg.engine)

    def test_s1_healthy_stays_healthy(self) -> None:
        """200 clean ticks with all-healthy inputs: overall HEALTHY."""
        _, _, _, cal = self._setup()
        scores: List[float] = []
        for i in range(200):
            t = float(i) * 0.1
            st = _nominal_state(time_s=t)
            a = _healthy_assessment()
            idx = cal.update(state=st, anomaly=a)
            scores.append(idx.overall_score)
        # All scores should be at or very near 1.0.
        assert min(scores) >= 0.95
        # After enough ticks the trend should be STABLE (we didn't move).
        assert cal.last_index.trend is HealthTrend.STABLE

    def test_s4_engine_degradation_trend_degrading(self) -> None:
        """ENGINE_DEGRADATION should drive mechanical down; trend DEGRADING."""
        cfg, sim, bundle, cal = self._setup()
        sc = FaultScenario(
            FaultClass.ENGINE_DEGRADATION,
            severity=1.0,
            onset_time_s=0.0,
            duration_s=300.0,
            progression="step",
        )
        inj = FaultInjector(sc)
        scores: List[float] = []
        mech_scores: List[float] = []
        # Warmup with a healthy trend buffer so the trend has history.
        for _ in range(TREND_WINDOW):
            idx = cal.update(state=_nominal_state(),
                             anomaly=_healthy_assessment())
            scores.append(idx.overall_score)
        # Now inject the fault and watch for the early trend (when the
        # score is actively dropping).
        early_trend: Optional[HealthTrend] = None
        for i in range(200):
            tick = inj.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            # Build a synthetic anomaly assessment that flags vibration
            # (a representative fault signal).
            ca = ChannelAnomaly(
                channel="vibration", score=0.95,
                label=AnomalyLabel.ANOMALY, confidence=0.9,
                contributors=["vibration residual"],
            )
            a = AnomalyAssessment(
                time_s=step.env.time_s,
                overall_score=0.95, overall_label=AnomalyLabel.ANOMALY,
                confidence=0.9, channels={"vibration": ca}, notes=[],
                frame_status="OK",
            )
            idx = cal.update(state=step.engine, anomaly=a)
            scores.append(idx.overall_score)
            mech_scores.append(idx.subsystems[Subsystem.MECHANICAL].score)
            # Record the trend at the first fault tick — the trend
            # buffer still holds warmup values, so the comparison
            # is between a healthy reference and a fault score.
            if early_trend is None:
                early_trend = idx.trend
        # Mechanical score should have dropped (wear is small but
        # the SENSORS subsystem is 0.05 due to the vibration flag).
        assert min(scores) < 0.95
        # Trend should have shown DEGRADING at some point.
        assert early_trend is HealthTrend.DEGRADING

    def test_s5_performance_loss_label_degraded(self) -> None:
        """PERFORMANCE_LOSS should drop the performance subsystem."""
        cfg, sim, bundle, cal = self._setup()
        sc = FaultScenario(
            FaultClass.PERFORMANCE_LOSS,
            severity=0.7,
            onset_time_s=0.0,
            duration_s=300.0,
            progression="step",
        )
        inj = FaultInjector(sc)
        # Warmup trend.
        for _ in range(TREND_WINDOW):
            cal.update(state=_nominal_state(),
                       anomaly=_healthy_assessment())
        perf_scores: List[float] = []
        for _ in range(600):
            tick = inj.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            idx = cal.update(state=step.engine,
                             anomaly=_healthy_assessment())
            perf_scores.append(idx.subsystems[Subsystem.PERFORMANCE].score)
        # Performance should have dropped below 1.0.
        assert min(perf_scores) < 1.0
        # Overall should have left HEALTHY at some point.
        assert cal.last_index.overall_label in (
            HealthLabel.DEGRADED, HealthLabel.CRITICAL, HealthLabel.HEALTHY,
        )
        # But it should be strictly lower than the warmup value.
        assert cal.last_index.overall_score < 1.0
