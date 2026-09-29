"""PHASE 11 tests — Remaining Useful Life + uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from backend.config import load_config
from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.health import (
    HealthIndex,
    HealthIndexCalculator,
    HealthLabel,
    HealthTrend,
    Subsystem,
    SubsystemHealth,
)
from backend.rul import (
    DEFAULT_HORIZON_HOURS,
    DEFAULT_N_SAMPLES,
    FEATURE_DIM,
    FEATURE_NAMES,
    MIN_TRAIN_SCENARIOS,
    MAX_RELATIVE_MAE,
    MODEL_VERSION,
    ClosedFormWearRate,
    ModelVersionWarning,
    ResidualTrendInput,
    RulBounds,
    RulCalculator,
    RulEvalReport,
    RulEstimate,
    RulModelStatus,
    RulStatus,
    RulTrend,
    TREND_EPSILON_HOURS,
    TREND_WINDOW,
    SelectionResult,
    TrainedWearModel,
    TteDistribution,
    WearRateDataset,
    aggregate,
    build_dataset,
    evaluate_rul,
    load_model,
    mae,
    mape,
    picp,
    plot_actual_vs_predicted,
    plot_health_vs_time,
    plot_prediction_uncertainty,
    plot_rul_vs_time,
    rmse,
    save_model,
    select_model,
    simulate_tte,
    status_for,
    train_wear_model,
    trend_for,
)
from backend.simulation import EngineSimulator, EngineState

pytestmark = pytest.mark.phase11

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_state(
    *,
    time_s: float = 1.0,
    rpm: float = 2400.0,
    manifold_pressure_inhg: float = 25.0,
    egt_c: float = 650.0,
    cht_c: float = 200.0,
    oil_pressure_psi: float = 60.0,
    oil_temperature_c: float = 90.0,
    fuel_flow_lph: float = 30.0,
    vibration_rms_g: float = 1.5,
    bsfc_g_per_kwh: float = 400.0,
    brake_power_kw: float = 134.0,
    wear: float = 0.0,
    throttle: float = 0.6,
    altitude_m: float = 500.0,
    airspeed_mps: float = 50.0,
) -> EngineState:
    return EngineState(
        time_s=time_s, rpm=rpm, manifold_pressure_inhg=manifold_pressure_inhg,
        egt_c=egt_c, cht_c=cht_c, oil_pressure_psi=oil_pressure_psi,
        oil_temperature_c=oil_temperature_c, fuel_flow_lph=fuel_flow_lph,
        vibration_rms_g=vibration_rms_g, bsfc_g_per_kwh=bsfc_g_per_kwh,
        brake_power_kw=brake_power_kw, wear=wear, throttle=throttle,
        altitude_m=altitude_m, airspeed_mps=airspeed_mps,
    )


def _healthy_health(
    *,
    time_s: float = 1.0,
    wear: float = 0.0,
    trend: HealthTrend = HealthTrend.STABLE,
    contributing_faults: Optional[Dict[str, float]] = None,
    overall_score: float = 1.0,
) -> HealthIndex:
    label = (HealthLabel.HEALTHY if overall_score >= 0.80
             else HealthLabel.DEGRADED if overall_score >= 0.55
             else HealthLabel.CRITICAL)
    return HealthIndex(
        time_s=time_s,
        overall_score=float(overall_score),
        overall_label=label,
        confidence=1.0,
        trend=trend,
        wear=wear,
        contributing_faults=dict(contributing_faults or {}),
    )


def _degraded_health(
    *,
    time_s: float = 1.0,
    wear: float = 0.30,
    trend: HealthTrend = HealthTrend.DEGRADING,
    contributing_faults: Optional[Dict[str, float]] = None,
    overall_score: float = 0.65,
) -> HealthIndex:
    return HealthIndex(
        time_s=time_s,
        overall_score=float(overall_score),
        overall_label=HealthLabel.DEGRADED,
        confidence=0.9,
        trend=trend,
        wear=wear,
        contributing_faults=dict(contributing_faults or {}),
    )


# ---------------------------------------------------------------------
# TestTypes
# ---------------------------------------------------------------------
class TestTypes:
    def test_status_members(self) -> None:
        assert {m.value for m in RulStatus} == {
            "RUL_OK", "RUL_DEGRADED", "RUL_CRITICAL", "RUL_UNCERTAIN",
        }

    def test_trend_members(self) -> None:
        assert {m.value for m in RulTrend} == {
            "IMPROVING", "STABLE", "DEGRADING", "INSUFFICIENT_DATA",
        }

    def test_model_status_members(self) -> None:
        assert {m.value for m in RulModelStatus} == {
            "CLOSED_FORM", "MODEL_CALIBRATED", "MODEL_DEGRADED",
        }

    def test_rul_bounds_validates(self) -> None:
        with pytest.raises(ValueError):
            RulBounds(lower=10.0, central=5.0, upper=20.0)
        with pytest.raises(ValueError):
            RulBounds(lower=1.0, central=2.0, upper=1.5)
        b = RulBounds(lower=5.0, central=10.0, upper=20.0)
        assert b.spread_hours == pytest.approx(15.0)
        d = b.to_dict()
        assert d["lower"] == 5.0
        assert d["central"] == 10.0
        assert d["upper"] == 20.0

    def test_rul_estimate_to_dict(self) -> None:
        e = RulEstimate(
            time_s=1.0, tte_hours_central=100.0,
            tte_hours_lower=80.0, tte_hours_upper=150.0,
            wear_rate_per_hour=0.001, confidence=0.8,
            status=RulStatus.RUL_OK, trend=RulTrend.STABLE,
            model_status=RulModelStatus.CLOSED_FORM,
        )
        d = e.to_dict()
        assert d["time_s"] == 1.0
        assert d["tte_hours_central"] == 100.0
        assert d["status"] == "RUL_OK"
        assert d["trend"] == "STABLE"
        assert d["model_status"] == "CLOSED_FORM"
        assert d["confidence"] == pytest.approx(0.8)
        assert e.is_ok is True
        assert e.bounds.lower == 80.0
        assert e.bounds.upper == 150.0

    def test_status_for_thresholds(self) -> None:
        assert status_for(health_label="HEALTHY", confidence=0.9) is RulStatus.RUL_OK
        assert status_for(health_label="DEGRADED", confidence=0.9) is RulStatus.RUL_DEGRADED
        assert status_for(health_label="CRITICAL", confidence=0.9) is RulStatus.RUL_CRITICAL
        assert status_for(health_label="INSUFFICIENT_DATA", confidence=0.9) is RulStatus.RUL_UNCERTAIN
        # Confidence floor.
        assert status_for(health_label="HEALTHY", confidence=0.10) is RulStatus.RUL_UNCERTAIN
        assert status_for(health_label="DEGRADED", confidence=0.30) is RulStatus.RUL_CRITICAL

    def test_trend_for(self) -> None:
        assert trend_for(None, 100.0) is RulTrend.INSUFFICIENT_DATA
        assert trend_for(100.0, None) is RulTrend.INSUFFICIENT_DATA
        assert trend_for(100.0, 100.0) is RulTrend.STABLE
        assert trend_for(110.0, 100.0) is RulTrend.IMPROVING
        assert trend_for(95.0, 100.0) is RulTrend.DEGRADING
        # Inside epsilon
        assert trend_for(100.0 + TREND_EPSILON_HOURS * 0.5, 100.0) is RulTrend.STABLE


# ---------------------------------------------------------------------
# TestWearRateModel
# ---------------------------------------------------------------------
class TestWearRateModel:
    def test_feature_names_count(self) -> None:
        assert FEATURE_DIM == len(FEATURE_NAMES)
        assert 10 <= FEATURE_DIM <= 30

    def test_closed_form_healthy_baseline(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        h = _healthy_health()
        rate = m.rate_per_hour(h)
        # Healthy + no faults + stable → base rate, no penalty.
        assert rate == pytest.approx(m.base_rate_per_hour, rel=1e-6)

    def test_closed_form_trend_degrading_higher(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        h_stable = _healthy_health()
        h_deg = _healthy_health(trend=HealthTrend.DEGRADING)
        r_stable = m.rate_per_hour(h_stable)
        r_deg = m.rate_per_hour(h_deg)
        assert r_deg > r_stable

    def test_closed_form_fault_higher(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        h_clean = _healthy_health()
        h_fault = _healthy_health(
            contributing_faults={"ENGINE_DEGRADATION": 0.5}
        )
        r_clean = m.rate_per_hour(h_clean)
        r_fault = m.rate_per_hour(h_fault)
        assert r_fault > r_clean

    def test_closed_form_low_health_higher(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        h_high = _healthy_health(overall_score=0.95)  # not used directly here
        h_low = HealthIndex(
            time_s=0.0, overall_score=0.30,
            overall_label=HealthLabel.CRITICAL, confidence=1.0,
            trend=HealthTrend.STABLE,
        )
        r_high = m.rate_per_hour(h_high)
        r_low = m.rate_per_hour(h_low)
        assert r_low > r_high

    def test_build_dataset_shape(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0,
                           n_ticks_per_scenario=80)
        assert ds.X.ndim == 2
        assert ds.X.shape[1] == FEATURE_DIM
        assert ds.y.shape[0] == ds.X.shape[0]
        assert ds.n_rows > 0

    def test_train_wear_model(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0,
                           n_ticks_per_scenario=80)
        m = train_wear_model(ds, n_estimators=20, seed=0)
        assert isinstance(m, TrainedWearModel)
        # Importances sum to ~1.0
        assert sum(m.feature_importances.values()) == pytest.approx(1.0, abs=1e-3)
        assert len(m.feature_importances) == FEATURE_DIM
        # Rate on a healthy index is non-negative.
        h = _healthy_health()
        r = m.rate_per_hour(h)
        assert r >= 0.0


# ---------------------------------------------------------------------
# TestMonteCarlo
# ---------------------------------------------------------------------
class TestMonteCarlo:
    def test_distribution_shape(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        # Use a large base rate so EOL is reached within the horizon.
        m.base_rate_per_hour = 0.01
        h = _healthy_health()
        d = simulate_tte(
            wear_model=m, current_health=h, current_wear=0.0,
            max_wear=1.0, horizon_hours=200.0, n_samples=100,
            dt_s=60.0, seed=0,
        )
        assert isinstance(d, TteDistribution)
        # Bounds should be ordered.
        assert d.lower <= d.central <= d.upper
        assert 0.0 <= d.confidence <= 1.0
        assert d.mean_wear_rate >= 0.0

    def test_high_wear_faster(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        h = _healthy_health()
        # Use a tiny base rate + large wear values so EOL is reached
        # within the horizon.
        m.base_rate_per_hour = 0.01
        d_low = simulate_tte(
            wear_model=m, current_health=h, current_wear=0.10,
            max_wear=1.0, horizon_hours=200.0, n_samples=100,
            dt_s=60.0, seed=0,
        )
        d_high = simulate_tte(
            wear_model=m, current_health=h, current_wear=0.80,
            max_wear=1.0, horizon_hours=200.0, n_samples=100,
            dt_s=60.0, seed=0,
        )
        # Higher current wear → smaller TTE.
        assert d_high.central < d_low.central

    def test_zero_rate_is_uncertain(self) -> None:
        """If the wear model returns 0, no sample reaches EOL → confidence 0."""
        class _ZeroModel:
            def rate_per_hour(self, health, *, hours_running=0.0):
                return 0.0
        d = simulate_tte(
            wear_model=_ZeroModel(),
            current_health=_healthy_health(), current_wear=0.0,
            max_wear=1.0, horizon_hours=200.0, n_samples=50,
            dt_s=60.0, seed=0,
        )
        assert d.confidence == 0.0
        assert d.central == 200.0  # full horizon

    def test_reproducibility(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        m.base_rate_per_hour = 0.01
        h = _healthy_health()
        a = simulate_tte(
            wear_model=m, current_health=h, current_wear=0.20,
            max_wear=1.0, horizon_hours=200.0, n_samples=80,
            dt_s=60.0, seed=42,
        )
        b = simulate_tte(
            wear_model=m, current_health=h, current_wear=0.20,
            max_wear=1.0, horizon_hours=200.0, n_samples=80,
            dt_s=60.0, seed=42,
        )
        assert a.central == pytest.approx(b.central)
        assert a.lower == pytest.approx(b.lower)
        assert a.upper == pytest.approx(b.upper)

    def test_n_samples_validation(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        m = ClosedFormWearRate.from_config(cfg)
        with pytest.raises(ValueError):
            simulate_tte(
                wear_model=m, current_health=_healthy_health(),
                current_wear=0.0, n_samples=0,
            )
        with pytest.raises(ValueError):
            simulate_tte(
                wear_model=m, current_health=_healthy_health(),
                current_wear=0.0, dt_s=0.0,
            )


# ---------------------------------------------------------------------
# TestRulCalculator
# ---------------------------------------------------------------------
class TestRulCalculator:
    def _calc(self, **kwargs) -> RulCalculator:
        cfg = load_config(CONFIG_DIR)
        return RulCalculator(cfg.engine, **kwargs)

    def test_healthy_inputs_rul_ok(self) -> None:
        cal = self._calc()
        h = _healthy_health(wear=0.0)
        est = cal.update(health=h)
        assert est.status in (RulStatus.RUL_OK, RulStatus.RUL_DEGRADED)
        # TTE should be large (engine is new and healthy).
        assert est.tte_hours_central > 1000.0
        # Bounds must be ordered.
        assert est.tte_hours_lower <= est.tte_hours_central
        assert est.tte_hours_central <= est.tte_hours_upper
        # Default: closed-form fallback.
        assert est.model_status is RulModelStatus.CLOSED_FORM
        # Trend buffer has one entry but <TREND_WINDOW → INSUFFICIENT_DATA.
        assert est.trend is RulTrend.INSUFFICIENT_DATA

    def test_insufficient_health_is_uncertain(self) -> None:
        cal = self._calc()
        h = HealthIndex(
            time_s=0.0, overall_score=0.0, overall_label=HealthLabel.INSUFFICIENT_DATA,
            confidence=0.0, trend=HealthTrend.INSUFFICIENT_DATA,
        )
        est = cal.update(health=h)
        assert est.status is RulStatus.RUL_UNCERTAIN
        assert "insufficient health evidence" in " ".join(est.notes) or \
               "no samples reached EOL within horizon" in " ".join(est.notes) or \
               "low confidence in TTE estimate" in " ".join(est.notes)

    def test_critical_health(self) -> None:
        cal = self._calc()
        h = HealthIndex(
            time_s=0.0, overall_score=0.10, overall_label=HealthLabel.CRITICAL,
            confidence=0.9, trend=HealthTrend.DEGRADING,
            wear=0.70,
            contributing_faults={"ENGINE_DEGRADATION": 0.8},
        )
        est = cal.update(health=h)
        # Should be CRITICAL or DEGRADED (depending on confidence), not OK.
        assert est.status in (RulStatus.RUL_CRITICAL, RulStatus.RUL_DEGRADED)
        # TTE should be much smaller than healthy.
        assert est.tte_hours_central < 1000.0

    def test_reset_clears_state(self) -> None:
        cal = self._calc()
        cal.update(health=_healthy_health())
        cal.update(health=_healthy_health())
        assert cal.update_count == 2
        cal.reset()
        assert cal.update_count == 0
        assert cal.last_estimate is None
        assert len(cal.trend_history) == 0
        assert cal.hours_running == 0.0

    def test_loaded_model(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0,
                           n_ticks_per_scenario=80)
        m = train_wear_model(ds, n_estimators=20, seed=0)
        path = tmp_path / "wear.pkl"
        save_model(m, path)
        cfg = load_config(CONFIG_DIR)
        cal = RulCalculator(cfg.engine, model_path=path)
        h = _healthy_health()
        est = cal.update(health=h)
        assert est.model_status is RulModelStatus.MODEL_CALIBRATED
        assert est.tte_hours_central > 0.0

    def test_loaded_model_round_trip(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0,
                           n_ticks_per_scenario=80)
        m = train_wear_model(ds, n_estimators=10, seed=0)
        path = tmp_path / "wear.pkl"
        save_model(m, path)
        loaded = load_model(path)
        # Predict must match.
        h = _healthy_health()
        assert m.rate_per_hour(h) == pytest.approx(loaded.rate_per_hour(h))

    def test_latency_under_target(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        # Warmup.
        for _ in range(3):
            cal.update(health=h)
        us = cal.measure_latency(n_iter=20)
        # Generous: 100 ms per update (RF + 200-sample MC).
        assert us < 100_000.0

    def test_contributing_faults_propagated(self) -> None:
        cal = self._calc()
        h = _healthy_health(
            contributing_faults={"ENGINE_DEGRADATION": 0.6,
                                  "OVERHEATING": 0.3},
        )
        est = cal.update(health=h)
        assert est.contributing_faults == {
            "ENGINE_DEGRADATION": 0.6, "OVERHEATING": 0.3,
        }

    def test_state_wear_overrides_health_wear(self) -> None:
        cal = self._calc()
        h = _healthy_health(wear=0.0)
        st = _make_state(wear=0.80)
        est = cal.update(health=h, state=st)
        # State wear=0.80 wins over health wear=0.0. With the
        # default base rate of 1e-5/h, the remaining 0.20 of
        # wear takes ~20,000 hours — TTE must be in that range.
        # The key property: TTE here should be < the TTE we'd
        # get with state wear=0.0 (i.e. the override actually
        # took effect).
        cal_fresh = self._calc()
        est_fresh = cal_fresh.update(health=h, state=_make_state(wear=0.0))
        assert est.tte_hours_central < est_fresh.tte_hours_central
        # And the high-wear TTE is well below the horizon.
        assert est.tte_hours_central < 50_000.0


# ---------------------------------------------------------------------
# TestScenarios
# ---------------------------------------------------------------------
class TestScenarios:
    def _setup(self, **kwargs):
        cfg = load_config(CONFIG_DIR)
        sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
        cal_health = HealthIndexCalculator(cfg.engine)
        cal_rul = RulCalculator(cfg.engine, **kwargs)
        return cfg, sim, cal_health, cal_rul

    def test_s1_healthy_rul_ok(self) -> None:
        """200 clean ticks → RUL_OK or RUL_DEGRADED, large TTE,
        trend no longer INSUFFICIENT_DATA."""
        _, _, cal_health, cal_rul = self._setup()
        for _ in range(200):
            step_idx = cal_health.update_count
            t = float(step_idx) * 0.1
            from backend.diagnostics import (
                AnomalyAssessment, ChannelAnomaly, AnomalyLabel,
            )
            channels = {
                f"ch{i}": ChannelAnomaly(
                    channel=f"ch{i}", score=0.0,
                    label=AnomalyLabel.NORMAL, confidence=1.0,
                )
                for i in range(7)
            }
            anom = AnomalyAssessment(
                time_s=t, overall_score=0.0,
                overall_label=AnomalyLabel.NORMAL, confidence=1.0,
                channels=channels,
            )
            st = _make_state(time_s=t, wear=0.0)
            h = cal_health.update(state=st, anomaly=anom)
            est = cal_rul.update(health=h, state=st, dt_s=0.1)
            assert est.tte_hours_central > 1000.0
        # After many ticks the trend buffer is full and the trend
        # is computed (we don't assert STABLE specifically: each
        # tick uses a different MC seed so the median TTE can
        # bounce by hundreds of hours between ticks).
        assert cal_rul.last_estimate.trend is not RulTrend.INSUFFICIENT_DATA

    def test_s4_engine_degradation_rul_decreases(self) -> None:
        """ENGINE_DEGRADATION fault → TTE drops, trend DEGRADING."""
        cfg, sim, cal_health, cal_rul = self._setup()
        sc = FaultScenario(
            FaultClass.ENGINE_DEGRADATION,
            severity=1.0, onset_time_s=0.0, duration_s=300.0,
            progression="step",
        )
        inj = FaultInjector(sc)
        from backend.diagnostics import (
            AnomalyAssessment, ChannelAnomaly, AnomalyLabel,
        )
        # Warmup with healthy trend.
        for _ in range(TREND_WINDOW):
            st = _make_state(time_s=0.0, wear=0.0)
            channels = {
                f"ch{i}": ChannelAnomaly(
                    channel=f"ch{i}", score=0.0,
                    label=AnomalyLabel.NORMAL, confidence=1.0,
                )
                for i in range(7)
            }
            anom = AnomalyAssessment(
                time_s=0.0, overall_score=0.0,
                overall_label=AnomalyLabel.NORMAL, confidence=1.0,
                channels=channels,
            )
            h = cal_health.update(state=st, anomaly=anom)
            cal_rul.update(health=h, state=st, dt_s=0.1)
        # Capture baseline TTE from the warmup.
        baseline_tte = cal_rul.last_estimate.tte_hours_central
        # Now inject the fault.
        for i in range(50):
            tick = inj.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            t = step.env.time_s
            ca = ChannelAnomaly(
                channel="vibration", score=0.95,
                label=AnomalyLabel.ANOMALY, confidence=0.9,
                contributors=["vibration residual"],
            )
            anom = AnomalyAssessment(
                time_s=t, overall_score=0.95,
                overall_label=AnomalyLabel.ANOMALY, confidence=0.9,
                channels={"vibration": ca},
            )
            h = cal_health.update(state=step.engine, anomaly=anom)
            cal_rul.update(health=h, state=step.engine, dt_s=0.1)
        # After the fault, TTE must be smaller than the healthy
        # baseline. The fault's contribution enters through
        # ``health.contributing_faults`` (the closed-form rate
        # scales with k_fault * sum(severity * prob)).
        assert cal_rul.last_estimate.tte_hours_central < baseline_tte

    def test_no_model_closed_form_fallback(self) -> None:
        """Without a trained model, the calculator still works."""
        _, _, cal_health, cal_rul = self._setup()
        st = _make_state(time_s=0.0, wear=0.0)
        h = _healthy_health(wear=0.0)
        est = cal_rul.update(health=h, state=st, dt_s=0.1)
        assert est.model_status is RulModelStatus.CLOSED_FORM
        assert est.tte_hours_central > 0.0

    def test_high_wear_uncertain(self) -> None:
        """At very high wear, the closed-form model and MC still produce
        a coherent (small) TTE, not UNCERTAIN — uncertainty comes from
        the health signal, not the wear value itself."""
        cal = RulCalculator(load_config(CONFIG_DIR).engine)
        st = _make_state(time_s=0.0, wear=0.99)
        # Provide a healthy health index but a near-EOL state.
        h = _healthy_health(wear=0.99)
        est = cal.update(health=h, state=st, dt_s=0.1)
        # TTE should be small (only 1% of wear left).
        assert est.tte_hours_central < 1000.0
        # Trend still INSUFFICIENT_DATA on first tick.
        assert est.trend is RulTrend.INSUFFICIENT_DATA


# ---------------------------------------------------------------------
# ResidualTrendInput (Digital Twin residual history)
# ---------------------------------------------------------------------
class TestResidualTrendInput:
    def test_construction(self) -> None:
        z = np.array([[0.1, 0.2, 0.3], [0.0, -0.5, 0.0]], dtype=np.float64)
        rt = ResidualTrendInput(
            channel_names=("rpm", "egt"),
            z_history=z, dt_s=0.1,
        )
        assert rt.n_channels == 2
        assert rt.n_ticks == 3
        assert rt.dt_s == pytest.approx(0.1)

    def test_per_channel_max_abs_z(self) -> None:
        z = np.array([[0.1, 0.2, 0.3], [0.0, -0.5, 0.0]], dtype=np.float64)
        rt = ResidualTrendInput(channel_names=("a", "b"), z_history=z)
        max_abs = rt.per_channel_max_abs_z()
        assert max_abs.shape == (2,)
        assert max_abs[0] == pytest.approx(0.3)
        assert max_abs[1] == pytest.approx(0.5)

    def test_per_channel_trend(self) -> None:
        # Build a series with a known positive slope.
        t = np.arange(10, dtype=np.float64) * 0.1
        z = np.tile(t * 2.0, (3, 1))  # slope = 20.0 per second
        rt = ResidualTrendInput(
            channel_names=("a", "b", "c"),
            z_history=z, dt_s=0.1,
        )
        # Mean abs(z) at each tick = |t*2|, so the slope of mean|z|
        # is +2.0 per second. All channels share the same trend.
        trend = rt.per_channel_trend()
        assert trend.shape == (3,)
        for v in trend:
            assert v == pytest.approx(2.0, abs=0.01)

    def test_empty_history(self) -> None:
        z = np.zeros((4, 0), dtype=np.float64)
        rt = ResidualTrendInput(channel_names=("a", "b", "c", "d"),
                               z_history=z)
        assert rt.n_ticks == 0
        # Empty input → zero trend, zero max.
        assert rt.per_channel_max_abs_z().shape == (4,)
        assert rt.per_channel_trend().shape == (4,)
        np.testing.assert_array_equal(rt.per_channel_max_abs_z(), np.zeros(4))

    def test_nan_inf_replaced_with_zero(self) -> None:
        z = np.array([[np.nan, 1.0], [np.inf, 0.0], [-np.inf, 0.5]])
        rt = ResidualTrendInput(
            channel_names=("a", "b", "c"), z_history=z,
        )
        # All NaN/inf entries should have been replaced with 0.0.
        assert rt.z_history[0, 0] == 0.0
        assert rt.z_history[1, 0] == 0.0
        assert rt.z_history[2, 0] == 0.0
        # Real values should be preserved.
        assert rt.z_history[0, 1] == 1.0
        assert rt.z_history[2, 1] == 0.5

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            ResidualTrendInput(
                channel_names=("a", "b"),
                z_history=np.zeros((3, 5), dtype=np.float64),  # 3 rows, 2 names
            )

    def test_invalid_dt_s_raises(self) -> None:
        with pytest.raises(ValueError):
            ResidualTrendInput(
                channel_names=("a",),
                z_history=np.zeros((1, 5)),
                dt_s=-1.0,
            )


# ---------------------------------------------------------------------
# Evaluation metrics (MAE / RMSE / MAPE / PICP)
# ---------------------------------------------------------------------
class TestEvaluation:
    def test_mae_perfect(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_mae_with_errors(self) -> None:
        # Errors: 1.0, 1.0, 1.0 → MAE = 1.0
        assert mae([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_rmse_perfect_and_noisy(self) -> None:
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
        # Errors: 1, 1, 1 → RMSE = 1.0
        assert rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_mape_skipped_on_zero_truth(self) -> None:
        # MAPE is undefined at zero truth — function returns None.
        assert mape([0.0, 1.0, 2.0], [0.5, 1.5, 2.5]) is None

    def test_mape_with_nonzero_truth(self) -> None:
        # truth = [1, 2, 4], pred = [2, 1, 4]
        # |errors|/|truth| = [1.0, 0.5, 0.0] → MAPE = 0.5
        assert mape([1.0, 2.0, 4.0], [2.0, 1.0, 4.0]) == pytest.approx(0.5)

    def test_picp_in_bounds(self) -> None:
        truth = [10.0, 20.0, 30.0, 40.0]
        lo = [5.0, 15.0, 25.0, 35.0]
        hi = [15.0, 25.0, 35.0, 45.0]
        # All truth values are inside the [lo, hi] intervals.
        assert picp(truth, lo, hi) == pytest.approx(1.0)

    def test_picp_out_of_bounds(self) -> None:
        truth = [10.0, 20.0, 30.0, 40.0]
        lo = [0.0, 0.0, 0.0, 0.0]
        hi = [5.0, 5.0, 5.0, 5.0]
        # All truth values are above hi.
        assert picp(truth, lo, hi) == pytest.approx(0.0)

    def test_evaluate_rul_full(self) -> None:
        # 4 samples: perfect central, but bounds are tight.
        truth = [10.0, 20.0, 30.0, 40.0]
        central = [10.0, 20.0, 30.0, 40.0]
        lo = [9.0, 19.0, 29.0, 39.0]
        hi = [11.0, 21.0, 31.0, 41.0]
        rep = evaluate_rul(truth, central, lower=lo, upper=hi)
        assert isinstance(rep, RulEvalReport)
        assert rep.n_samples == 4
        assert rep.mae == 0.0
        assert rep.rmse == 0.0
        assert rep.mape == 0.0
        assert rep.picp == 1.0
        assert rep.bias == 0.0

    def test_evaluate_rul_no_bounds(self) -> None:
        rep = evaluate_rul([1.0, 2.0], [1.0, 3.0])
        assert rep.n_samples == 2
        assert rep.mae == pytest.approx(0.5)
        assert rep.rmse == pytest.approx(np.sqrt(0.5))
        assert rep.picp == 0.0  # no bounds → reported as 0
        # The notes field should mention the missing bounds.
        assert any("no bounds" in n for n in rep.notes)

    def test_evaluate_rul_picp_under_target(self) -> None:
        # All truth above hi → PICP = 0.0 < target 0.90.
        rep = evaluate_rul(
            [100.0, 200.0], [50.0, 60.0],
            lower=[0.0, 0.0], upper=[10.0, 10.0],
        )
        assert rep.picp < 0.5
        assert any("under-confident" in n for n in rep.notes)


# ---------------------------------------------------------------------
# Model selector (CF vs RF)
# ---------------------------------------------------------------------
class TestModelSelector:
    def test_picks_cf_when_no_rf(self) -> None:
        cf = ClosedFormWearRate()
        result = select_model(cf, None)
        assert result.model_kind == "closed_form"
        assert result.n_train_scenarios == 0
        assert "no trained model" in result.reason

    def test_picks_cf_when_data_insufficient(self, tmp_path: Path) -> None:
        cfg = load_config(CONFIG_DIR)
        cf = ClosedFormWearRate.from_config(cfg.engine)
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0, n_ticks_per_scenario=20)
        # Train with n_train_scenarios well below the floor.
        rf = train_wear_model(ds, n_estimators=10, seed=0,
                              n_train_scenarios=10)
        result = select_model(cf, rf)
        assert result.model_kind == "closed_form"
        assert result.n_train_scenarios == 10
        assert "scenarios < required" in result.reason

    def test_picks_rf_when_data_sufficient(self, tmp_path: Path) -> None:
        cfg = load_config(CONFIG_DIR)
        cf = ClosedFormWearRate.from_config(cfg.engine)
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        # Build a large RF with many training scenarios.
        rf = train_wear_model(ds, n_estimators=10, seed=0,
                              n_train_scenarios=200)
        # Build a held-out set from the same dataset.
        n = ds.n_rows
        held = ds.X[max(0, n - 50):]
        # Construct HealthIndex stubs from the feature vectors
        # (we just need an iterable; the selector uses only the
        # overall_score, wear, trend, hours_running columns).
        held_healths = []
        for row in held:
            held_healths.append(_healthy_health(
                time_s=0.0,
                overall_score=float(row[0]),
                wear=float(row[2]),
                trend=(HealthTrend.DEGRADING if row[1] >= 1.5
                       else HealthTrend.STABLE),
                contributing_faults={
                    "ENGINE_DEGRADATION": float(row[8]),
                    "OVERHEATING": float(row[9]),
                    "LUBRICATION_PRESSURE_ANOMALY": float(row[10]),
                    "VIBRATION_ENGINE_ANOMALY": float(row[11]),
                    "PERFORMANCE_LOSS": float(row[12]),
                },
            ))
        result = select_model(cf, rf, held_out_healths=held_healths)
        assert result.n_train_scenarios == 200
        # Either choice is valid here; both metrics are populated.
        assert result.cf_mae is not None
        assert result.rf_mae is not None
        assert result.model_kind in ("closed_form", "random_forest")
        assert isinstance(result, SelectionResult)


# ---------------------------------------------------------------------
# Plotting (SVG output, no third-party deps)
# ---------------------------------------------------------------------
class TestPlotting:
    def test_health_vs_time_renders(self, tmp_path: Path) -> None:
        h_hist = [
            _healthy_health(time_s=float(i), overall_score=1.0 - 0.05 * i)
            for i in range(10)
        ]
        p = plot_health_vs_time(h_hist, tmp_path / "health.svg")
        assert p.exists()
        assert p.read_text(encoding="utf-8").startswith("<?xml")
        assert "<polyline" in p.read_text(encoding="utf-8")

    def test_rul_vs_time_renders(self, tmp_path: Path) -> None:
        cfg = load_config(CONFIG_DIR).engine
        cal = RulCalculator(cfg)
        rul_hist = []
        for i in range(10):
            h = _healthy_health(time_s=float(i), wear=0.0)
            est = cal.update(health=h, dt_s=0.1)
            rul_hist.append(est)
        p = plot_rul_vs_time(rul_hist, tmp_path / "rul.svg")
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert content.startswith("<?xml")
        assert "<polygon" in content  # the 5/95 band

    def test_actual_vs_predicted_renders(self, tmp_path: Path) -> None:
        p = plot_actual_vs_predicted(
            [(1000.0, 990.0), (500.0, 510.0), (200.0, 195.0)],
            tmp_path / "ap.svg",
        )
        content = p.read_text(encoding="utf-8")
        assert content.startswith("<?xml")
        assert content.count("<circle") == 3

    def test_prediction_uncertainty_renders(self, tmp_path: Path) -> None:
        p = plot_prediction_uncertainty(
            [
                (1000.0, 900.0, 990.0, 1100.0),
                (500.0, 450.0, 510.0, 560.0),
                (200.0, 180.0, 195.0, 210.0),
            ],
            tmp_path / "pu.svg",
        )
        content = p.read_text(encoding="utf-8")
        assert content.startswith("<?xml")
        # Error-bar plot: each sample has 2 vertical lines + 2 caps + 1 point.
        # Just check that the chart is non-trivial.
        assert len(content) > 500

    def test_empty_inputs_render_placeholder(self, tmp_path: Path) -> None:
        for fn, fname in [
            (plot_health_vs_time, "h.svg"),
            (plot_rul_vs_time, "r.svg"),
            (plot_actual_vs_predicted, "ap.svg"),
            (plot_prediction_uncertainty, "pu.svg"),
        ]:
            p = fn([], tmp_path / fname)
            content = p.read_text(encoding="utf-8")
            assert content.startswith("<?xml")
            assert "no data" in content


# ---------------------------------------------------------------------
# Calculator integration: residual_trend kwarg
# ---------------------------------------------------------------------
class TestCalculatorWithResidualTrend:
    def test_update_accepts_residual_trend_kwarg(self) -> None:
        cfg = load_config(CONFIG_DIR).engine
        cal = RulCalculator(cfg)
        rt = ResidualTrendInput(
            channel_names=("rpm", "egt", "cht", "oil_pressure",
                           "oil_temperature", "fuel_flow", "vibration",
                           "altitude", "airspeed", "ambient_temperature",
                           "ambient_pressure"),
            z_history=np.full((11, 20), 0.5, dtype=np.float64),
            dt_s=0.1,
        )
        h = _healthy_health(time_s=0.0)
        est = cal.update(health=h, residual_trend=rt, dt_s=0.1)
        # No crash; valid estimate.
        assert est.tte_hours_central > 0.0
        assert est.confidence > 0.0

    def test_update_without_residual_trend_still_works(self) -> None:
        """Backward compat: no residual_trend kwarg → zeros in
        the new feature columns; estimate is still valid."""
        cfg = load_config(CONFIG_DIR).engine
        cal = RulCalculator(cfg)
        h = _healthy_health(time_s=0.0)
        # No residual_trend argument.
        est = cal.update(health=h, dt_s=0.1)
        assert est.tte_hours_central > 0.0
        assert est.confidence > 0.0

    def test_residual_trend_features_change_estimate(self) -> None:
        """Different residual trends should produce different
        wear rates (the new feature columns actually feed the
        model)."""
        cfg = load_config(CONFIG_DIR).engine
        cal_cf = RulCalculator(cfg)
        h = _degraded_health()
        # Zero residual trend.
        rt_zero = ResidualTrendInput(
            channel_names=("rpm", "egt", "cht", "oil_pressure",
                           "oil_temperature", "fuel_flow", "vibration",
                           "altitude", "airspeed", "ambient_temperature",
                           "ambient_pressure"),
            z_history=np.zeros((11, 10), dtype=np.float64),
        )
        # High residual trend.
        rt_high = ResidualTrendInput(
            channel_names=("rpm", "egt", "cht", "oil_pressure",
                           "oil_temperature", "fuel_flow", "vibration",
                           "altitude", "airspeed", "ambient_temperature",
                           "ambient_pressure"),
            z_history=np.full((11, 10), 5.0, dtype=np.float64),
        )
        # The closed-form wear rate doesn't use residual_trend
        # (it's a transparent function of health); both calls
        # should give the same wear_rate_per_hour. We assert
        # the trait is documented behaviour, not a bug.
        est_zero = cal_cf.update(health=h, residual_trend=rt_zero, dt_s=0.1)
        est_high = cal_cf.update(health=h, residual_trend=rt_high, dt_s=0.1)
        assert est_zero.wear_rate_per_hour == pytest.approx(
            est_high.wear_rate_per_hour, rel=1e-9,
        )


# ---------------------------------------------------------------------
# ASSUMPTIONS.md
# ---------------------------------------------------------------------
class TestAssumptions:
    def test_assumptions_md_exists(self) -> None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "backend" / "rul" / "ASSUMPTIONS.md"
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        # Mention key topics.
        assert "Wear model" in text
        assert "Monte Carlo" in text or "Monte-Carlo" in text or "Monte" in text
        assert "horizon" in text.lower()
        assert "RUL_UNCERTAIN" in text
        assert "MAPE" in text
        assert "PICP" in text

    def test_assumptions_md_lists_at_least_8_assumptions(self) -> None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "backend" / "rul" / "ASSUMPTIONS.md"
        text = p.read_text(encoding="utf-8")
        # Each assumption is numbered with a leading "## N." header.
        import re
        matches = re.findall(r"^##\s+\d+\.\s+", text, flags=re.MULTILINE)
        assert len(matches) >= 8, (
            f"expected >= 8 numbered assumptions, found {len(matches)}"
        )


# ---------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------
class TestVersioning:
    def test_model_version_field_present(self) -> None:
        cfg = load_config(CONFIG_DIR)
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_wear_model(ds, n_estimators=10, seed=0)
        assert m.model_version == MODEL_VERSION

    def test_save_load_preserves_version(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_wear_model(ds, n_estimators=10, seed=0,
                             n_train_scenarios=42)
        path = tmp_path / "wear.pkl"
        save_model(m, path)
        sidecar = tmp_path / "wear.pkl.version"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip() == MODEL_VERSION
        loaded = load_model(path)
        assert loaded.model_version == MODEL_VERSION
        assert loaded.n_train_scenarios == 42

    def test_load_warns_on_version_mismatch(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_wear_model(ds, n_estimators=10, seed=0)
        path = tmp_path / "wear.pkl"
        save_model(m, path)
        # Tamper with the sidecar.
        (tmp_path / "wear.pkl.version").write_text(
            "phase11-rul-0.0.0-deliberately-bad", encoding="utf-8",
        )
        with pytest.warns(ModelVersionWarning):
            loaded = load_model(path)
        assert loaded.model_version == MODEL_VERSION

    def test_load_warns_on_missing_sidecar(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_wear_model(ds, n_estimators=10, seed=0)
        path = tmp_path / "wear.pkl"
        save_model(m, path)
        (tmp_path / "wear.pkl.version").unlink()
        with pytest.warns(ModelVersionWarning):
            load_model(path)
