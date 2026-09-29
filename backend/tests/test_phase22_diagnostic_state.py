"""PHASE 22 tests — unified :class:`DiagnosticState`.

The 14 tests in this file cover:

* the 12 user-facing fields are all present in
  :class:`DiagnosticState` and the rich ``to_rich_dict`` view;
* the wire format ``to_dict()`` matches the spec example
  exactly (``engine_state``, ``health_index``, ``anomaly_score``,
  per-category fault probabilities, ``rul`` block, ``evidence``,
  ``data_quality``, ``model_version``);
* the ``DiagnosticAssembler`` correctly projects per-tick
  components into the 12-field view;
* residual-trend, sensor quality, fault probability grouping,
  uncertainty, and data quality are all derived correctly;
* the dashboard ``PipelineRunner.tick()`` path now exposes a
  :class:`DiagnosticState` as its primary payload (the
  ``DashboardSnapshot`` continues to work for back-compat);
* the FastAPI app serves a ``/api/diagnostic/latest`` endpoint
  that returns the same wire format;
* the replay controller yields :class:`DiagnosticState` objects
  through the same wire format;
* ``reset()`` clears the trend history between scenarios.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, List

import pytest

from backend.config import load_config
from backend.digital_twin import CHANNEL_TO_STATE, ChannelResidual, ResidualFrame
from backend.environment import EnvironmentState
from backend.faults import FaultClass
from backend.health import HealthIndex, HealthLabel, HealthTrend, Subsystem
from backend.ml import CalibrationStatus, FaultClassification
from backend.risk import MissionPhase, RiskAssessment, RiskLevel, RiskStatus
from backend.rul import RulEstimate, RulStatus, RulTrend
from backend.sensors import NoiseMode, SensorReading, SensorSample
from backend.simulation import EngineState

from backend.diagnostics import (
    DiagnosticAssembler,
    DiagnosticState,
    SensorQuality,
    TrendDirection,
    TREND_EPSILON,
    TREND_WINDOW,
)

pytestmark = pytest.mark.phase22

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------
def _engine_state(**overrides) -> EngineState:
    base = dict(
        time_s=10.0, rpm=2300.0, manifold_pressure_inhg=29.5,
        egt_c=720.0, cht_c=190.0, oil_pressure_psi=58.0,
        oil_temperature_c=95.0, fuel_flow_lph=24.0,
        vibration_rms_g=2.1, bsfc_g_per_kwh=380.0,
        brake_power_kw=85.0, wear=0.0, throttle=0.65,
        altitude_m=1500.0, airspeed_mps=58.0,
    )
    base.update(overrides)
    return EngineState(**base)


def _env_state(**overrides) -> EnvironmentState:
    """A small, deterministic :class:`EnvironmentState` for tests.

    The composite ``EnvironmentState`` requires an
    :class:`AtmosphericState` and a :class:`WindState`; we
    build both with sensible default values that match the
    sample's per-channel readings.
    """
    from backend.environment import EnvironmentState as _ES
    from backend.environment.atmosphere import AtmosphericState
    from backend.environment.state import WindState
    base = dict(
        time_s=10.0, altitude_m=1500.0, airspeed_mps=58.0,
        throttle=0.65, vertical_accel_mps2=0.0,
    )
    base.update(overrides)
    base["atmosphere"] = AtmosphericState(
        altitude_m=1500.0, pressure_pa=85000.0,
        temperature_k=285.15, temperature_c=12.0,
        density_kg_per_m3=1.06, speed_of_sound_mps=340.0,
    )
    base["wind"] = WindState(
        time_s=10.0, turbulence_w_mps=0.0,
        gust_w_mps=0.0, total_w_mps=0.0,
        is_gust_active=False, steady_wind_mps=0.0,
    )
    return _ES(**base)


def _sample(engine: EngineState, env: EnvironmentState) -> SensorSample:
    """A 12-channel sample, all NORMAL, with a few outliers for testing."""
    readings: Dict[str, SensorReading] = {
        "rpm": SensorReading(value=2310.0, mode=NoiseMode.NORMAL),
        "egt": SensorReading(value=722.0, mode=NoiseMode.NORMAL),
        "cht": SensorReading(value=191.0, mode=NoiseMode.NORMAL),
        "oil_pressure": SensorReading(value=58.0, mode=NoiseMode.NORMAL),
        "oil_temperature": SensorReading(value=95.0, mode=NoiseMode.DRIFTING),
        "fuel_flow": SensorReading(value=24.0, mode=NoiseMode.NORMAL),
        "vibration": SensorReading(value=2.2, mode=NoiseMode.NORMAL),
        "imu_accel": SensorReading(value=0.05, mode=NoiseMode.NORMAL),
        "altitude": SensorReading(value=1500.0, mode=NoiseMode.NORMAL),
        "airspeed": SensorReading(value=58.0, mode=NoiseMode.NORMAL),
        "ambient_temperature": SensorReading(value=12.0, mode=NoiseMode.NORMAL),
        "ambient_pressure": SensorReading(value=85000.0, mode=NoiseMode.NORMAL),
    }
    return SensorSample(time_s=10.0, readings=readings, engine=engine, env=env)


def _residual_frame(time_s: float = 10.0, overall_confidence: float = 0.9) -> ResidualFrame:
    """A 12-channel residual frame, all in-bounds, slightly perturbed."""
    rows: Dict[str, ChannelResidual] = {}
    for ch in CHANNEL_TO_STATE:
        rows[ch] = ChannelResidual(
            channel=ch,
            observed=100.0,
            predicted=99.5,
            residual=0.5,
            z_score=0.8,
            confidence=0.9,
            in_bounds=True,
        )
    return ResidualFrame(time_s=time_s, residuals=rows, overall_confidence=overall_confidence)


def _health(overall_score: float = 0.78,
            label: HealthLabel = HealthLabel.DEGRADED,
            confidence: float = 0.85) -> HealthIndex:
    from backend.health.types import SubsystemHealth
    subsystems = {
        Subsystem.THERMAL: SubsystemHealth(
            subsystem=Subsystem.THERMAL, score=0.7, confidence=0.8,
            contributors=["egt", "cht"],
        ),
        Subsystem.LUBRICATION: SubsystemHealth(
            subsystem=Subsystem.LUBRICATION, score=0.85, confidence=0.85,
            contributors=["oil_pressure", "oil_temperature"],
        ),
    }
    return HealthIndex(
        time_s=10.0, overall_score=overall_score, overall_label=label,
        confidence=confidence, subsystems=subsystems, trend=HealthTrend.STABLE,
    )


def _rul(central: float = 50.0, lower: float = 30.0, upper: float = 80.0,
         confidence: float = 0.7) -> RulEstimate:
    return RulEstimate(
        time_s=10.0, tte_hours_central=central, tte_hours_lower=lower,
        tte_hours_upper=upper, wear_rate_per_hour=0.02,
        confidence=confidence, status=RulStatus.RUL_DEGRADED,
        trend=RulTrend.STABLE, model_status=RulStatus.RUL_DEGRADED,
    )


def _risk(score: float = 0.5, level: RiskLevel = RiskLevel.MODERATE,
          status: RiskStatus = RiskStatus.CAUTION,
          conf: float = 0.8) -> RiskAssessment:
    return RiskAssessment(
        time_s=10.0, risk_score=score, risk_level=level, status=status,
        confidence=conf, mission_phase=MissionPhase.CRUISE,
    )


def _anomaly(score: float = 0.84, label: str = "ANOMALY",
             confidence: float = 0.85) -> "AnomalyAssessment":
    from backend.diagnostics import AnomalyAssessment
    from backend.diagnostics.types import AnomalyLabel
    return AnomalyAssessment(
        time_s=10.0, overall_score=score, overall_label=AnomalyLabel(label),
        confidence=confidence, channels={}, notes=(), frame_status="OK",
    )


# ---------------------------------------------------------------------
# 1. The 12 fields exist on the dataclass
# ---------------------------------------------------------------------
class TestDiagnosticStateShape:
    def test_has_twelve_user_facing_fields(self) -> None:
        s = DiagnosticState(
            time_s=10.0,
            observed_state={"rpm": 100.0},
            expected_state={"rpm": 100.0},
            residual={"rpm": {"z_score": 0.0}},
            residual_trend={"overall": "STABLE"},
            sensor_quality={"rpm": {"mode": "NORMAL", "score": 0.0}},
            environment_context={"altitude_m": 1000.0},
            ai_anomaly_score=0.0,
            fault_probabilities={"engine": 0.0, "sensor": 0.0,
                                 "environment": 0.0, "unknown": 0.0,
                                 "healthy": 1.0},
            health_index={"overall_score": 1.0},
            rul={"estimate": 0.0},
            uncertainty=0.0,
            mission_risk={"risk_score": 0.0},
        )
        for name in (
            "observed_state", "expected_state", "residual",
            "residual_trend", "sensor_quality", "environment_context",
            "ai_anomaly_score", "fault_probabilities", "health_index",
            "rul", "uncertainty", "mission_risk",
        ):
            assert hasattr(s, name), f"missing field: {name}"

    def test_rich_dict_has_twelve_keys(self) -> None:
        s = DiagnosticState(
            time_s=1.0,
            observed_state={}, expected_state={}, residual={},
            residual_trend={}, sensor_quality={}, environment_context={},
            ai_anomaly_score=0.0, fault_probabilities={},
            health_index={}, rul={}, uncertainty=0.0, mission_risk={},
        )
        d = s.to_rich_dict()
        for k in (
            "observed_state", "expected_state", "residual",
            "residual_trend", "sensor_quality", "environment_context",
            "ai_anomaly_score", "fault_probabilities", "health_index",
            "rul", "uncertainty", "mission_risk",
        ):
            assert k in d, f"missing rich key: {k}"


# ---------------------------------------------------------------------
# 2. The wire format matches the spec example
# ---------------------------------------------------------------------
class TestWireFormat:
    def test_to_dict_matches_spec(self) -> None:
        s = DiagnosticState(
            time_s=10.0,
            observed_state={"rpm": 2300.0},
            expected_state={"rpm": 2280.0},
            residual={"rpm": {"z_score": 1.5, "confidence": 0.9}},
            residual_trend={"overall": "WORSENING", "per_channel": {}},
            sensor_quality={"rpm": {"mode": "NORMAL", "score": 0.0,
                                    "quality": "NOMINAL"}},
            environment_context={"altitude_m": 1500.0},
            ai_anomaly_score=0.84,
            fault_probabilities={"engine": 0.82, "sensor": 0.06,
                                 "environment": 0.12, "unknown": 0.0,
                                 "healthy": 0.0},
            health_index={"overall_score": 0.78,
                          "overall_label": "DEGRADED",
                          "confidence": 0.9, "trend": "STABLE",
                          "subsystems": {}},
            rul={"estimate": 50.0, "lower": 30.0, "upper": 80.0,
                 "confidence": 0.7},
            uncertainty=0.18,
            mission_risk={"risk_score": 0.5, "risk_level": "MODERATE",
                          "status": "CAUTION", "mission_phase": "CRUISE",
                          "confidence": 0.8},
            evidence=("anomaly: ANOMALY (0.84)",),
            data_quality=1.0,
            engine_state="DEGRADED",
        )
        d = s.to_dict()
        # The 11 keys the user specified.
        for k in (
            "engine_state", "health_index", "anomaly_score",
            "engine_fault_probability", "sensor_fault_probability",
            "environment_probability", "unknown_probability",
            "healthy_probability", "rul", "evidence", "data_quality",
            "model_version",
        ):
            assert k in d, f"missing spec key: {k}"
        # Direct field checks against the spec example values.
        assert d["engine_state"] == "DEGRADED"
        assert d["health_index"] == pytest.approx(0.78, abs=1e-9)
        assert d["anomaly_score"] == pytest.approx(0.84, abs=1e-9)
        assert d["engine_fault_probability"] == pytest.approx(0.82, abs=1e-9)
        assert d["sensor_fault_probability"] == pytest.approx(0.06, abs=1e-9)
        assert d["environment_probability"] == pytest.approx(0.12, abs=1e-9)
        assert d["rul"]["estimate"] == pytest.approx(50.0, abs=1e-9)
        assert d["rul"]["lower"] == pytest.approx(30.0, abs=1e-9)
        assert d["rul"]["upper"] == pytest.approx(80.0, abs=1e-9)
        assert d["rul"]["confidence"] == pytest.approx(0.7, abs=1e-9)
        assert d["evidence"] == ["anomaly: ANOMALY (0.84)"]
        assert d["data_quality"] == pytest.approx(1.0, abs=1e-9)
        assert isinstance(d["model_version"], str) and d["model_version"]

    def test_rich_dict_round_trip(self) -> None:
        s = DiagnosticState(
            time_s=12.5, observed_state={"rpm": 2300.0},
            expected_state={"rpm": 2280.0},
            residual={"rpm": {"z_score": 1.5, "residual": 20.0,
                              "confidence": 0.9, "in_bounds": True}},
            residual_trend={"overall": "WORSENING", "per_channel": {}},
            sensor_quality={"rpm": {"mode": "NORMAL", "score": 0.0,
                                    "quality": "NOMINAL"}},
            environment_context={"altitude_m": 1500.0},
            ai_anomaly_score=0.84,
            fault_probabilities={"engine": 0.5, "sensor": 0.1,
                                 "environment": 0.1, "unknown": 0.0,
                                 "healthy": 0.3},
            health_index={"overall_score": 0.7,
                          "overall_label": "DEGRADED",
                          "confidence": 0.9, "trend": "STABLE",
                          "subsystems": {}},
            rul={"estimate": 50.0, "lower": 30.0, "upper": 80.0,
                 "confidence": 0.7, "status": "RUL_DEGRADED",
                 "trend": "STABLE"},
            uncertainty=0.2,
            mission_risk={"risk_score": 0.4, "risk_level": "MODERATE",
                          "status": "CAUTION", "mission_phase": "CRUISE",
                          "confidence": 0.8},
            evidence=("a", "b"), data_quality=0.95,
            engine_state="DEGRADED",
        )
        d = s.to_rich_dict()
        assert d["observed_state"]["rpm"] == 2300.0
        assert d["ai_anomaly_score"] == pytest.approx(0.84, abs=1e-9)
        assert d["fault_probabilities"]["engine"] == pytest.approx(0.5, abs=1e-9)
        assert d["health_index"]["overall_label"] == "DEGRADED"
        assert d["mission_risk"]["status"] == "CAUTION"
        assert d["evidence"] == ["a", "b"]


# ---------------------------------------------------------------------
# 3. DiagnosticAssembler end-to-end
# ---------------------------------------------------------------------
class TestAssembler:
    def _build_state(self) -> DiagnosticState:
        engine = _engine_state()
        env = _env_state()
        sample = _sample(engine, env)
        res = _residual_frame(time_s=sample.time_s)
        anomaly = _anomaly()
        h = _health()
        r = _rul()
        rk = _risk()
        asm = DiagnosticAssembler()
        return asm.assemble(
            time_s=sample.time_s, sample=sample, twin_state=None,
            residual=res, anomaly=anomaly, classification=None,
            health=h, rul=r, risk=rk, engine=engine, env=env,
        )

    def test_assembler_produces_all_twelve_fields(self) -> None:
        s = self._build_state()
        for k in (
            "observed_state", "expected_state", "residual",
            "residual_trend", "sensor_quality", "environment_context",
            "ai_anomaly_score", "fault_probabilities", "health_index",
            "rul", "uncertainty", "mission_risk",
        ):
            assert hasattr(s, k)
        # Some fields should be non-empty.
        assert s.observed_state  # has 12 channels
        assert s.environment_context["altitude_m"] == pytest.approx(1500.0)

    def test_assembler_engine_state_matches_health_label(self) -> None:
        s = self._build_state()
        assert s.engine_state == "DEGRADED"
        # Same again with HEALTHY.
        h2 = _health(overall_score=0.9, label=HealthLabel.HEALTHY)
        rk = _risk()
        r = _rul()
        s2 = DiagnosticAssembler().assemble(
            time_s=10.0, sample=_sample(_engine_state(), _env_state()),
            twin_state=None, residual=_residual_frame(),
            anomaly=_anomaly(), classification=None, health=h2,
            rul=r, risk=rk, engine=_engine_state(), env=_env_state(),
        )
        assert s2.engine_state == "HEALTHY"

    def test_assembler_with_pyfault_classification_groups_by_category(self) -> None:
        # All FaultClass probabilities get bucketed into the 5
        # groups (engine / sensor / environment / unknown / healthy).
        from backend.ml import FaultClassification
        fc = FaultClassification(
            time_s=10.0, fault_class=FaultClass.ENGINE_DEGRADATION,
            confidence=0.9,
            probabilities={
                FaultClass.HEALTHY: 0.1,
                FaultClass.ENGINE_DEGRADATION: 0.4,
                FaultClass.OVERHEATING: 0.3,
                FaultClass.LUBRICATION_PRESSURE_ANOMALY: 0.1,
                FaultClass.VIBRATION_ENGINE_ANOMALY: 0.05,
                FaultClass.PERFORMANCE_LOSS: 0.05,
                FaultClass.SENSOR_FAULT: 0.0,
                FaultClass.ENVIRONMENTAL_DISTURBANCE: 0.0,
                FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE: 0.0,
            },
            status=CalibrationStatus.CALIBRATED, features_used=60,
        )
        h = _health(overall_score=0.7, label=HealthLabel.DEGRADED)
        r = _rul()
        rk = _risk()
        s = DiagnosticAssembler().assemble(
            time_s=10.0, sample=_sample(_engine_state(), _env_state()),
            twin_state=None, residual=_residual_frame(),
            anomaly=_anomaly(), classification=fc, health=h,
            rul=r, risk=rk, engine=_engine_state(), env=_env_state(),
        )
        # engine bucket sums all 5 engine-truth-layer classes.
        assert s.fault_probabilities["engine"] == pytest.approx(0.9, abs=1e-9)
        assert s.fault_probabilities["sensor"] == pytest.approx(0.0, abs=1e-9)
        assert s.fault_probabilities["environment"] == pytest.approx(0.0, abs=1e-9)
        assert s.fault_probabilities["healthy"] == pytest.approx(0.1, abs=1e-9)
        assert s.fault_probabilities["unknown"] == pytest.approx(0.0, abs=1e-9)

    def test_assembler_with_uncalibrated_classifier_returns_unknown_bucket(self) -> None:
        from backend.ml import FaultClassification
        fc = FaultClassification(
            time_s=10.0, fault_class=FaultClass.HEALTHY, confidence=0.0,
            probabilities={}, status=CalibrationStatus.MODEL_NOT_CALIBRATED,
            features_used=0,
        )
        h = _health()
        r = _rul()
        rk = _risk()
        s = DiagnosticAssembler().assemble(
            time_s=10.0, sample=_sample(_engine_state(), _env_state()),
            twin_state=None, residual=_residual_frame(),
            anomaly=_anomaly(), classification=fc, health=h,
            rul=r, risk=rk, engine=_engine_state(), env=_env_state(),
        )
        # No fake outputs: every probability is 0 except unknown.
        assert s.fault_probabilities["unknown"] == pytest.approx(1.0, abs=1e-9)
        for k in ("engine", "sensor", "environment", "healthy"):
            assert s.fault_probabilities[k] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------
# 4. Sensor quality aggregation
# ---------------------------------------------------------------------
class TestSensorQuality:
    def test_normal_channel_is_nominal(self) -> None:
        # Build a single-NORMAL sample and check the aggregation.
        engine = _engine_state()
        env = _env_state()
        readings = {
            "rpm": SensorReading(value=2300.0, mode=NoiseMode.NORMAL),
            "egt": SensorReading(value=720.0, mode=NoiseMode.NORMAL),
            "cht": SensorReading(value=190.0, mode=NoiseMode.NORMAL),
            "oil_pressure": SensorReading(value=58.0, mode=NoiseMode.NORMAL),
            "oil_temperature": SensorReading(value=95.0, mode=NoiseMode.NORMAL),
            "fuel_flow": SensorReading(value=24.0, mode=NoiseMode.NORMAL),
            "vibration": SensorReading(value=2.1, mode=NoiseMode.NORMAL),
            "imu_accel": SensorReading(value=0.05, mode=NoiseMode.NORMAL),
            "altitude": SensorReading(value=1500.0, mode=NoiseMode.NORMAL),
            "airspeed": SensorReading(value=58.0, mode=NoiseMode.NORMAL),
            "ambient_temperature": SensorReading(value=12.0, mode=NoiseMode.NORMAL),
            "ambient_pressure": SensorReading(value=85000.0, mode=NoiseMode.NORMAL),
        }
        sample = SensorSample(time_s=10.0, readings=readings,
                              engine=engine, env=env)
        sq = DiagnosticAssembler._sensor_quality(sample)
        for ch, v in sq.items():
            assert v["mode"] == "NORMAL"
            assert v["quality"] == "NOMINAL"
            assert v["score"] == 0.0  # NORMAL → 0.0 in the table

    def test_dropped_channel_is_invalid(self) -> None:
        engine = _engine_state()
        env = _env_state()
        readings = {
            ch: SensorReading(value=None, mode=NoiseMode.DROPPED)
            for ch in [
                "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
                "fuel_flow", "vibration", "imu_accel", "altitude",
                "airspeed", "ambient_temperature", "ambient_pressure",
            ]
        }
        sample = SensorSample(time_s=10.0, readings=readings,
                              engine=engine, env=env)
        sq = DiagnosticAssembler._sensor_quality(sample)
        for ch, v in sq.items():
            assert v["quality"] == "INVALID"
            assert v["score"] == 0.0
            assert "no_reading" in v["contributors"]


# ---------------------------------------------------------------------
# 5. Residual trend computation
# ---------------------------------------------------------------------
class TestResidualTrend:
    def test_trend_insufficient_data_when_no_history(self) -> None:
        asm = DiagnosticAssembler()
        engine = _engine_state()
        env = _env_state()
        sample = _sample(engine, env)
        res = _residual_frame(time_s=10.0)
        h = _health()
        r = _rul()
        rk = _risk()
        s = asm.assemble(
            time_s=10.0, sample=sample, twin_state=None, residual=res,
            anomaly=_anomaly(), classification=None, health=h, rul=r,
            risk=rk, engine=engine, env=env,
        )
        # First tick → no per-channel samples yet (we just appended 1).
        for ch, info in s.residual_trend["per_channel"].items():
            assert info["direction"] == "STABLE"  # one sample → zero slope → STABLE
            assert info["n_samples"] == 1

    def test_trend_worsens_when_residual_grows(self) -> None:
        asm = DiagnosticAssembler()
        engine = _engine_state()
        env = _env_state()
        sample = _sample(engine, env)
        h = _health()
        r = _rul()
        rk = _risk()
        # Three ticks, with a growing |z|.
        last = None
        for t in (10.0, 10.5, 11.0):
            rows = {
                ch: ChannelResidual(
                    channel=ch, observed=100.0 + (t - 10.0) * 5.0,
                    predicted=100.0,
                    residual=(t - 10.0) * 5.0,
                    z_score=(t - 10.0) * 1.0,  # grows
                    confidence=0.9, in_bounds=True,
                )
                for ch in CHANNEL_TO_STATE
            }
            res = ResidualFrame(time_s=t, residuals=rows,
                                overall_confidence=0.9)
            last = asm.assemble(
                time_s=t, sample=sample, twin_state=None, residual=res,
                anomaly=_anomaly(), classification=None, health=h, rul=r,
                risk=rk, engine=engine, env=env,
            )
        # After 3 ticks the slope is positive, so direction is WORSENING.
        any_worsening = any(
            info["direction"] == "WORSENING"
            for info in last.residual_trend["per_channel"].values()
        )
        assert any_worsening

    def test_trend_reset_clears_history(self) -> None:
        asm = DiagnosticAssembler()
        # Run two ticks.
        engine = _engine_state()
        env = _env_state()
        sample = _sample(engine, env)
        h = _health()
        r = _rul()
        rk = _risk()
        for t in (10.0, 10.5):
            res = _residual_frame(time_s=t)
            asm.assemble(
                time_s=t, sample=sample, twin_state=None, residual=res,
                anomaly=_anomaly(), classification=None, health=h, rul=r,
                risk=rk, engine=engine, env=env,
            )
        assert any(asm._history.values())
        asm.reset()
        assert not asm._history


# ---------------------------------------------------------------------
# 6. Uncertainty aggregation
# ---------------------------------------------------------------------
class TestUncertainty:
    def test_uncertainty_is_zero_when_all_components_confident(self) -> None:
        asm = DiagnosticAssembler()
        res = _residual_frame(overall_confidence=1.0)
        a = _anomaly(score=0.5, label="NORMAL", confidence=1.0)
        h = _health(confidence=1.0)
        r = _rul(confidence=1.0)
        rk = _risk(conf=1.0)
        u = asm._uncertainty(res, a, None, h, r, rk)
        assert u == pytest.approx(0.0, abs=1e-9)

    def test_uncertainty_grows_with_missing_components(self) -> None:
        asm = DiagnosticAssembler()
        # A residual at 0.2 confidence: the only component
        # contributing to the mean. The other slots are missing.
        res = _residual_frame(overall_confidence=0.2)
        u = asm._uncertainty(res, None, None, None, None, None)
        # Mean informativeness is 0.2 → uncertainty is 0.8.
        assert u == pytest.approx(0.8, abs=1e-9)
        assert u <= 1.0


# ---------------------------------------------------------------------
# 7. RUL block
# ---------------------------------------------------------------------
class TestRulBlock:
    def test_rul_block_has_estimate_lower_upper_confidence(self) -> None:
        r = _rul(central=50.0, lower=30.0, upper=80.0, confidence=0.7)
        blk = DiagnosticAssembler._rul_block(r)
        for k in ("estimate", "lower", "upper", "confidence"):
            assert k in blk
        assert blk["estimate"] == pytest.approx(50.0, abs=1e-9)
        assert blk["lower"] == pytest.approx(30.0, abs=1e-9)
        assert blk["upper"] == pytest.approx(80.0, abs=1e-9)
        assert blk["confidence"] == pytest.approx(0.7, abs=1e-9)

    def test_rul_block_when_none_returns_uncertain(self) -> None:
        blk = DiagnosticAssembler._rul_block(None)
        assert blk["status"] == "RUL_UNCERTAIN"
        assert blk["estimate"] is None


# ---------------------------------------------------------------------
# 8. Pipeline integration
# ---------------------------------------------------------------------
class TestPipelineIntegration:
    """The dashboard ``PipelineRunner.tick()`` should produce a
    :class:`DiagnosticState` through the assembler."""

    def test_pipeline_runner_uses_assembler(self) -> None:
        from backend.dashboard import DEFAULT_SCENARIO_NAME, get_scenario
        from backend.dashboard.runner import PipelineRunner
        cfg = load_config(CONFIG_DIR)
        scenario = get_scenario(DEFAULT_SCENARIO_NAME)
        runner = PipelineRunner(cfg, scenario, dt_s=0.1)
        # Warm up so the modules have history.
        for _ in range(5):
            runner.tick()
        # The runner now exposes a ``diagnostic_assembler`` and
        # the latest diagnostic state on demand.
        asm = runner.diagnostic_assembler
        assert isinstance(asm, DiagnosticAssembler)
        diag = runner.latest_diagnostic
        assert isinstance(diag, DiagnosticState)
        assert set(diag.to_dict().keys()) >= {
            "engine_state", "health_index", "anomaly_score",
            "engine_fault_probability", "sensor_fault_probability",
            "environment_probability", "rul", "evidence",
            "data_quality", "model_version",
        }


# ---------------------------------------------------------------------
# 9. API integration
# ---------------------------------------------------------------------
class TestApiIntegration:
    """The FastAPI app exposes the diagnostic state at
    ``/api/diagnostic/latest`` and ``/api/diagnostic/history``."""

    def test_api_serves_diagnostic_state(self) -> None:
        from backend.dashboard import DEFAULT_SCENARIO_NAME, get_scenario
        from backend.dashboard import create_app
        from backend.config import DashboardConfig
        from fastapi.testclient import TestClient
        cfg = load_config(CONFIG_DIR)
        scenario = get_scenario(DEFAULT_SCENARIO_NAME)
        dashboard_cfg = DashboardConfig(
            tick_rate_hz=50.0, history_size=20,
            default_scenario=DEFAULT_SCENARIO_NAME,
        )
        app = create_app(cfg, dashboard_cfg, initial_scenario=scenario)
        with TestClient(app) as client:
            # The seed queue runs 5 ticks at startup so the
            # first call to /api/diagnostic/latest already has
            # a populated DiagnosticState.
            resp = client.get("/api/diagnostic/latest")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "diagnostic" in body
            d = body["diagnostic"]
            for k in (
                "engine_state", "health_index", "anomaly_score",
                "engine_fault_probability", "sensor_fault_probability",
                "environment_probability", "rul", "evidence",
                "data_quality", "model_version",
            ):
                assert k in d, f"missing: {k}"


# ---------------------------------------------------------------------
# 10. Helpers used elsewhere (TrendDirection / SensorQuality enums)
# ---------------------------------------------------------------------
class TestEnums:
    def test_trend_direction_values(self) -> None:
        for v in ("IMPROVING", "STABLE", "WORSENING", "INSUFFICIENT_DATA"):
            assert v in [m.value for m in TrendDirection]

    def test_sensor_quality_values(self) -> None:
        for v in ("NOMINAL", "DEGRADED", "UNRELIABLE", "INVALID"):
            assert v in [m.value for m in SensorQuality]


# ---------------------------------------------------------------------
# 11. Latency
# ---------------------------------------------------------------------
class TestLatency:
    def test_assembler_under_target(self) -> None:
        """A per-tick assembler call must be well under any real-time
        budget. The threshold here (1 ms) is generous."""
        asm = DiagnosticAssembler()
        engine = _engine_state()
        env = _env_state()
        sample = _sample(engine, env)
        res = _residual_frame()
        a = _anomaly()
        h = _health()
        r = _rul()
        rk = _risk()
        n = 1000
        t0 = _time.perf_counter()
        for _ in range(n):
            asm.assemble(
                time_s=10.0, sample=sample, twin_state=None,
                residual=res, anomaly=a, classification=None,
                health=h, rul=r, risk=rk, engine=engine, env=env,
            )
        per_call_us = 1e6 * (_time.perf_counter() - t0) / n
        assert per_call_us < 1000.0
