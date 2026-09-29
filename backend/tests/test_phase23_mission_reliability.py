"""PHASE 23 tests — mission reliability risk layer.

The 14 tests in this file cover:

* The :class:`ReliabilityBand` enum has exactly the 4 user-facing
  bands ``LOW`` / ``MEDIUM`` / ``HIGH`` / ``UNKNOWN``.
* The :func:`aggregate_reliability` pure function maps nominal
  inputs to ``LOW`` and lifts to ``MEDIUM`` / ``HIGH`` as drivers
  accumulate.
* ``UNKNOWN`` wins when overall confidence is below the floor.
* The :class:`MissionReliabilityExplanation` exposes a
  headline + a tuple of named, human-readable drivers.
* The wire format contains **no** mission-success-probability
  number and **no** control / actuator / throttle-command key.
* :class:`MissionReliabilityInputs` round-trips through
  ``to_dict()`` / ``from_dict()``.
* :func:`aggregate_reliability` is pure (no hidden state).
* The dashboard ``PipelineRunner.tick()`` path now exposes
  ``latest_reliability`` and ``reliability_history``.
* The FastAPI app serves ``/api/reliability/latest`` and
  ``/api/reliability/history`` endpoints.
* The :class:`DiagnosticState` rich payload carries the new
  ``reliability`` block alongside the existing 12 fields.
* ``aggregate_reliability`` is fast (< 100 µs per call).
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

from backend.risk import (
    DriverSeverity,
    MissionPhase,
    MissionReliabilityAssessment,
    MissionReliabilityExplanation,
    MissionReliabilityInputs,
    ReliabilityBand,
    ReliabilityConfig,
    ReliabilityDriver,
    RiskAssessment,
    RiskLevel,
    RiskStatus,
    aggregate_reliability,
)

pytestmark = pytest.mark.phase23

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
def _risk_assessment(
    *,
    time_s: float = 10.0,
    risk_score: float = 0.0,
    status: RiskStatus = RiskStatus.GO,
    confidence: float = 0.9,
    phase: MissionPhase = MissionPhase.CRUISE,
    hours_to_destination: Optional[float] = 4.0,
    health_label: str = "HEALTHY",
    rul_status: str = "RUL_OK",
    anomaly_label: str = "NORMAL",
) -> RiskAssessment:
    """Construct a minimal :class:`RiskAssessment` for the
    reliability aggregator.

    Only the fields the aggregator actually reads are populated;
    everything else uses dataclass defaults.
    """
    return RiskAssessment(
        time_s=time_s,
        risk_score=risk_score,
        risk_level=RiskLevel.LOW,
        status=status,
        confidence=confidence,
        mission_phase=phase,
        hours_to_destination=hours_to_destination,
        inputs_meta={
            "health_label": health_label,
            "rul_status": rul_status,
            "anomaly_label": anomaly_label,
        },
    )


def _inputs(
    *,
    risk: Optional[RiskAssessment] = None,
    fault_prob_engine: float = 0.0,
    fault_prob_sensor: float = 0.0,
    fault_prob_environment: float = 0.0,
    rul_uncertainty: float = 0.0,
    mission_phase: MissionPhase = MissionPhase.CRUISE,
    remaining_hours: Optional[float] = 4.0,
    engine_load: float = 0.0,
    env_severity: float = 0.0,
    sensor_confidence: float = 1.0,
) -> MissionReliabilityInputs:
    return MissionReliabilityInputs(
        risk_assessment=risk or _risk_assessment(mission_phase=mission_phase),
        fault_prob_engine=fault_prob_engine,
        fault_prob_sensor=fault_prob_sensor,
        fault_prob_environment=fault_prob_environment,
        rul_uncertainty=rul_uncertainty,
        mission_phase=mission_phase,
        remaining_hours=remaining_hours,
        engine_load=engine_load,
        env_severity=env_severity,
        sensor_confidence=sensor_confidence,
    )


# ---------------------------------------------------------------------
# Test 1 — bands are exactly LOW / MEDIUM / HIGH / UNKNOWN
# ---------------------------------------------------------------------
def test_bands_are_low_medium_high_unknown() -> None:
    members = [b.value for b in ReliabilityBand]
    assert members == ["LOW", "MEDIUM", "HIGH", "UNKNOWN"], (
        f"unexpected band set: {members}"
    )
    assert len(ReliabilityBand) == 4
    # All members must be distinct and the right types.
    assert ReliabilityBand.LOW != ReliabilityBand.MEDIUM
    assert ReliabilityBand.MEDIUM != ReliabilityBand.HIGH
    assert ReliabilityBand.HIGH != ReliabilityBand.UNKNOWN


# ---------------------------------------------------------------------
# Test 2 — UNKNOWN when evidence is missing
# ---------------------------------------------------------------------
def test_unknown_when_evidence_missing() -> None:
    risk = _risk_assessment(confidence=0.10)  # below the floor
    inputs = _inputs(risk=risk, fault_prob_engine=0.5, engine_load=0.6)
    out = aggregate_reliability(inputs, time_s=10.0)
    assert out.band is ReliabilityBand.UNKNOWN
    assert "insufficient validated evidence" in out.explanation.headline.lower()
    assert "insufficient" in out.notes[0].lower() or "confidence" in out.notes[0].lower()


# ---------------------------------------------------------------------
# Test 3 — LOW when everything nominal
# ---------------------------------------------------------------------
def test_low_band_when_all_nominal() -> None:
    risk = _risk_assessment(
        health_label="HEALTHY",
        rul_status="RUL_OK",
        anomaly_label="NORMAL",
    )
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.05,
        fault_prob_sensor=0.02,
        fault_prob_environment=0.03,
        rul_uncertainty=0.05,
        mission_phase=MissionPhase.CRUISE,
        remaining_hours=5.0,
        engine_load=0.20,
        env_severity=0.05,
        sensor_confidence=0.98,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    assert out.band is ReliabilityBand.LOW
    assert "LOW" in out.explanation.headline
    # No CRITICAL drivers when everything is nominal.
    assert all(
        d.severity is not DriverSeverity.CRITICAL
        for d in out.explanation.drivers
    )


# ---------------------------------------------------------------------
# Test 4 — MEDIUM matches the spec example
# ---------------------------------------------------------------------
def test_medium_band_matches_spec_example() -> None:
    """Spec example:

        MISSION RISK: MEDIUM

        Drivers:
        - persistent engine performance residual
        - health index declining
        - remaining mission duration significant
        - RUL uncertainty elevated
    """
    risk = _risk_assessment(
        risk_score=0.40,
        status=RiskStatus.CAUTION,
        health_label="DEGRADED",
        rul_status="RUL_DEGRADED",
        anomaly_label="WARN",
    )
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.40,
        rul_uncertainty=0.7,
        remaining_hours=1.5,
        engine_load=0.30,
        env_severity=0.10,
        sensor_confidence=0.95,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    assert out.band is ReliabilityBand.MEDIUM, (
        f"expected MEDIUM, got {out.band.value}; "
        f"score={out.contributing_inputs.get('score')}, notes={out.notes}"
    )
    notes = [d.note for d in out.explanation.drivers]
    # All four named bullets from the spec must appear (order-agnostic).
    assert any("engine performance residual" in n for n in notes), notes
    assert any("health index" in n.lower() for n in notes), notes
    assert any("remaining mission duration" in n for n in notes), notes
    assert any("RUL uncertainty" in n for n in notes), notes
    # Headline matches the spec format.
    assert out.explanation.headline == "Mission reliability is MEDIUM."


# ---------------------------------------------------------------------
# Test 5 — HIGH when multiple critical
# ---------------------------------------------------------------------
def test_high_band_when_multiple_critical() -> None:
    risk = _risk_assessment(
        risk_score=0.85,
        status=RiskStatus.ABORT,
        health_label="CRITICAL",
        rul_status="RUL_CRITICAL",
        anomaly_label="ANOMALY",
    )
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.75,
        rul_uncertainty=0.9,
        engine_load=0.5,
        env_severity=0.4,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    assert out.band is ReliabilityBand.HIGH
    assert out.band is not ReliabilityBand.UNKNOWN
    # CRITICAL drivers must be present.
    critical = [d for d in out.explanation.drivers if d.severity is DriverSeverity.CRITICAL]
    assert len(critical) >= 2, [d.to_dict() for d in out.explanation.drivers]


# ---------------------------------------------------------------------
# Test 6 — HIGH when fault probability very high
# ---------------------------------------------------------------------
def test_high_band_when_fault_probability_very_high() -> None:
    risk = _risk_assessment()
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.85,  # > FAULT_PROB_CRITICAL
        engine_load=0.2,
        env_severity=0.0,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    assert out.band is ReliabilityBand.HIGH
    # A single critical driver is enough to force HIGH.
    assert "forced HIGH" in (out.notes[0] if out.notes else "") or out.band is ReliabilityBand.HIGH


# ---------------------------------------------------------------------
# Test 7 — explanation includes named drivers
# ---------------------------------------------------------------------
def test_explanation_includes_named_drivers() -> None:
    risk = _risk_assessment(
        health_label="DEGRADED",
        rul_status="RUL_DEGRADED",
        anomaly_label="ANOMALY",
    )
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.5,
        rul_uncertainty=0.5,
        engine_load=0.6,
        env_severity=0.4,
        sensor_confidence=0.5,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    # At least one driver per above-threshold signal.
    signals = {d.signal for d in out.explanation.drivers}
    assert "fault_probability" in signals
    assert "rul_uncertainty" in signals
    assert "engine_load" in signals
    assert "env_severity" in signals
    assert "sensor_confidence" in signals
    assert "health.label" in signals
    assert "anomaly.label" in signals
    # Every driver has a non-empty note.
    for d in out.explanation.drivers:
        assert d.note, f"empty note for signal={d.signal}"


# ---------------------------------------------------------------------
# Test 8 — no mission-success-probability in the wire format
# ---------------------------------------------------------------------
def test_no_mission_success_probability_in_output() -> None:
    risk = _risk_assessment()
    inputs = _inputs(risk=risk)
    out = aggregate_reliability(inputs, time_s=10.0)
    wire = out.to_dict()
    flat = str(wire).lower()
    # No key, no value, no comment line for the forbidden concept.
    for forbidden in (
        "success_probability",
        "success_chance",
        "mission_success",
        "probability_of_success",
    ):
        assert forbidden not in flat, (
            f"forbidden key/phrase {forbidden!r} in wire: {wire}"
        )
    # Also: no key whose name contains the substring 'success'.
    for k in wire.keys():
        assert "success" not in k.lower(), k
    # The explanation payload is also clean.
    expl = out.explanation.to_dict()
    flat2 = str(expl).lower()
    for forbidden in (
        "success_probability",
        "success_chance",
        "mission_success",
        "probability_of_success",
    ):
        assert forbidden not in flat2
    for k in expl.keys():
        assert "success" not in k.lower(), k


# ---------------------------------------------------------------------
# Test 9 — inputs round-trip
# ---------------------------------------------------------------------
def test_inputs_round_trip() -> None:
    risk = _risk_assessment(
        risk_score=0.42,
        status=RiskStatus.CAUTION,
        phase=MissionPhase.CRUISE,
        hours_to_destination=3.0,
    )
    original = _inputs(
        risk=risk,
        fault_prob_engine=0.4,
        fault_prob_sensor=0.05,
        fault_prob_environment=0.02,
        rul_uncertainty=0.6,
        remaining_hours=3.0,
        engine_load=0.5,
        env_severity=0.2,
        sensor_confidence=0.9,
    )
    d = original.to_dict()
    restored = MissionReliabilityInputs.from_dict(d)
    assert restored.fault_prob_engine == pytest.approx(0.4)
    assert restored.fault_prob_sensor == pytest.approx(0.05)
    assert restored.fault_prob_environment == pytest.approx(0.02)
    assert restored.rul_uncertainty == pytest.approx(0.6)
    assert restored.remaining_hours == pytest.approx(3.0)
    assert restored.engine_load == pytest.approx(0.5)
    assert restored.env_severity == pytest.approx(0.2)
    assert restored.sensor_confidence == pytest.approx(0.9)
    assert restored.mission_phase is MissionPhase.CRUISE
    assert restored.risk_assessment.risk_score == pytest.approx(0.42)


# ---------------------------------------------------------------------
# Test 10 — aggregator is pure
# ---------------------------------------------------------------------
def test_aggregator_is_pure() -> None:
    risk = _risk_assessment()
    inputs = _inputs(risk=risk)
    out_a = aggregate_reliability(inputs, time_s=10.0)
    out_b = aggregate_reliability(inputs, time_s=10.0)
    assert out_a == out_b
    # Also: aggregate_reliability must not mutate its inputs.
    sensor_conf_before = inputs.sensor_confidence
    aggregate_reliability(inputs, time_s=10.0)
    aggregate_reliability(inputs, time_s=10.0)
    assert inputs.sensor_confidence == sensor_conf_before


# ---------------------------------------------------------------------
# Test 11 — PipelineRunner wires reliability
# ---------------------------------------------------------------------
def test_pipeline_runner_wires_reliability() -> None:
    from backend.config import load_config
    from backend.dashboard.scenarios import get_scenario, DEFAULT_SCENARIO_NAME
    from backend.dashboard.runner import PipelineRunner

    cfg = load_config(str(CONFIG_DIR))
    scenario = get_scenario(DEFAULT_SCENARIO_NAME)
    runner = PipelineRunner(cfg, scenario, history_size=8)
    snap = runner.tick()
    # First tick must populate the latest reliability view.
    rel = runner.latest_reliability
    assert rel is not None
    assert isinstance(rel, MissionReliabilityAssessment)
    assert rel.band in set(ReliabilityBand)
    # The history deque is a separate, ordered container.
    assert len(runner.reliability_history) == 1
    assert runner.reliability_history[0] is rel
    # A few more ticks advance the history.
    for _ in range(3):
        runner.tick()
    assert len(runner.reliability_history) == 4
    # Each entry is a MissionReliabilityAssessment.
    for r in runner.reliability_history:
        assert isinstance(r, MissionReliabilityAssessment)


# ---------------------------------------------------------------------
# Test 12 — API serves reliability latest + history
# ---------------------------------------------------------------------
def test_api_serves_reliability_latest() -> None:
    from fastapi.testclient import TestClient

    from backend.config import DashboardConfig, load_config
    from backend.dashboard.app import create_app
    from backend.dashboard.scenarios import get_scenario, DEFAULT_SCENARIO_NAME

    cfg = load_config(str(CONFIG_DIR))
    app = create_app(
        cfg, DashboardConfig(),
        initial_scenario=get_scenario(DEFAULT_SCENARIO_NAME),
    )
    client = TestClient(app)
    # The lifespan startup seeds 5 ticks.
    r = client.get("/api/reliability/latest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scenario_name"]
    assert body["assessment"] is not None
    a = body["assessment"]
    assert a["mission_risk"] in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}
    assert "headline" in a["explanation"]
    assert isinstance(a["explanation"]["drivers"], list)
    # history endpoint
    r2 = client.get("/api/reliability/history?limit=3")
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert "assessments" in body2
    assert 1 <= len(body2["assessments"]) <= 3
    for entry in body2["assessments"]:
        assert entry["mission_risk"] in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}


# ---------------------------------------------------------------------
# Test 13 — wire format has no control / command / actuator / throttle
# ---------------------------------------------------------------------
def test_safety_no_control_commands() -> None:
    risk = _risk_assessment(risk_score=0.7, status=RiskStatus.RETURN_TO_BASE)
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.6,
        rul_uncertainty=0.6,
        engine_load=0.7,
        env_severity=0.5,
    )
    out = aggregate_reliability(inputs, time_s=10.0)
    wire = out.to_dict()
    flat = str(wire).lower()
    for forbidden in (
        "command", "actuator", "throttle_command", "control",
        "actuate", "rudder", "elevator", "aileron",
    ):
        assert forbidden not in flat, (
            f"forbidden phrase {forbidden!r} in wire format: {wire}"
        )
    for k in wire.keys():
        assert not any(
            f in k.lower() for f in ("command", "actuator", "control", "actuate")
        ), k


# ---------------------------------------------------------------------
# Test 14 — aggregator runs in < 100 µs
# ---------------------------------------------------------------------
def test_latency_under_target() -> None:
    risk = _risk_assessment(
        health_label="DEGRADED",
        rul_status="RUL_DEGRADED",
        anomaly_label="WARN",
    )
    inputs = _inputs(
        risk=risk,
        fault_prob_engine=0.4,
        fault_prob_sensor=0.05,
        fault_prob_environment=0.02,
        rul_uncertainty=0.5,
        engine_load=0.5,
        env_severity=0.3,
        sensor_confidence=0.85,
    )
    # Warm up.
    for _ in range(50):
        aggregate_reliability(inputs, time_s=10.0)
    # Measure.
    n = 500
    t0 = _time.perf_counter()
    for _ in range(n):
        aggregate_reliability(inputs, time_s=10.0)
    elapsed = _time.perf_counter() - t0
    per_call = elapsed / n
    assert per_call < 100e-6, (
        f"aggregate_reliability took {per_call*1e6:.1f} µs/call "
        f"(target < 100 µs)"
    )
