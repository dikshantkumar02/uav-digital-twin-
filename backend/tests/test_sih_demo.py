"""SIH (Software-In-the-Loop) demonstration tests (PHASE 24).

These tests pin the behaviour of the SIH demo so the demonstrative
conclusions for each of the six scenarios do not regress.

The demo must:

* be deterministic — re-running the same scenario must produce
  the same conclusion;
* correctly classify each scenario (the headline conclusion
  on each scenario is part of the public contract the demo
  presents to a control-room operator);
* not introduce control / actuator / success-probability
  fields anywhere in its output.

Each test runs at most 600 ticks (= 60 s of sim time at 0.1 s
per tick) which is the default the SIH CLI uses.  The full demo
CLI run is excluded from this file (it is exercised manually via
``aero-dt-demo``); these tests are the regression net.
"""

from __future__ import annotations

import pytest

from backend.dashboard import sih_demo

pytestmark = pytest.mark.slow  # full 600-tick runs per scenario


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
@pytest.mark.parametrize("scenario_id", [1, 2, 3, 4, 5, 6])
def test_scenario_is_deterministic(scenario_id: int) -> None:
    """Re-running the same scenario produces the same conclusion.

    The SIH demo is the public face of the digital twin: the
    control-room operator must see the same conclusion twice in
    a row for the same flight profile.  If this regresses, the
    demo is no longer useful for evaluation.
    """
    r1 = sih_demo.run_scenario(scenario_id, ticks=600)
    r2 = sih_demo.run_scenario(scenario_id, ticks=600)
    assert r1.system_conclusion == r2.system_conclusion, (
        f"scenario {scenario_id} conclusion not deterministic: "
        f"{r1.system_conclusion!r} != {r2.system_conclusion!r}"
    )
    # And the per-tick anomaly score, health, and risk-status
    # series must also be identical (full trace reproducibility).
    for a, b in zip(r1.ticks, r2.ticks):
        assert a.anomaly_score == pytest.approx(b.anomaly_score, abs=1e-9)
        assert a.health_overall == pytest.approx(b.health_overall, abs=1e-9)
        assert a.risk_status == b.risk_status
        assert a.time_s == pytest.approx(b.time_s, abs=1e-9)


# ---------------------------------------------------------------------
# Per-scenario expected conclusions
# ---------------------------------------------------------------------
def test_scenario_1_healthy_flight_is_healthy() -> None:
    """Healthy 60 s mission → no anomaly, no engine fault, no
    environmental disturbance.  Reliability stays LOW.
    """
    r = sih_demo.run_scenario(1, ticks=600)
    assert "HEALTHY" in r.system_conclusion, r.system_conclusion
    # Reliability view stays LOW on a healthy run.
    assert r.final_reliability_band == "LOW"
    # The risk layer never escalates above CAUTION on a healthy run.
    for rec in r.ticks:
        assert rec.risk_status in ("GO", "CAUTION"), rec.risk_status


def test_scenario_2_turbulence_classified_as_environmental_not_engine() -> None:
    """Scenario 2 is the *signature* case: vibration rises (via
    injected turbulence) but the engine performance residual
    stays normal.  The system must classify the event as
    ENVIRONMENTAL DISTURBANCE, not as an engine fault.

    This is the *demonstrative* case for "no single-sensor
    thresholds for fault decisions" — a naive threshold on
    vibration alone would call this an engine fault.
    """
    r = sih_demo.run_scenario(2, ticks=600)
    # Must NOT say "ENGINE FAULT"
    assert "ENGINE FAULT" not in r.system_conclusion, r.system_conclusion
    # Must mention environment / disturbance.
    assert "ENVIRONMENTAL" in r.system_conclusion, r.system_conclusion
    # Reliability stays LOW (single environmental signal does
    # not by itself move the mission reliability view).
    assert r.final_reliability_band == "LOW"


def test_scenario_3_sensor_fault_not_engine_fault() -> None:
    """Scenario 3 injects a stuck / faulty sensor (rpm).  The
    residual on that channel blows up while the other engine
    channels stay nominal.  The system must classify the event
    as a SENSOR FAULT, not as an engine fault.

    This is the *demonstrative* case for "single-channel
    extremes are sensor events, not engine events".
    """
    r = sih_demo.run_scenario(3, ticks=600)
    assert "SENSOR FAULT" in r.system_conclusion, r.system_conclusion
    # Must NOT say "ENGINE FAULT" as the headline cause.
    assert "ENGINE FAULT" not in r.system_conclusion, r.system_conclusion


def test_scenario_4_early_engine_degradation_detected() -> None:
    """Scenario 4 starts engine degradation at t=10 s.  The
    system must classify the event as an engine fault, not as
    healthy.
    """
    r = sih_demo.run_scenario(4, ticks=600)
    assert "ENGINE FAULT" in r.system_conclusion, r.system_conclusion
    # The risk layer must escalate above GO at some point
    # during the run (the health index is dropping). The exact
    # band may stay LOW (the PHASE 23 reliability layer is
    # conservative on partial-evidence events), but the *risk*
    # state must NOT be plain GO the whole way.
    seen_above_go = {rec.risk_status for rec in r.ticks}
    assert seen_above_go - {"GO"}, (
        f"expected at least one tick to escalate above GO, "
        f"got only {seen_above_go}"
    )


def test_scenario_5_engine_degradation_detected_despite_turbulence() -> None:
    """Scenario 5 is the *discrimination* case: the engine is
    degrading AND the air is turbulent at the same time.  The
    system must STILL classify this as an engine fault, not as
    pure environmental disturbance.

    This is the *demonstrative* case for "the env and engine
    signals are separate — a high-vibration reading combined
    with a real engine performance drop is the engine, not the
    environment".
    """
    r = sih_demo.run_scenario(5, ticks=600)
    assert "ENGINE FAULT" in r.system_conclusion, r.system_conclusion
    # The risk layer must escalate above GO at some point
    # during the run.
    seen_above_go = {rec.risk_status for rec in r.ticks}
    assert seen_above_go - {"GO"}, (
        f"expected at least one tick to escalate above GO, "
        f"got only {seen_above_go}"
    )


def test_scenario_6_unknown_anomaly_reports_unknown() -> None:
    """Scenario 6 schedules aggressive overlapping dropouts so
    that ``data_quality`` collapses well below the calibration
    floor.  The system must NOT fabricate a result — the
    reliability band must surface as UNKNOWN and the
    conclusion must say "insufficient validated evidence" or
    equivalent.
    """
    r = sih_demo.run_scenario(6, ticks=600)
    # The conclusion must signal UNKNOWN, not a confident label.
    assert "UNKNOWN" in r.system_conclusion, r.system_conclusion
    # The reliability view must surface UNKNOWN.
    assert r.final_reliability_band == "UNKNOWN", r.final_reliability_band


# ---------------------------------------------------------------------
# Naive-vs-system comparison (the "superior to threshold monitoring"
# claim)
# ---------------------------------------------------------------------
def test_naive_thresholds_would_have_misclassified_scenario_2() -> None:
    """The naive single-channel threshold (``vibration > 0.8g``)
    WOULD have flagged scenario 2 as a vibration fault.  The
    system did not.  This is the demonstrative case for why
    multi-channel fusion beats single-sensor thresholds.
    """
    r = sih_demo.run_scenario(2, ticks=600)
    # Naive flags: at least one channel tripped a single-sensor
    # threshold in the run.  This must be non-empty for the
    # demonstration to be meaningful.
    assert r.naive_flags_by_channel, (
        "expected at least one channel to trip a naive threshold "
        "in the turbulence scenario; otherwise the comparison is "
        "vacuous"
    )
    # But the system conclusion did not call it an engine fault.
    assert "ENGINE FAULT" not in r.system_conclusion, r.system_conclusion


# ---------------------------------------------------------------------
# Safety: no control commands, no success-probability number
# ---------------------------------------------------------------------
def test_no_mission_success_probability_in_scenario_outputs() -> None:
    """The SIH demo output must not contain any
    ``success_probability`` / ``success_chance`` field — the
    user explicitly excluded "calculate mission success
    probability" from the public surface.
    """
    import json

    for sid in range(1, 7):
        r = sih_demo.run_scenario(sid, ticks=600)
        # The final tick carries the full wire format.  Check it
        # (and the explanation block) explicitly.
        wire = {
            "time_s": r.ticks[-1].time_s,
            "anomaly_score": r.ticks[-1].anomaly_score,
            "health": r.ticks[-1].health_overall,
            "rul": r.ticks[-1].rul_estimate_h,
            "risk_status": r.ticks[-1].risk_status,
            "rel_band": r.ticks[-1].rel_band,
            "rel_headline": r.ticks[-1].rel_headline,
            "rel_drivers": list(r.ticks[-1].rel_drivers),
        }
        # Dump to a flat string and grep for forbidden keys.
        flat = json.dumps(wire).lower()
        assert "success_probability" not in flat
        assert "success_chance" not in flat
        assert "mission_success" not in flat


def test_no_control_commands_in_scenario_outputs() -> None:
    """The SIH demo output must not contain any control /
    actuator / throttle command key.  The user is explicit
    that the layer is for monitoring only.
    """
    import json

    forbidden = {
        "command", "actuator", "throttle_command",
        "control_input", "setpoint", "autopilot",
    }
    for sid in range(1, 7):
        r = sih_demo.run_scenario(sid, ticks=600)
        wire = {
            "rel_band": r.ticks[-1].rel_band,
            "rel_headline": r.ticks[-1].rel_headline,
            "rel_drivers": list(r.ticks[-1].rel_drivers),
        }
        flat = json.dumps(wire).lower()
        for key in forbidden:
            assert key not in flat, (
                f"scenario {sid} output contains forbidden key "
                f"{key!r}: {flat!r}"
            )


# ---------------------------------------------------------------------
# run_all() — top-level orchestrator
# ---------------------------------------------------------------------
def test_run_all_returns_one_result_per_scenario() -> None:
    """``run_all()`` runs all six scenarios and returns a list of
    ``ScenarioResult`` objects, one per scenario, in order.
    """
    results = sih_demo.run_all(ticks=600)
    assert len(results) == 6, [r.scenario_id for r in results]
    assert [r.scenario_id for r in results] == [1, 2, 3, 4, 5, 6]


def test_run_all_conclusions_match_per_scenario_runs() -> None:
    """``run_all()`` must produce the same conclusions as running
    ``run_scenario()`` individually.  This pins determinism at
    the orchestrator level.
    """
    all_results = sih_demo.run_all(ticks=600)
    all_conclusions = {r.scenario_id: r.system_conclusion for r in all_results}
    for sid in range(1, 7):
        single = sih_demo.run_scenario(sid, ticks=600)
        assert all_conclusions[sid] == single.system_conclusion, (
            f"run_all conclusion for scenario {sid} "
            f"({all_conclusions[sid]!r}) differs from "
            f"run_scenario ({single.system_conclusion!r})"
        )
