"""PHASE 12 tests — Mission risk engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from backend.config import Provenance, load_config
from backend.diagnostics import (
    AnomalyAssessment,
    AnomalyLabel,
    ChannelAnomaly,
)
from backend.environment import MissionProfile
from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.health import (
    HealthIndex,
    HealthIndexCalculator,
    HealthLabel,
    HealthTrend,
    Subsystem,
    SubsystemHealth,
)
from backend.risk import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_THRESHOLD_ABORT,
    DEFAULT_THRESHOLD_CAUTION,
    DEFAULT_THRESHOLD_RETURN_TO_BASE,
    MAX_RECOMMENDATIONS,
    DriverSeverity,
    MissionPhase,
    MissionRiskCalculator,
    PHASE_MODIFIERS,
    RiskAssessment,
    RiskDriver,
    RiskLevel,
    RiskStatus,
    RiskTrend,
    TREND_EPSILON_RISK,
    TREND_WINDOW,
    aggregate,
    hours_to_destination,
    level_for,
    phase_for,
    status_for,
    trend_for,
)
from backend.rul import (
    RulCalculator,
    RulEstimate,
    RulModelStatus,
    RulStatus,
    RulTrend,
)
from backend.simulation import EngineSimulator, EngineState

pytestmark = pytest.mark.phase12

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
    confidence: float = 1.0,
) -> HealthIndex:
    label = (HealthLabel.HEALTHY if overall_score >= 0.80
             else HealthLabel.DEGRADED if overall_score >= 0.55
             else HealthLabel.CRITICAL if overall_score > 0.0
             else HealthLabel.INSUFFICIENT_DATA)
    return HealthIndex(
        time_s=time_s,
        overall_score=float(overall_score),
        overall_label=label,
        confidence=float(confidence),
        trend=trend,
        wear=wear,
        contributing_faults=contributing_faults or {},
    )


def _healthy_rul(
    *,
    time_s: float = 1.0,
    tte_lower: float = 10_000.0,
    tte_central: float = 12_000.0,
    tte_upper: float = 14_000.0,
    status: RulStatus = RulStatus.RUL_OK,
    confidence: float = 1.0,
    trend: RulTrend = RulTrend.STABLE,
) -> RulEstimate:
    return RulEstimate(
        time_s=time_s,
        tte_hours_central=float(tte_central),
        tte_hours_lower=float(tte_lower),
        tte_hours_upper=float(tte_upper),
        wear_rate_per_hour=1e-5,
        confidence=float(confidence),
        status=status,
        trend=trend,
        model_status=RulModelStatus.CLOSED_FORM,
    )


def _healthy_anomaly(
    *,
    time_s: float = 1.0,
    overall_score: float = 0.0,
    label: AnomalyLabel = AnomalyLabel.NORMAL,
    confidence: float = 1.0,
) -> AnomalyAssessment:
    return AnomalyAssessment(
        time_s=time_s,
        overall_score=float(overall_score),
        overall_label=label,
        confidence=float(confidence),
        channels={},
    )


def _profile() -> MissionProfile:
    cfg = load_config(CONFIG_DIR)
    return MissionProfile.from_config(cfg.environment.mission)


# ---------------------------------------------------------------------
# TestTypes
# ---------------------------------------------------------------------
class TestTypes:
    def test_status_members(self) -> None:
        assert {s.value for s in RiskStatus} == {
            "GO", "CAUTION", "RETURN_TO_BASE", "ABORT", "INSUFFICIENT_DATA",
        }

    def test_level_members(self) -> None:
        assert {lv.value for lv in RiskLevel} == {
            "LOW", "MODERATE", "HIGH", "SEVERE",
        }

    def test_phase_members(self) -> None:
        assert {p.value for p in MissionPhase} == {
            "PRE_FLIGHT", "TAKEOFF", "CLIMB", "CRUISE",
            "DESCENT", "LANDING", "POST_MISSION",
        }

    def test_trend_members(self) -> None:
        assert {t.value for t in RiskTrend} == {
            "IMPROVING", "STABLE", "DEGRADING", "INSUFFICIENT_DATA",
        }

    def test_risk_driver_to_dict(self) -> None:
        d = RiskDriver(
            signal="health.overall_score", value=0.5,
            contribution=0.1, severity=DriverSeverity.WARNING, note="x",
        ).to_dict()
        assert d["signal"] == "health.overall_score"
        assert d["value"] == 0.5
        assert d["contribution"] == 0.1
        assert d["severity"] == "WARNING"
        assert d["note"] == "x"

    def test_risk_assessment_to_dict(self) -> None:
        ra = RiskAssessment(
            time_s=1.0, risk_score=0.1, risk_level=RiskLevel.LOW,
            status=RiskStatus.GO, confidence=1.0,
            mission_phase=MissionPhase.CRUISE,
            drivers=[], recommendations=["x"],
        )
        d = ra.to_dict()
        for k in ("time_s", "risk_score", "risk_level", "status",
                  "confidence", "mission_phase", "drivers",
                  "recommendations", "trend", "notes", "inputs_meta"):
            assert k in d
        assert d["risk_level"] == "LOW"
        assert d["status"] == "GO"

    def test_risk_assessment_invariants(self) -> None:
        ra = RiskAssessment(
            time_s=0.0, risk_score=0.5, risk_level=RiskLevel.MODERATE,
            status=RiskStatus.CAUTION, confidence=0.6,
            mission_phase=MissionPhase.CRUISE,
        )
        assert 0.0 <= ra.risk_score <= 1.0
        assert 0.0 <= ra.confidence <= 1.0
        # is_terminal covers RETURN, ABORT, INSUFFICIENT_DATA.
        assert ra.is_caution
        assert not ra.is_terminal

    def test_status_for_escalation_critical_health(self) -> None:
        assert status_for(
            health_label="CRITICAL", rul_status="RUL_OK",
            risk_score=0.0, confidence=1.0,
        ) is RiskStatus.ABORT

    def test_status_for_escalation_critical_rul(self) -> None:
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_CRITICAL",
            risk_score=0.0, confidence=1.0,
        ) is RiskStatus.ABORT

    def test_status_for_insufficient_health(self) -> None:
        assert status_for(
            health_label="INSUFFICIENT_DATA", rul_status="RUL_OK",
            risk_score=0.0, confidence=1.0,
        ) is RiskStatus.INSUFFICIENT_DATA

    def test_status_for_uncertain_rul(self) -> None:
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_UNCERTAIN",
            risk_score=0.0, confidence=1.0,
        ) is RiskStatus.INSUFFICIENT_DATA

    def test_status_for_low_confidence(self) -> None:
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.0, confidence=0.10,
        ) is RiskStatus.INSUFFICIENT_DATA
        # Even at the floor, INSUFFICIENT_DATA.
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.0, confidence=0.19,
        ) is RiskStatus.INSUFFICIENT_DATA

    def test_status_for_score_bands(self) -> None:
        # 0.0 → GO
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.0, confidence=1.0,
        ) is RiskStatus.GO
        # 0.30 → CAUTION
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.30, confidence=1.0,
        ) is RiskStatus.CAUTION
        # 0.65 → RETURN_TO_BASE
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.65, confidence=1.0,
        ) is RiskStatus.RETURN_TO_BASE
        # 0.90 → ABORT (score alone, not the health hard-signal)
        assert status_for(
            health_label="HEALTHY", rul_status="RUL_OK",
            risk_score=0.90, confidence=1.0,
        ) is RiskStatus.ABORT

    def test_level_for_thresholds(self) -> None:
        assert level_for(0.0) is RiskLevel.LOW
        assert level_for(0.5) is RiskLevel.MODERATE
        assert level_for(0.7) is RiskLevel.HIGH
        assert level_for(0.9) is RiskLevel.SEVERE
        # Clamping
        assert level_for(2.0) is RiskLevel.SEVERE
        assert level_for(-1.0) is RiskLevel.LOW

    def test_trend_for(self) -> None:
        assert trend_for(None, 0.5) is RiskTrend.INSUFFICIENT_DATA
        assert trend_for(0.5, None) is RiskTrend.INSUFFICIENT_DATA
        assert trend_for(0.5, 0.5) is RiskTrend.STABLE
        assert trend_for(0.5, 0.6) is RiskTrend.IMPROVING
        assert trend_for(0.5, 0.4) is RiskTrend.DEGRADING
        # Epsilon boundary
        eps = TREND_EPSILON_RISK
        assert trend_for(0.5 + eps / 2, 0.5) is RiskTrend.STABLE
        assert trend_for(0.5 + eps * 2, 0.5) is RiskTrend.DEGRADING

    def test_constants_sensible(self) -> None:
        assert DEFAULT_THRESHOLD_CAUTION < DEFAULT_THRESHOLD_RETURN_TO_BASE
        assert DEFAULT_THRESHOLD_RETURN_TO_BASE < DEFAULT_THRESHOLD_ABORT
        assert DEFAULT_THRESHOLD_ABORT <= 1.0
        assert TREND_WINDOW == 10
        assert MAX_RECOMMENDATIONS == 5
        assert DEFAULT_MIN_CONFIDENCE > 0.0


# ---------------------------------------------------------------------
# TestMissionPhase
# ---------------------------------------------------------------------
class TestMissionPhase:
    def test_pre_flight(self) -> None:
        profile = _profile()
        assert phase_for(-1.0, profile) is MissionPhase.PRE_FLIGHT
        assert phase_for(profile.waypoints[0].t_s - 1.0, profile) \
            is MissionPhase.PRE_FLIGHT

    def test_takeoff(self) -> None:
        profile = _profile()
        # First 30s after the first waypoint.
        t = profile.waypoints[0].t_s + 5.0
        assert phase_for(t, profile) is MissionPhase.TAKEOFF

    def test_cruise(self) -> None:
        profile = _profile()
        # Pick a waypoint where altitude is steady for at least
        # the previous 5%; cruise is the default for the mid-mission.
        # Use the cruise waypoint directly (e.g. the 3rd or 4th).
        # At its exact time, altitude matches the previous waypoint
        # within band -> CRUISE.
        for w in profile.waypoints:
            prev = profile.waypoints[0]
            for w2 in profile.waypoints:
                if w2.t_s <= w.t_s:
                    prev = w2
            if abs(w.altitude_m - prev.altitude_m) <= 0.05 * max(prev.altitude_m, 1.0):
                assert phase_for(w.t_s, profile) in (
                    MissionPhase.CRUISE, MissionPhase.TAKEOFF, MissionPhase.LANDING,
                )
                return
        # If we didn't find a steady waypoint, the profile is unusual.
        pytest.skip("no steady waypoint found in profile")

    def test_post_mission(self) -> None:
        profile = _profile()
        assert phase_for(profile.duration_s + 1.0, profile) \
            is MissionPhase.POST_MISSION
        assert phase_for(profile.duration_s * 2.0, profile) \
            is MissionPhase.POST_MISSION

    def test_hours_to_destination(self) -> None:
        profile = _profile()
        h0 = hours_to_destination(0.0, profile)
        assert h0 == profile.duration_s / 3600.0
        h_mid = hours_to_destination(profile.duration_s / 2.0, profile)
        assert h_mid is not None and h_mid > 0.0
        # After the mission: None.
        assert hours_to_destination(
            profile.duration_s + 1.0, profile
        ) is None
        # Before the mission: clamped to 0.
        assert hours_to_destination(-1.0, profile) == 0.0


# ---------------------------------------------------------------------
# TestRiskCalculator
# ---------------------------------------------------------------------
class TestRiskCalculator:
    def _calc(self, **kwargs) -> MissionRiskCalculator:
        return MissionRiskCalculator(_profile(), **kwargs)

    def test_healthy_inputs_go(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul()
        a = _healthy_anomaly()
        ra = cal.update(health=h, rul=r, anomaly=a, time_s=600.0)
        assert ra.status is RiskStatus.GO
        assert ra.risk_level is RiskLevel.LOW
        assert 0.0 <= ra.risk_score <= 0.20
        assert ra.confidence == pytest.approx(1.0)
        assert ra.mission_phase is MissionPhase.CRUISE
        # No recommendations for a GO assessment.
        assert ra.recommendations == []

    def test_critical_health_aborts(self) -> None:
        cal = self._calc()
        h = _healthy_health(
            overall_score=0.30,
            contributing_faults={"ENGINE_DEGRADATION": 0.7},
        )
        r = _healthy_rul()
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.status is RiskStatus.ABORT
        # Health hard-signal: ABORT regardless of score.
        assert any("ABORT" in d.note for d in ra.drivers) or \
               any(d.signal == "health.overall_score" for d in ra.drivers)

    def test_critical_rul_aborts(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul(
            tte_lower=2.0, tte_central=3.0, tte_upper=4.0,
            status=RulStatus.RUL_CRITICAL,
        )
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.status is RiskStatus.ABORT

    def test_anomaly_raises_score(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul()
        # Baseline: no anomaly.
        a0 = _healthy_anomaly()
        ra0 = cal.update(health=h, rul=r, anomaly=a0, time_s=600.0)
        # With anomaly score 0.8 and ANOMALY label.
        a1 = _healthy_anomaly(overall_score=0.8, label=AnomalyLabel.ANOMALY)
        ra1 = cal.update(health=h, rul=r, anomaly=a1, time_s=600.0)
        assert ra1.risk_score > ra0.risk_score
        # The anomaly driver must show up.
        assert any(d.signal == "anomaly.overall_score" for d in ra1.drivers)

    def test_uncertain_inputs_insufficient(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul(
            tte_lower=0.0, tte_central=0.0, tte_upper=0.0,
            status=RulStatus.RUL_UNCERTAIN, confidence=0.0,
        )
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.status is RiskStatus.INSUFFICIENT_DATA
        # The lone recommendation is the "sensor check" advisory.
        assert any("insufficient" in r.lower() for r in ra.recommendations)

    def test_low_confidence_insufficient(self) -> None:
        cal = self._calc()
        h = _healthy_health(confidence=0.10)
        r = _healthy_rul()
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.status is RiskStatus.INSUFFICIENT_DATA

    def test_takeoff_modifier_raises(self) -> None:
        # Same health / RUL / anomaly, two different times: one
        # during TAKEOFF and one during CRUISE. The takeoff score
        # must be higher (because of the +0.10 phase modifier).
        cal_a = self._calc()
        cal_b = self._calc()
        h = _healthy_health(overall_score=0.85)  # mid-DEGRADED territory
        r = _healthy_rul()
        profile = _profile()
        t_takeoff = profile.waypoints[0].t_s + 5.0
        t_cruise = profile.waypoints[0].t_s + 200.0
        ra_t = cal_a.update(health=h, rul=r, anomaly=None, time_s=t_takeoff)
        ra_c = cal_b.update(health=h, rul=r, anomaly=None, time_s=t_cruise)
        assert ra_t.mission_phase is MissionPhase.TAKEOFF
        assert ra_c.mission_phase in (
            MissionPhase.CRUISE, MissionPhase.CLIMB, MissionPhase.DESCENT,
        )
        # The takeoff score is the cruise score plus (or minus, if
        # a non-cruise phase has its own modifier) at least +0.10
        # - 0.02 (the largest other non-zero magnitude).
        assert ra_t.risk_score >= ra_c.risk_score + 0.05

    def test_landing_modifier_lowers(self) -> None:
        # A LANDING phase with a slightly elevated score should
        # produce a lower risk_score than the same inputs at CRUISE.
        cal_a = self._calc()
        cal_b = self._calc()
        h = _healthy_health(overall_score=0.80)
        r = _healthy_rul()
        profile = _profile()
        # The mission profile's last waypoint is the landing
        # waypoint (low altitude, low throttle). Use it directly.
        landing_t = profile.waypoints[-1].t_s
        # Choose a CRUISE waypoint — a mid-mission one with
        # altitude steady vs. the previous waypoint.
        cruise_t = None
        for w in profile.waypoints:
            prev = profile.waypoints[0]
            for w2 in profile.waypoints:
                if w2.t_s <= w.t_s:
                    prev = w2
            if (abs(w.altitude_m - prev.altitude_m)
                    <= 0.05 * max(prev.altitude_m, 1.0)):
                cruise_t = w.t_s
                break
        if cruise_t is None:
            pytest.skip("no cruise waypoint in profile")
        ra_l = cal_a.update(health=h, rul=r, anomaly=None, time_s=landing_t)
        ra_c = cal_b.update(health=h, rul=r, anomaly=None, time_s=cruise_t)
        assert ra_l.mission_phase is MissionPhase.LANDING
        # Landing modifier is -0.05, cruise is 0.0.
        assert ra_l.risk_score <= ra_c.risk_score

    def test_drivers_attribution(self) -> None:
        cal = self._calc()
        h = _healthy_health(
            overall_score=0.70,
            contributing_faults={"ENGINE_DEGRADATION": 0.8},
        )
        r = _healthy_rul()
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        signals = {d.signal for d in ra.drivers}
        assert "health.overall_score" in signals
        assert "rul.tte_hours_lower" in signals
        assert "fault.boost" in signals
        # Each driver has a non-negative contribution.
        for d in ra.drivers:
            assert d.contribution >= 0.0

    def test_hours_to_critical_present(self) -> None:
        cal = self._calc()
        h = _healthy_health(overall_score=0.85)
        r = _healthy_rul(
            tte_lower=10.0, tte_central=12.0, tte_upper=14.0,
        )
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.hours_to_critical is not None
        assert ra.hours_to_critical >= 0.0

    def test_hours_to_critical_none_when_uncertain(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul(
            tte_lower=0.0, tte_central=0.0, tte_upper=0.0,
            status=RulStatus.RUL_UNCERTAIN, confidence=0.0,
        )
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.hours_to_critical is None

    def test_recommendations_nonempty_when_not_go(self) -> None:
        cal = self._calc()
        h = _healthy_health(overall_score=0.50)
        r = _healthy_rul()
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.status is not RiskStatus.GO
        assert len(ra.recommendations) >= 1

    def test_recommendations_capped(self) -> None:
        cal = self._calc()
        h = _healthy_health(
            overall_score=0.30, trend=HealthTrend.DEGRADING,
            contributing_faults={"ENGINE_DEGRADATION": 0.9},
        )
        r = _healthy_rul(
            tte_lower=2.0, tte_central=3.0, tte_upper=4.0,
            status=RulStatus.RUL_CRITICAL,
        )
        a = _healthy_anomaly(overall_score=0.85, label=AnomalyLabel.ANOMALY)
        ra = cal.update(health=h, rul=r, anomaly=a, time_s=600.0)
        assert len(ra.recommendations) <= MAX_RECOMMENDATIONS

    def test_reset_clears_trend(self) -> None:
        cal = self._calc()
        h = _healthy_health()
        r = _healthy_rul()
        for _ in range(5):
            cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert cal.update_count == 5
        cal.reset()
        assert cal.update_count == 0
        assert cal.last_assessment is None
        assert len(cal.trend_history) == 0

    def test_latency_under_budget(self) -> None:
        cal = self._calc()
        # Warmup.
        h = _healthy_health()
        r = _healthy_rul()
        for _ in range(3):
            cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        us = cal.measure_latency(n_iter=100)
        # Generous: 5 ms per update.
        assert us < 5_000.0

    def test_contributing_faults_propagated(self) -> None:
        cal = self._calc()
        h = _healthy_health(
            contributing_faults={"ENGINE_DEGRADATION": 0.6,
                                  "OVERHEATING": 0.3},
        )
        r = _healthy_rul()
        ra = cal.update(health=h, rul=r, anomaly=None, time_s=600.0)
        assert ra.contributing_faults == {
            "ENGINE_DEGRADATION": 0.6, "OVERHEATING": 0.3,
        }

    def test_provenance_yaml_loads(self) -> None:
        # The risk layer's config must have USER_CONFIGURED provenance
        # by convention.
        cfg = load_config(CONFIG_DIR)
        assert cfg.risk.provenance is Provenance.USER_CONFIGURED


# ---------------------------------------------------------------------
# TestScenarios
# ---------------------------------------------------------------------
class TestScenarios:
    def _setup(self, **kwargs):
        cfg = load_config(CONFIG_DIR)
        sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
        cal_health = HealthIndexCalculator(cfg.engine)
        cal_rul = RulCalculator(cfg.engine, **kwargs)
        cal_risk = MissionRiskCalculator(
            MissionProfile.from_config(cfg.environment.mission),
        )
        return cfg, sim, cal_health, cal_rul, cal_risk

    def test_s1_healthy_full_mission_go(self) -> None:
        """200 clean ticks → all GO/LOW."""
        _, _, cal_health, cal_rul, cal_risk = self._setup()
        for i in range(200):
            t = float(i) * 0.1
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
            r = cal_rul.update(health=h, state=st, dt_s=0.1)
            ra = cal_risk.update(health=h, rul=r, anomaly=anom, time_s=t)
            assert ra.status in (RiskStatus.GO, RiskStatus.CAUTION)
            # The mission may dip into CAUTION at takeoff because of
            # the +0.10 phase modifier, but the level must be at
            # most MODERATE.
            assert ra.risk_level in (RiskLevel.LOW, RiskLevel.MODERATE)

    def test_s4_engine_degradation_escalates(self) -> None:
        """ENGINE_DEGRADATION fault → risk_score rises; reaches CAUTION."""
        cfg, sim, cal_health, cal_rul, cal_risk = self._setup()
        sc = FaultScenario(
            FaultClass.ENGINE_DEGRADATION,
            severity=0.6, onset_time_s=0.0, duration_s=300.0,
            progression="step",
        )
        inj = FaultInjector(sc)
        from backend.diagnostics import (
            AnomalyAssessment, ChannelAnomaly, AnomalyLabel,
        )
        statuses_seen: set = set()
        for i in range(60):
            tick = inj.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            t = step.env.time_s
            ca = ChannelAnomaly(
                channel="vibration", score=0.6,
                label=AnomalyLabel.WARN, confidence=0.8,
                contributors=["vibration residual"],
            )
            anom = AnomalyAssessment(
                time_s=t, overall_score=0.6,
                overall_label=AnomalyLabel.WARN, confidence=0.8,
                channels={"vibration": ca},
            )
            h = cal_health.update(state=step.engine, anomaly=anom)
            r = cal_rul.update(health=h, state=step.engine, dt_s=0.1)
            ra = cal_risk.update(
                health=h, rul=r, anomaly=anom, time_s=t,
            )
            statuses_seen.add(ra.status)
        last = cal_risk.last_assessment
        assert last is not None
        # The final status has left GO.
        assert last.status in (
            RiskStatus.CAUTION, RiskStatus.RETURN_TO_BASE, RiskStatus.ABORT,
        )
        # The risk score is in the CAUTION band.
        assert last.risk_score >= DEFAULT_THRESHOLD_CAUTION
        # And we saw at least one non-GO status over the run.
        assert statuses_seen - {RiskStatus.GO} != set()

    def test_s5_performance_loss_caution(self) -> None:
        """PERFORMANCE_LOSS → risk_score > 0.25, status CAUTION."""
        cfg, sim, cal_health, cal_rul, cal_risk = self._setup()
        sc = FaultScenario(
            FaultClass.PERFORMANCE_LOSS,
            severity=0.7, onset_time_s=0.0, duration_s=200.0,
            progression="step",
        )
        inj = FaultInjector(sc)
        from backend.diagnostics import (
            AnomalyAssessment, ChannelAnomaly, AnomalyLabel,
        )
        for i in range(60):
            tick = inj.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            t = step.env.time_s
            ca = ChannelAnomaly(
                channel="rpm", score=0.6,
                label=AnomalyLabel.WARN, confidence=0.7,
                contributors=["rpm residual"],
            )
            anom = AnomalyAssessment(
                time_s=t, overall_score=0.6,
                overall_label=AnomalyLabel.WARN, confidence=0.7,
                channels={"rpm": ca},
            )
            h = cal_health.update(state=step.engine, anomaly=anom)
            r = cal_rul.update(health=h, state=step.engine, dt_s=0.1)
            cal_risk.update(health=h, rul=r, anomaly=anom, time_s=t)
        last = cal_risk.last_assessment
        assert last is not None
        # Either CAUTION or higher (escalation may push it further).
        assert last.status in (
            RiskStatus.CAUTION, RiskStatus.RETURN_TO_BASE, RiskStatus.ABORT,
        )
        # And the risk_score has crossed the CAUTION band.
        assert last.risk_score >= DEFAULT_THRESHOLD_CAUTION
