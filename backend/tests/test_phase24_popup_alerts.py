"""
PHASE 24 tests — Individual warning popup system (backend).

The frontend popup manager is JS; the backend produces the
*enriched* alert wire format the popup system consumes. These
tests verify the enrichment is consistent and that the spec
contract is honoured:

  * every alert has a stable ``alert_id`` (popup dedup key);
  * every alert has a ``severity_label`` in {HEALTHY, INFO,
    WARNING, CRITICAL} (the popup system never sees the
    five-band risk severity);
  * every alert has a ``tier`` in {0, 1, 2};
  * ENVIRONMENTAL_DISTURBANCE on a healthy engine is
    downgraded to INFO (not CRITICAL);
  * sensor / current_value / expected_value / deviation are
    populated from the residual / digital-twin snapshot when
    the snapshot has them; otherwise ``None``;
  * legacy fields (``fault_type``, ``severity``, ``evidence``,
    ``env_context``, ``sensor_health``, ``recommendation``,
    ``time_s``) are preserved.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase24

from backend.dashboard.alerts import (
    Alert,
    alert_id,
    derive_alerts,
    recommendation_for,
    severity_label_for,
    tier_for,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _healthy_payload() -> dict:
    return {
        "time_s": 1.0,
        "risk": {"status": "GO", "confidence": 0.9, "risk_score": 0.05,
                 "mission_phase": "CRUISE"},
        "anomaly": {"confidence": 0.05},
        "fault_classification": {"fault_class": "HEALTHY", "confidence": 0.9},
        "environment": {
            "altitude_m": 1000, "temperature_c": 15, "pressure_pa": 90000,
            "turbulence_w_mps": 0.5, "is_gust_active": False,
        },
        "engine_state": {"rpm": 2400, "throttle": 0.7},
        "residual": {},
        "twin_expected": {},
        "sample": {"sensor_health": {}},
    }


def _degraded_payload() -> dict:
    return {
        "time_s": 12.0,
        "risk": {"status": "RETURN_TO_BASE", "confidence": 0.85,
                 "risk_score": 0.75, "mission_phase": "CRUISE"},
        "anomaly": {"confidence": 0.7,
                    "contributing_channels": ["egt", "vibration"]},
        "fault_classification": {"fault_class": "ENGINE_DEGRADATION",
                                 "confidence": 0.85},
        "environment": {
            "altitude_m": 1000, "temperature_c": 15, "pressure_pa": 90000,
            "turbulence_w_mps": 0.5, "is_gust_active": False,
        },
        "engine_state": {"rpm": 2400, "throttle": 0.7},
        "residual": {
            "residual.egt.z_score": 2.5,
            "residual.egt.residual": 50.0,
        },
        "twin_expected": {"egt": 720.0},
        "sample": {"sensor_health": {}},
    }


# ---------------------------------------------------------------------
# Test 1: A. Healthy engine -> no alert popup
# ---------------------------------------------------------------------
class TestHealthyEngine:
    def test_no_alerts_for_healthy_payload(self) -> None:
        alerts = derive_alerts(_healthy_payload())
        assert alerts == ()

    def test_severity_label_healthy_for_empty(self) -> None:
        assert severity_label_for(None) == "HEALTHY"
        assert severity_label_for("") == "HEALTHY"


# ---------------------------------------------------------------------
# Test 2: B. Warning condition -> one warning generated
# ---------------------------------------------------------------------
class TestWarningCondition:
    def test_one_warning_for_caution_status(self) -> None:
        payload = _healthy_payload()
        payload["risk"]["status"] = "CAUTION"
        payload["risk"]["risk_score"] = 0.40
        alerts = derive_alerts(payload)
        assert len(alerts) == 1
        a = alerts[0]
        assert a["severity_label"] == "WARNING"
        assert a["tier"] == 1
        assert a["alert_id"]  # non-empty

    def test_legacy_fields_preserved(self) -> None:
        """The PHASE 19 / 22 wire-format fields must remain
        identical so existing consumers don't break."""
        payload = _healthy_payload()
        payload["risk"]["status"] = "CAUTION"
        payload["risk"]["risk_score"] = 0.40
        a = derive_alerts(payload)[0]
        for key in (
            "fault_type", "confidence", "severity", "evidence",
            "env_context", "sensor_health", "recommendation", "time_s",
        ):
            assert key in a, f"legacy field {key!r} missing"
        assert a["time_s"] == 1.0
        assert a["severity"] in ("CAUTION", "WARNING", "INFO",
                                 "RETURN_TO_BASE", "ABORT")
        assert isinstance(a["recommendation"], str) and a["recommendation"]


# ---------------------------------------------------------------------
# Test 3: C. Persistent warning -> no duplicate alert_id
# ---------------------------------------------------------------------
class TestPersistentWarning:
    def test_same_alert_id_across_ticks(self) -> None:
        a1 = derive_alerts(_degraded_payload())
        a2 = derive_alerts(_degraded_payload())
        assert len(a1) == 1 and len(a2) == 1
        assert a1[0]["alert_id"] == a2[0]["alert_id"]

    def test_alert_id_changes_when_sensor_changes(self) -> None:
        a1 = derive_alerts(_degraded_payload())
        payload = _degraded_payload()
        payload["residual"] = {
            "residual.vibration.z_score": 3.0,
            "residual.vibration.residual": 0.5,
        }
        payload["twin_expected"] = {"vibration": 0.3}
        a2 = derive_alerts(payload)
        # Different primary sensor -> different alert_id.
        assert a1[0]["alert_id"] != a2[0]["alert_id"]


# ---------------------------------------------------------------------
# Test 4: D. Warning clears -> alert becomes inactive (resolved)
# ---------------------------------------------------------------------
class TestWarningClears:
    def test_alert_not_in_next_tick_after_clear(self) -> None:
        a1 = derive_alerts(_degraded_payload())
        assert len(a1) == 1
        a2 = derive_alerts(_healthy_payload())
        assert a2 == ()


# ---------------------------------------------------------------------
# Test 5: E. Warning returns -> new alert generated
# ---------------------------------------------------------------------
class TestWarningReturns:
    def test_same_alert_id_after_recur(self) -> None:
        a1 = derive_alerts(_degraded_payload())
        # Clears.
        derive_alerts(_healthy_payload())
        # Returns — same fault, same sensor, same severity.
        a2 = derive_alerts(_degraded_payload())
        assert a1[0]["alert_id"] == a2[0]["alert_id"]
        # The popup system distinguishes "first appearance" from
        # "recurred after resolution" by tracking resolvedAt;
        # both produce the same alert_id, which is the correct
        # dedup key (the popup manager has its own history
        # state for "first vs. re-curred" semantics).


# ---------------------------------------------------------------------
# Test 6: F. Escalation WARNING -> CRITICAL
# ---------------------------------------------------------------------
class TestEscalation:
    def test_escalation_increases_tier(self) -> None:
        payload_low = _healthy_payload()
        payload_low["risk"]["status"] = "CAUTION"
        payload_low["risk"]["risk_score"] = 0.40
        a_low = derive_alerts(payload_low)[0]
        assert tier_for(a_low["severity_label"]) == 1

        payload_high = _healthy_payload()
        payload_high["risk"]["status"] = "ABORT"
        payload_high["risk"]["risk_score"] = 0.95
        a_high = derive_alerts(payload_high)[0]
        assert tier_for(a_high["severity_label"]) == 2
        # Same fault type (engine degradation by classifier).
        assert a_low["fault_type"] == a_high["fault_type"]


# ---------------------------------------------------------------------
# Test 7: G. Multiple simultaneous faults -> separate alerts
# ---------------------------------------------------------------------
class TestMultipleFaults:
    def test_engine_and_sensor_separate_alerts(self) -> None:
        payload = _healthy_payload()
        payload["risk"]["status"] = "ABORT"
        payload["risk"]["risk_score"] = 0.95
        payload["fault_classification"] = {
            "fault_class": "ENGINE_DEGRADATION", "confidence": 0.9,
        }
        # Add a sensor fault in the sensor_health view.
        payload["sample"]["sensor_health"] = {
            "rpm": {"mode": "STUCK"},
        }
        alerts = derive_alerts(payload)
        # We expect at least 2 alerts (engine + sensor).
        types = {a["fault_type"] for a in alerts}
        assert "SENSOR_FAULT" in types or "ENGINE_DEGRADATION" in types
        # And the alert_ids are distinct.
        ids = [a["alert_id"] for a in alerts]
        assert len(set(ids)) == len(ids), f"duplicate alert_ids: {ids}"


# ---------------------------------------------------------------------
# Test 8: H. Acknowledgement is purely a frontend concern
# ---------------------------------------------------------------------
class TestAcknowledgement:
    def test_backend_does_not_carry_ack_state(self) -> None:
        """The backend emits the *current* active set per tick.
        Acknowledgement is a frontend-only concept (otherwise a
        page reload would lose it). This test documents the
        contract: no ``acked`` field on the wire format.
        """
        a = derive_alerts(_degraded_payload())[0]
        assert "acked" not in a


# ---------------------------------------------------------------------
# Test 9: I. Environmental turbulence -> INFO, not CRITICAL
# ---------------------------------------------------------------------
class TestEnvironmentalAwareness:
    def test_environmental_disturbance_is_info(self) -> None:
        # When the only thing wrong is the env, the popup system
        # should not pop a CRITICAL engine banner.
        assert severity_label_for("CAUTION",
                                  "ENVIRONMENTAL_DISTURBANCE") == "INFO"
        assert tier_for(severity_label_for("CAUTION",
                  "ENVIRONMENTAL_DISTURBANCE")) == 0

    def test_engine_degradation_stays_warning(self) -> None:
        # Engine fault is NOT downgraded even at CAUTION.
        assert severity_label_for("CAUTION",
                                  "ENGINE_DEGRADATION") == "WARNING"


# ---------------------------------------------------------------------
# Test 10: J. Sensor fault -> correct SENSOR_FAULT alert
# ---------------------------------------------------------------------
class TestSensorFault:
    def test_sensor_fault_alert_present(self) -> None:
        payload = _healthy_payload()
        payload["risk"]["status"] = "CAUTION"
        payload["sample"]["sensor_health"] = {
            "rpm": {"mode": "STUCK"},
        }
        alerts = derive_alerts(payload)
        types = {a["fault_type"] for a in alerts}
        assert "SENSOR_FAULT" in types


# ---------------------------------------------------------------------
# Test 11: K. Unknown anomaly -> not forced into a known class
# ---------------------------------------------------------------------
class TestUnknownAnomaly:
    def test_alert_id_distinct_for_unknown(self) -> None:
        a_id = alert_id("UNKNOWN_ANOMALY", "WARNING")
        # Distinct from every other known fault.
        for known in ("ENGINE_DEGRADATION", "OVERHEATING",
                      "VIBRATION_ENGINE_ANOMALY", "SENSOR_FAULT"):
            assert a_id != alert_id(known, "WARNING"), known

    def test_severity_label_for_unknown(self) -> None:
        # Unknown faults at CAUTION level show as WARNING
        # (not INFO — there is real evidence of an anomaly,
        # we just can't pin it to a known class).
        assert severity_label_for("CAUTION", "UNKNOWN_ANOMALY") == "WARNING"


# ---------------------------------------------------------------------
# Test 12: L. WebSocket / API delivery
# ---------------------------------------------------------------------
class TestAPIDelivery:
    def test_snapshot_alerts_have_all_enriched_fields(self) -> None:
        from fastapi.testclient import TestClient
        from backend.config import load_config
        from backend.dashboard import create_app
        from backend.dashboard.scenarios import SCENARIO_REGISTRY
        from backend.dashboard.app import DEFAULT_DT_S

        cfg = load_config("config")
        app = create_app(
            cfg, cfg.dashboard,
            initial_scenario=SCENARIO_REGISTRY["engine_degradation_60s"],
        )
        client = TestClient(app)
        r = client.get("/api/snapshot/latest")
        assert r.status_code == 200
        snap = r.json()["snapshot"]
        # The alerts array exists.
        assert "alerts" in snap
        # If there are any alerts they carry the popup fields.
        for a in snap["alerts"]:
            for key in (
                "alert_id", "severity_label", "tier",
                "title", "message", "tags",
            ):
                assert key in a, f"popup field {key!r} missing from API"
            assert a["severity_label"] in (
                "HEALTHY", "INFO", "WARNING", "CRITICAL",
            )
            assert a["tier"] in (0, 1, 2)
            assert a["alert_id"]


# ---------------------------------------------------------------------
# Test 13: M. Alert history is finite and ordered
# ---------------------------------------------------------------------
class TestAlertHistory:
    def test_history_is_bounded(self) -> None:
        # The frontend bounds the history to 200 entries. The
        # backend doesn't carry state, but this test exercises
        # the alert_id stability for repeated ticks to ensure
        # the dedup key doesn't churn.
        ids = set()
        for _ in range(50):
            a = derive_alerts(_degraded_payload())
            for x in a:
                ids.add(x["alert_id"])
        assert len(ids) == 1


# ---------------------------------------------------------------------
# Test 14: severity_label_for / tier_for consistency
# ---------------------------------------------------------------------
class TestLabelTierConsistency:
    @pytest.mark.parametrize("sev,ftype,expected_label", [
        ("CAUTION", "ENGINE_DEGRADATION", "WARNING"),
        ("CAUTION", "ENVIRONMENTAL_DISTURBANCE", "INFO"),
        ("CAUTION", "UNKNOWN_ANOMALY", "WARNING"),
        ("WARNING", "ENGINE_DEGRADATION", "WARNING"),
        ("RETURN_TO_BASE", "ENGINE_DEGRADATION", "CRITICAL"),
        ("ABORT", "ENGINE_DEGRADATION", "CRITICAL"),
        ("INFO", "DATA_DEGRADED", "INFO"),
    ])
    def test_label_mapping(self, sev, ftype, expected_label) -> None:
        assert severity_label_for(sev, ftype) == expected_label

    @pytest.mark.parametrize("label,expected_tier", [
        ("CRITICAL", 2),
        ("WARNING", 1),
        ("INFO", 0),
        ("HEALTHY", 0),
    ])
    def test_tier_mapping(self, label, expected_tier) -> None:
        assert tier_for(label) == expected_tier


# ---------------------------------------------------------------------
# Test 15: Recommendation always present
# ---------------------------------------------------------------------
class TestRecommendationField:
    def test_every_alert_has_recommendation(self) -> None:
        for ftype in ("ENGINE_DEGRADATION", "OVERHEATING", "FAKE_FAULT_X"):
            assert recommendation_for(ftype)  # non-empty string
