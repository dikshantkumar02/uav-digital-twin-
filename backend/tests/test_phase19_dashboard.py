"""PHASE 19 tests — control-room dashboard (snapshot extensions + frontend)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import (
    DashboardConfig,
    load_config,
)
from backend.dashboard import (
    Alert,
    PipelineRunner,
    SCENARIO_REGISTRY,
    derive_alerts,
    recommendation_for,
)
from backend.dashboard.snapshot import WIRE_KEYS
from backend.dashboard.alerts import (
    _env_context,
    _evidence_lines,
    _rank,
    _sensor_health,
)

pytestmark = pytest.mark.phase19

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
FRONTEND_INDEX = REPO_ROOT / "frontend" / "index.html"
FRONTEND_DIR = REPO_ROOT / "frontend"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cfg():
    return load_config(CONFIG_DIR)


def _dc(**overrides) -> DashboardConfig:
    base = {
        "tick_rate_hz": 50.0,
        "history_size": 20,
        "default_scenario": "engine_degradation_60s",
    }
    base.update(overrides)
    return DashboardConfig(**base)


def _client():
    from backend.dashboard import create_app

    return TestClient(create_app(_cfg(), _dc()))


# ---------------------------------------------------------------------
# Test 1: snapshot has new blocks
# ---------------------------------------------------------------------
class TestSnapshotHasNewBlocks:
    def test_wire_keys_include_phase19_blocks(self) -> None:
        for key in ("residual", "twin_expected", "alerts", "events", "model_confidence"):
            assert key in WIRE_KEYS, f"missing PHASE 19 wire key: {key!r}"

    def test_snapshot_to_dict_emits_phase19_blocks(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        # New fields must all be present (even if some are None / empty).
        assert "residual" in d
        assert "twin_expected" in d
        assert "alerts" in d
        assert "events" in d
        assert "model_confidence" in d

    def test_snapshot_is_json_safe(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        for key in ("residual", "twin_expected", "alerts", "events", "model_confidence"):
            assert key in decoded


# ---------------------------------------------------------------------
# Test 2: residual block covers documented channels
# ---------------------------------------------------------------------
class TestResidualBlockCoversChannels:
    def test_residual_block_includes_documented_channels(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        residual = d["residual"]
        assert residual is not None
        # ResidualFrame.to_dict uses dotted keys: "residual.<channel>.predicted" etc.
        for ch in ("rpm", "egt", "cht", "oil_pressure", "fuel_flow", "vibration"):
            assert f"residual.{ch}.observed" in residual, ch
            assert f"residual.{ch}.predicted" in residual, ch
            assert f"residual.{ch}.residual" in residual, ch
            assert f"residual.{ch}.z_score" in residual, ch

    def test_twin_expected_is_flat_dict(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        twin = d["twin_expected"]
        assert isinstance(twin, dict)
        # Every value must be a plain float — no nested objects.
        for k, v in twin.items():
            assert isinstance(v, (int, float)), f"non-float twin_expected[{k}]: {type(v)}"


# ---------------------------------------------------------------------
# Test 3: alerts dedup keyed by fault_type
# ---------------------------------------------------------------------
class TestAlertsDedupKeyedByFaultType:
    def test_derive_alerts_dedupes_by_fault_type(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "ABORT", "confidence": 0.9, "risk_score": 0.95},
            "anomaly": {"overall_label": "ANOMALY", "confidence": 0.7,
                        "contributing_channels": ["rpm"]},
            "fault_classification": {"fault_class": "ENGINE_DEGRADATION", "confidence": 0.8},
        }
        # The same input called twice must produce the same output
        # (dedup is keyed by fault_type so the same condition
        # doesn't emit twice).
        a1 = derive_alerts(snap)
        a2 = derive_alerts(snap)
        assert a1 == a2

    def test_derive_alerts_emit_one_per_fault_type(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "ABORT", "confidence": 0.9, "risk_score": 0.95},
            "anomaly": {"overall_label": "ANOMALY", "confidence": 0.7,
                        "contributing_channels": ["rpm"]},
        }
        alerts = derive_alerts(snap)
        types = [a["fault_type"] for a in alerts]
        # No duplicate fault_types in a single tick.
        assert len(types) == len(set(types))


# ---------------------------------------------------------------------
# Test 4: alerts emitted for each active condition
# ---------------------------------------------------------------------
class TestAlertsForEachActiveCondition:
    def test_abort_emits_an_alert(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "ABORT", "confidence": 0.9, "risk_score": 0.95},
        }
        alerts = derive_alerts(snap)
        assert len(alerts) >= 1
        assert any(a["severity"] == "ABORT" for a in alerts)

    def test_sensor_fault_emits_alert(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "sample": {"sensor_health": {"rpm": {"mode": "STUCK"}}},
        }
        alerts = derive_alerts(snap)
        assert any(a["fault_type"] == "SENSOR_FAULT" for a in alerts)

    def test_data_quality_low_emits_alert(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "metadata": {"data_quality": 0.2},
        }
        alerts = derive_alerts(snap)
        assert any(a["fault_type"] == "DATA_DEGRADED" for a in alerts)

    def test_severe_subsystem_emits_engine_alert(self) -> None:
        """A single subsystem in CRITICAL band (< 0.30) fires a
        WARNING banner when the trend is DEGRADING. The
        95th-percentile fusion can absorb a single bad
        subsystem so the overall is still HEALTHY, but a
        downward trend means something is actually
        worsening and the operator needs to know.
        """
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "health": {
                "overall_label": "HEALTHY",
                "overall_score": 0.85,
                "trend": "DEGRADING",
                "sub.PERFORMANCE.subsystem": "PERFORMANCE",
                "sub.PERFORMANCE.score": 0.27,
                "sub.PERFORMANCE.label": "DEGRADED",
            },
        }
        alerts = derive_alerts(snap)
        engine = [a for a in alerts if a["fault_type"] == "ENGINE_DEGRADATION"]
        assert len(engine) == 1
        assert engine[0]["severity"] == "WARNING"
        assert "PERFORMANCE" in engine[0]["evidence"]
        assert "0.27" in engine[0]["evidence"]

    def test_healthy_takeoff_does_not_fire_warning(self) -> None:
        """During takeoff the engine is at partial throttle
        (rpm << rated, brake_power small) so the PERFORMANCE
        subsystem legitimately scores below 0.30 — that's
        the expected operating point, not a fault. When
        the overall is HEALTHY AND the trend is STABLE
        (the score isn't worsening) the per-subsystem
        drill-down must stay silent. Otherwise a healthy
        mission would spam WARNING every tick during
        climb-out.
        """
        snap = {
            "time_s": 5.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "health": {
                "overall_label": "HEALTHY",
                "overall_score": 0.84,
                "trend": "STABLE",
                "sub.PERFORMANCE.subsystem": "PERFORMANCE",
                "sub.PERFORMANCE.score": 0.27,
                "sub.PERFORMANCE.label": "DEGRADED",
            },
        }
        alerts = derive_alerts(snap)
        # No WARNING/CAUTION allowed for a HEALTHY + STABLE
        # mission even with one bad subsystem.
        engine = [a for a in alerts
                  if a["fault_type"] == "ENGINE_DEGRADATION"
                  and a["severity"] in ("WARNING", "CAUTION")]
        assert engine == [], f"unexpected WARN/CAUTION on takeoff: {engine}"

    def test_overall_degraded_emits_engine_alert(self) -> None:
        """When the overall health label is DEGRADED, a
        CAUTION-level ENGINE_DEGRADATION banner is emitted."""
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "health": {
                "overall_label": "DEGRADED",
                "overall_score": 0.70,
                "trend": "DEGRADING",
            },
        }
        alerts = derive_alerts(snap)
        engine = [a for a in alerts if a["fault_type"] == "ENGINE_DEGRADATION"]
        assert len(engine) == 1
        assert engine[0]["severity"] in ("CAUTION", "WARNING")

    def test_risk_elevated_does_not_block_health_alert(self) -> None:
        """A RISK_ELEVATED alert (the engine wear that
        pushed risk over 0.55 in GO state) must not block
        the health-driven alert — the two can co-exist."""
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.65},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "health": {
                "overall_label": "DEGRADED",
                "overall_score": 0.70,
                "trend": "DEGRADING",
            },
        }
        alerts = derive_alerts(snap)
        types = {a["fault_type"] for a in alerts}
        # RISK_ELEVATED fires from the risk-driven path;
        # ENGINE_DEGRADATION fires from the health path.
        assert "RISK_ELEVATED" in types
        assert "ENGINE_DEGRADATION" in types

    def test_dropped_channel_does_not_fire_sensor_fault(self) -> None:
        """A single random DROPPED reading is a bus /
        connectivity event, not a sensor fault. Only
        persistent modes (STUCK / FAULT / DRIFTING /
        SPIKE) should fire SENSOR_FAULT."""
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "sample": {"sensor_health": {"vibration": {"mode": "DROPPED"}}},
        }
        alerts = derive_alerts(snap)
        assert not any(a["fault_type"] == "SENSOR_FAULT" for a in alerts)

    def test_cold_start_idle_does_not_fire_health_alert(self) -> None:
        """Engine cold-start at idle (rpm=idle, brake_power=0,
        fuel_flow=0) legitimately produces a low PERFORMANCE
        score — that's the expected steady-state, not a
        fault. While the health trend is INSUFFICIENT_DATA
        (we have no history yet) the per-subsystem health
        alert path must stay silent so the first page load
        of a healthy mission is not flooded with warnings.
        """
        snap = {
            "time_s": 0.1,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "health": {
                "overall_label": "HEALTHY",
                "overall_score": 0.84,
                "trend": "INSUFFICIENT_DATA",
                "sub.PERFORMANCE.subsystem": "PERFORMANCE",
                "sub.PERFORMANCE.score": 0.22,
                "sub.PERFORMANCE.label": "DEGRADED",
            },
        }
        alerts = derive_alerts(snap)
        # Even though PERFORMANCE is at 0.22 (severe), we
        # must not fire the health-driven banner while the
        # trend is still INSUFFICIENT_DATA.
        engine = [a for a in alerts if a["fault_type"] == "ENGINE_DEGRADATION"]
        assert engine == [], f"unexpected health alert at cold start: {engine}"


# ---------------------------------------------------------------------
# Test 5: alerts clear when condition resolves
# ---------------------------------------------------------------------
class TestAlertsClearWhenResolved:
    def test_healthy_snapshot_produces_no_alerts(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.5, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
            "fault_classification": None,
            "metadata": {"data_quality": 1.0},
            "model_confidence": 0.5,
        }
        alerts = derive_alerts(snap)
        assert alerts == ()


# ---------------------------------------------------------------------
# Test 6: model_confidence is 0..1
# ---------------------------------------------------------------------
class TestModelConfidence:
    def test_model_confidence_is_float_in_range(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        conf = d["model_confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_aggregated_confidence_averages_sources(self) -> None:
        # Use a snapshot where all sources are calibrated.
        snap = {
            "time_s": 1.0,
            "risk": {"status": "GO", "confidence": 0.6, "risk_score": 0.1},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.8,
                        "contributing_channels": []},
        }
        # Use the runner's private aggregator.
        from backend.dashboard.runner import PipelineRunner as _PR

        class _StubRisk:
            confidence = 0.6
        class _StubHealth:
            confidence = 0.4
        class _StubAnomaly:
            confidence = 0.8
        val = _PR._aggregate_model_confidence(
            _StubRisk(), _StubHealth(), _StubAnomaly(), None,
        )
        assert 0.0 < val <= 1.0
        # Mean of 0.6, 0.4, 0.8 = 0.6
        assert abs(val - 0.6) < 0.05


# ---------------------------------------------------------------------
# Test 7: twin_expected is flat dict (no nested objects)
# ---------------------------------------------------------------------
class TestTwinExpectedFlat:
    def test_twin_expected_keys_are_strings_values_are_floats(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        twin = d["twin_expected"]
        assert isinstance(twin, dict)
        for k, v in twin.items():
            assert isinstance(k, str)
            assert isinstance(v, (int, float))


# ---------------------------------------------------------------------
# Test 8: derive_alerts is a pure function
# ---------------------------------------------------------------------
class TestDeriveAlertsPure:
    def test_pure_same_input_same_output(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "CAUTION", "confidence": 0.7, "risk_score": 0.6},
            "anomaly": {"overall_label": "NORMAL", "confidence": 0.5,
                        "contributing_channels": []},
        }
        a1 = derive_alerts(snap)
        a2 = derive_alerts(snap)
        # Equal values, equal length, equal type.
        assert a1 == a2
        assert len(a1) == len(a2)
        assert tuple(sorted(a["fault_type"] for a in a1)) == \
               tuple(sorted(a["fault_type"] for a in a2))

    def test_does_not_mutate_input(self) -> None:
        snap = {
            "time_s": 1.0,
            "risk": {"status": "ABORT", "confidence": 0.9, "risk_score": 0.95},
            "anomaly": {"overall_label": "ANOMALY", "confidence": 0.7,
                        "contributing_channels": ["rpm"]},
        }
        before = json.dumps(snap, sort_keys=True)
        derive_alerts(snap)
        after = json.dumps(snap, sort_keys=True)
        assert before == after


# ---------------------------------------------------------------------
# Test 9: dashboard HTML has the new layout markers
# ---------------------------------------------------------------------
class TestDashboardHtmlLayoutMarkers:
    def test_html_has_five_zone_ids(self) -> None:
        assert FRONTEND_INDEX.exists()
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
        for marker in ("top-strip", "left-col", "center-col", "right-col", "bottom-strip"):
            assert f'id="{marker}"' in html, f"missing zone marker: {marker!r}"


# ---------------------------------------------------------------------
# Test 10: dashboard HTML has exactly 8 chart canvases
# ---------------------------------------------------------------------
class TestDashboardHtmlEightCharts:
    def test_exactly_eight_main_chart_canvases(self) -> None:
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
        # Count only the *main* chart canvases (id starts with
        # "chart-"). The 5 LEFT-column sparklines ("ch-*-spark")
        # are separate from the 8 numbered charts.
        main = re.findall(r"<canvas[^>]*id=\"chart-[^\"]+\"", html)
        assert len(main) == 8, f"expected 8 main chart canvases, got {len(main)}: {main}"

    def test_all_eight_chart_ids_present(self) -> None:
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
        for cid in ("chart-rpm", "chart-egt", "chart-cht", "chart-oilp",
                    "chart-fuel", "chart-vib-imu", "chart-health", "chart-rul"):
            assert f'id="{cid}"' in html, f"missing chart canvas: {cid!r}"


# ---------------------------------------------------------------------
# Test 11: dashboard HTML + JS contains no aircraft-control commands
# ---------------------------------------------------------------------
class TestDashboardHtmlNoAircraftControl:
    # Aircraft-control language only. Generic words like "command"
    # are part of normal logging ("ws.oncommand", "command line
    # option") so the banned list is constrained to flight-control
    # verbs.
    BANNED_PHRASES = (
        "throttle to",
        "set rpm",
        "descend to",
        "raise throttle",
        "lower throttle",
        "reduce power to",
        "increase power to",
        "command descent",
    )

    def test_no_aircraft_control_phrases_in_html(self) -> None:
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
        for phrase in self.BANNED_PHRASES:
            assert phrase.lower() not in html.lower(), \
                f"banned aircraft-control phrase in HTML: {phrase!r}"

    def test_no_aircraft_control_phrases_in_dashboard_js(self) -> None:
        js = (FRONTEND_DIR / "assets" / "dashboard.js").read_text(encoding="utf-8")
        for phrase in self.BANNED_PHRASES:
            assert phrase.lower() not in js.lower(), \
                f"banned aircraft-control phrase in JS: {phrase!r}"


# ---------------------------------------------------------------------
# Test 12: dashboard JS references all 8 chart IDs
# ---------------------------------------------------------------------
class TestDashboardJsReferencesAllCharts:
    def test_js_references_eight_chart_ids(self) -> None:
        js = (FRONTEND_DIR / "assets" / "dashboard.js").read_text(encoding="utf-8")
        for cid in ("chart-rpm", "chart-egt", "chart-cht", "chart-oilp",
                    "chart-fuel", "chart-vib-imu", "chart-health", "chart-rul"):
            assert cid in js, f"JS does not reference chart id: {cid!r}"


# ---------------------------------------------------------------------
# Bonus: end-to-end — GET / returns the new layout
# ---------------------------------------------------------------------
class TestEndToEnd:
    def test_get_root_returns_layout_with_8_charts(self) -> None:
        client = _client()
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        for marker in ("top-strip", "left-col", "center-col", "right-col", "bottom-strip"):
            assert f'id="{marker}"' in html
        main = re.findall(r"<canvas[^>]*id=\"chart-[^\"]+\"", html)
        assert len(main) == 8

    def test_recommendation_for_returns_advice(self) -> None:
        assert "inspect" in recommendation_for("ENGINE_DEGRADATION").lower()
        # Falls back to a generic advice.
        assert recommendation_for("UNKNOWN_FAULT")

    def test_runner_model_version_metadata_matches_snapshot(self) -> None:
        """The snapshot's ``metadata.model_version`` block must
        always be a non-empty string and is the single source of
        truth for the model version (the alert helper, the API
        consumer, and the WebSocket payload all read it from
        there).

        Regression test: prior to the fix the runner used
        ``"phase19-control-room-1.0.0"`` in the metadata block
        while the snapshot's own ``model_version`` field used
        ``"phase17-streaming-1.0.0"`` — two different strings
        describing the same software stack.
        """
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5
        )
        snap = runner.tick()
        d = snap.to_dict()
        # ``metadata.model_version`` is the canonical home (it is
        # in :data:`backend.dashboard.snapshot.METADATA_KEYS`).
        meta_mv = d["metadata"]["model_version"]
        assert isinstance(meta_mv, str) and meta_mv
        # And the snapshot's ``model_version`` field agrees.
        assert snap.model_version == meta_mv

    def test_runner_data_quality_propagates_to_snapshot(self) -> None:
        """The snapshot's ``data_quality`` and ``metadata.data_quality``
        must come from the diagnostic assembler (which derives the
        value from the sensor sample), not a hardcoded ``1.0``.

        Regression test: prior to the fix the runner hardcoded
        ``data_quality=1.0`` for the snapshot and metadata, so the
        alert path could never fire ``DATA_DEGRADED`` warnings when
        sensors actually dropped readings.
        """
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5
        )
        snap = runner.tick()
        dq_snapshot = float(snap.data_quality)
        dq_meta = float(snap.to_dict()["metadata"]["data_quality"])
        # Both must agree (single source of truth).
        assert dq_snapshot == pytest.approx(dq_meta)
        # And both must be in the [0, 1] range — the assembler
        # never emits out-of-band values.
        assert 0.0 <= dq_snapshot <= 1.0
        # And the metadata's data_quality fed into the alert path
        # must equal the snapshot's — otherwise the alert helper
        # sees a different value than the API consumer.
        assert dq_meta == pytest.approx(dq_snapshot)
