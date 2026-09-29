"""PHASE 13 tests — real-time dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import (
    DashboardConfig,
    load_config,
    load_dashboard_config,
)
from backend.dashboard import (
    DEFAULT_DT_S,
    DEFAULT_SCENARIO_NAME,
    SCENARIO_REGISTRY,
    DashboardSnapshot,
    PipelineRunner,
    ScenarioSpec,
    create_app,
    get_scenario,
)
from backend.dashboard.snapshot import WIRE_KEYS
from backend.simulation import EngineState

pytestmark = pytest.mark.phase13

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
FRONTEND_INDEX = REPO_ROOT / "frontend" / "index.html"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cfg():
    return load_config(CONFIG_DIR)


def _dc(**overrides) -> DashboardConfig:
    """A small, fast dashboard config for tests."""
    base = {
        "tick_rate_hz": 50.0,
        "history_size": 20,
        "default_scenario": "engine_degradation_60s",
    }
    base.update(overrides)
    return DashboardConfig(**base)


# ---------------------------------------------------------------------
# TestDashboardConfig
# ---------------------------------------------------------------------
class TestDashboardConfig:
    def test_defaults_are_valid(self) -> None:
        dc = DashboardConfig()
        assert dc.host == "127.0.0.1"
        assert dc.port == 8000
        assert dc.tick_rate_hz == pytest.approx(10.0)
        assert dc.history_size == 600
        assert dc.default_scenario == DEFAULT_SCENARIO_NAME
        assert dc.static_dir == "frontend"
        assert dc.max_ws_clients == 8

    def test_load_dashboard_config_missing_file_returns_defaults(self, tmp_path) -> None:
        dc = load_dashboard_config(tmp_path)
        assert dc == DashboardConfig.defaults()

    def test_load_dashboard_config_validates_yaml(self, tmp_path) -> None:
        path = tmp_path / "dashboard.yaml"
        path.write_text(
            "dashboard:\n  port: 9000\n  tick_rate_hz: 25.0\n",
            encoding="utf-8",
        )
        dc = load_dashboard_config(tmp_path)
        assert dc.port == 9000
        assert dc.tick_rate_hz == pytest.approx(25.0)

    def test_dashboard_config_in_loaded_config(self) -> None:
        cfg = load_config()
        assert isinstance(cfg.dashboard, DashboardConfig)
        # dashboard.yaml exists in the repo so port is 8000 (matches default).
        assert cfg.dashboard.port == 8000


# ---------------------------------------------------------------------
# TestDashboardSnapshot
# ---------------------------------------------------------------------
class TestDashboardSnapshot:
    def test_snapshot_to_dict_shape(self) -> None:
        # Drive the pipeline for one tick to get a real snapshot.
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        for key in WIRE_KEYS:
            assert key in d, f"missing key in wire format: {key!r}"
        # Sub-shape spot-checks
        assert d["scenario_name"] == "healthy_60s"
        assert d["frame_status"] == "OK"
        assert d["tick_index"] == 0
        assert "risk" in d and "status" in d["risk"]
        assert "health" in d and "overall_score" in d["health"]
        assert "engine_state" in d and "rpm" in d["engine_state"]

    def test_snapshot_round_trip_is_json_safe(self) -> None:
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        # JSON round-trip must succeed and preserve top-level keys.
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert set(decoded.keys()) == set(WIRE_KEYS)

    def test_snapshot_handles_none_anomaly_and_classification(self) -> None:
        # The default scenarios do not enable the optional PHASE 9
        # classifier, so fault_classification must be None. anomaly is
        # always produced.
        runner = PipelineRunner(_cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5)
        snap = runner.tick()
        d = snap.to_dict()
        assert d["fault_classification"] is None
        assert d["anomaly"] is not None  # PHASE 8 always fires
        # Rest of payload is intact.
        assert d["risk"] is not None
        assert d["health"] is not None
        assert d["engine_state"] is not None


# ---------------------------------------------------------------------
# TestPipelineRunner
# ---------------------------------------------------------------------
class TestPipelineRunner:
    def test_tick_advances_clock(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=20
        )
        for _ in range(10):
            runner.tick()
        assert runner.tick_count == 10

    def test_tick_returns_snapshot(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5
        )
        snap = runner.tick()
        assert isinstance(snap, DashboardSnapshot)
        assert snap.scenario_name == "healthy_60s"
        assert isinstance(snap.engine_state, EngineState)

    def test_run_yields_expected_count(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=100
        )
        snaps = list(runner.run(duration_s=2.0))
        # 2.0 s / 0.1 s dt = 20 ticks
        assert len(snaps) == 20
        for s in snaps:
            assert isinstance(s, DashboardSnapshot)

    def test_history_buffer_caps_at_maxlen(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=10
        )
        for _ in range(30):
            runner.tick()
        assert len(runner.history) == 10
        assert runner.tick_count == 30

    def test_set_scenario_resets_pipeline(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["engine_degradation_60s"], history_size=10
        )
        for _ in range(5):
            runner.tick()
        assert runner.tick_count == 5
        assert len(runner.history) == 5
        runner.set_scenario(SCENARIO_REGISTRY["overheating_60s"])
        assert runner.scenario.name == "overheating_60s"
        assert runner.tick_count == 0
        assert len(runner.history) == 0

    def test_default_scenario_escalates(self) -> None:
        """The default scenario (engine_degradation_60s) must
        demonstrate escalation within a 60 s mission — the dashboard's
        first page load should never be boring."""
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["engine_degradation_60s"], history_size=200
        )
        statuses = set()
        for _ in runner.run(duration_s=60.0):
            statuses.add(_.risk.status.value)
        # At least one non-GO status must occur (proves the demo scenario escalates).
        non_go = statuses - {"GO"}
        assert non_go, f"default scenario produced only GO statuses: {statuses}"

    def test_scenario_spec_to_fault_scenario(self) -> None:
        # Spot-check: the engine_degradation scenario round-trips into a PHASE 7 FaultScenario.
        # Severity was bumped from 0.6 to 0.85 (with "step"
        # progression) so the diagnostic system picks up the
        # fault within the 60-s mission; the test follows
        # the new contract.
        spec = SCENARIO_REGISTRY["engine_degradation_60s"]
        fs = spec.to_fault_scenario()
        from backend.faults import FaultClass
        assert fs.fault_class is FaultClass.ENGINE_DEGRADATION
        assert fs.severity == pytest.approx(0.85)
        assert fs.onset_time_s == pytest.approx(10.0)
        assert fs.duration_s == pytest.approx(40.0)

    def test_sensor_fault_scenario_has_channel(self) -> None:
        spec = SCENARIO_REGISTRY["sensor_fault_60s"]
        fs = spec.to_fault_scenario()
        from backend.sensors import NoiseMode
        assert fs.target_channel == "rpm"
        assert fs.target_sensor_mode is NoiseMode.STUCK

    def test_scenarios_to_dict_round_trip(self) -> None:
        spec = SCENARIO_REGISTRY["vibration_anomaly_60s"]
        d = spec.to_dict()
        assert d["name"] == "vibration_anomaly_60s"
        assert d["fault_class"] == "VIBRATION_ENGINE_ANOMALY"
        assert d["run_classifier"] is False

    def test_get_scenario_returns_none_for_unknown(self) -> None:
        assert get_scenario("nope") is None
        assert get_scenario("healthy_60s") is SCENARIO_REGISTRY["healthy_60s"]

    def test_dt_s_default(self) -> None:
        runner = PipelineRunner(
            _cfg(), SCENARIO_REGISTRY["healthy_60s"], history_size=5
        )
        assert runner.dt_s == DEFAULT_DT_S
        assert runner.dt_s == pytest.approx(0.1)


# ---------------------------------------------------------------------
# TestFastAPIApp
# ---------------------------------------------------------------------
class TestFastAPIApp:
    def test_get_root_serves_html(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 200
            assert "text/html" in r.headers["content-type"]
            # PHASE 19 — control-room layout markers.
            assert 'id="top-strip"' in r.text
            assert 'id="left-col"' in r.text
            assert 'id="center-col"' in r.text
            assert 'id="right-col"' in r.text
            assert 'id="bottom-strip"' in r.text
            assert 'id="scenario-select"' in r.text

    def test_snapshot_latest_returns_json(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/api/snapshot/latest")
            assert r.status_code == 200
            j = r.json()
            assert "snapshot" in j
            assert j["snapshot"] is not None
            for key in WIRE_KEYS:
                assert key in j["snapshot"]

    def test_history_endpoint_respects_limit(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/api/history?limit=5")
            assert r.status_code == 200
            j = r.json()
            assert "snapshots" in j
            assert len(j["snapshots"]) <= 5

    def test_scenario_get_returns_name(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/api/scenario")
            assert r.status_code == 200
            j = r.json()
            assert j["name"] == "engine_degradation_60s"
            assert j["spec"]["fault_class"] == "ENGINE_DEGRADATION"

    def test_scenario_post_switches(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.post("/api/scenario", json={"name": "healthy_60s"})
            assert r.status_code == 200
            j = r.json()
            assert j["name"] == "healthy_60s"
            # Confirm the next GET reflects the change.
            r2 = client.get("/api/scenario")
            assert r2.json()["name"] == "healthy_60s"

    def test_scenario_post_unknown_returns_404(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.post("/api/scenario", json={"name": "nope"})
            assert r.status_code == 404

    def test_scenarios_list_endpoint(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/api/scenarios")
            assert r.status_code == 200
            j = r.json()
            assert "scenarios" in j
            assert len(j["scenarios"]) == 8
            names = [s["name"] for s in j["scenarios"]]
            assert "engine_degradation_60s" in names
            assert "healthy_60s" in names

    def test_health_endpoint(self) -> None:
        app = create_app(_cfg(), _dc())
        with TestClient(app) as client:
            r = client.get("/api/health")
            assert r.status_code == 200
            j = r.json()
            assert j["ok"] is True
            assert j["scenario"] == "engine_degradation_60s"
            assert j["tick_count"] >= 1


# ---------------------------------------------------------------------
# TestWebSocket
# ---------------------------------------------------------------------
class TestWebSocket:
    def test_websocket_receives_snapshot(self) -> None:
        # Fast cadence so the pump task delivers quickly.
        app = create_app(_cfg(), _dc(tick_rate_hz=100.0))
        with TestClient(app) as client:
            with client.websocket_connect("/api/stream") as ws:
                msg = ws.receive_json()
                for key in WIRE_KEYS:
                    assert key in msg
                assert msg["scenario_name"] in SCENARIO_REGISTRY


# ---------------------------------------------------------------------
# TestFrontend
# ---------------------------------------------------------------------
class TestFrontend:
    def test_index_html_exists_and_references_chartjs(self) -> None:
        assert FRONTEND_INDEX.exists(), f"missing: {FRONTEND_INDEX}"
        content = FRONTEND_INDEX.read_text(encoding="utf-8")
        # Chart.js pinned CDN (PHASE 19 keeps the same version).
        assert "chart.js@" in content
        # The page must not contain a build pipeline or external script aside
        # from the Chart.js CDN.
        assert "https://cdn.jsdelivr.net" in content
        # The HTML must reference the bundled dashboard.js (PHASE 19 split
        # the inline <script> into a separate asset).
        assert "/assets/dashboard.js" in content
        # The dashboard asset layer must reference the same backend APIs.
        js = (FRONTEND_INDEX.parent / "assets" / "dashboard.js").read_text(
            encoding="utf-8"
        )
        for path in ("/api/snapshot/latest", "/api/stream", "/api/scenarios"):
            assert path in js, f"dashboard.js missing {path}"

    def test_index_html_has_all_sections(self) -> None:
        assert FRONTEND_INDEX.exists()
        content = FRONTEND_INDEX.read_text(encoding="utf-8")
        # PHASE 19 — control-room dashboard zones + scenario selector.
        for section_id in [
            "top-strip",
            "left-col",
            "center-col",
            "right-col",
            "bottom-strip",
            "scenario-select",
        ]:
            assert f'id="{section_id}"' in content, f"missing id={section_id}"
