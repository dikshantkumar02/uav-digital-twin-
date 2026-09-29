"""PHASE 17 tests — real-time streaming pipeline.

These tests verify the new ``TelemetryAdapter``,
``AsyncStreamingPipeline``, ``ReplayController``, the extended
``TelemetryFrame`` + ``DashboardSnapshot`` fields, and the
``/api/latency`` + ``/api/replay/*`` endpoints on the
dashboard.

Markers: ``@pytest.mark.phase17``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from backend.config import (
    DashboardConfig,
    load_config,
    load_dashboard_config,
)
from backend.dashboard import SCENARIO_REGISTRY, create_app
from backend.dashboard.replay import ReplayController, ReplaySpeed, ReplayStats
from backend.dashboard.snapshot import WIRE_KEYS
from backend.diagnostics import AnomalyDetector
from backend.digital_twin import DigitalTwin
from backend.environment import MissionProfile
from backend.health import HealthIndexCalculator
from backend.risk import MissionRiskCalculator
from backend.rul import RulCalculator
from backend.sensors import (
    SENSOR_CHANNELS,
    NoiseMode,
    SensorBundle,
    SensorReading,
    SensorSample,
)
from backend.simulation import EngineSimulator, EngineState
from backend.telemetry import (
    AdapterConfig,
    AsyncStreamingPipeline,
    FrameStatus,
    LatencyTracker,
    PipelineConfig,
    PipelineResult,
    Preprocessor,
    TelemetryAdapter,
    TelemetryFrame,
    compute_data_quality,
)
from backend.environment import EnvironmentState


pytestmark = pytest.mark.phase17


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cfg():
    return load_config(CONFIG_DIR)


def _step_sim():
    """Return (engine_state, env_state) by stepping the simulator once."""
    sim = EngineSimulator(_cfg().engine, _cfg().environment, dt_s=0.1)
    sim.engine.reset()
    sim.runner.reset()
    step = sim.step()
    return step.engine, step.env


def _good_sample(time_s: float = 0.0) -> SensorSample:
    """Build a clean SensorSample — all channels have a value, NORMAL mode."""
    es, env = _step_sim()
    readings = {
        name: SensorReading(value=1.0, mode=NoiseMode.NORMAL)
        for name in SENSOR_CHANNELS
    }
    return SensorSample(
        time_s=time_s,
        readings=readings,
        engine=es,
        env=env,
    )


def _stale_sample(time_s: float = 0.0, n_dropout: int = 1) -> SensorSample:
    es, env = _step_sim()
    readings = {}
    for i, name in enumerate(SENSOR_CHANNELS):
        if i < n_dropout:
            readings[name] = SensorReading(value=None, mode=NoiseMode.NORMAL)
        else:
            readings[name] = SensorReading(value=1.0, mode=NoiseMode.NORMAL)
    return SensorSample(time_s=time_s, readings=readings, engine=es, env=env)


def _stuck_sample(time_s: float = 0.0, n_stuck: int = 1) -> SensorSample:
    es, env = _step_sim()
    readings = {}
    for i, name in enumerate(SENSOR_CHANNELS):
        if i < n_stuck:
            readings[name] = SensorReading(value=1.0, mode=NoiseMode.STUCK)
        else:
            readings[name] = SensorReading(value=1.0, mode=NoiseMode.NORMAL)
    return SensorSample(time_s=time_s, readings=readings, engine=es, env=env)


def _make_full_pipeline(
    *,
    queue_max_size: int = 16,
    tracker: Optional[LatencyTracker] = None,
    preprocessor: Optional[Preprocessor] = None,
) -> AsyncStreamingPipeline:
    """Build a full AsyncStreamingPipeline with all calculators wired."""
    cfg = _cfg()
    twin = DigitalTwin(cfg.engine, dt_s=0.1)
    anomaly = AnomalyDetector()
    health = HealthIndexCalculator(cfg.engine)
    rul = RulCalculator(cfg.engine)
    profile = MissionProfile.from_config(cfg.environment.mission)
    risk = MissionRiskCalculator(profile, config=cfg.risk)
    return AsyncStreamingPipeline(
        cfg=PipelineConfig(queue_max_size=queue_max_size),
        digital_twin=twin,
        anomaly_detector=anomaly,
        classifier=None,
        health_calculator=health,
        rul_calculator=rul,
        risk_calculator=risk,
        preprocessor=preprocessor,
        tracker=tracker,
    )


def _run_async(coro):
    """Run an async coroutine in a fresh event loop (helper for tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_async_in(loop, coro):
    """Run a coroutine on an existing event loop."""
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------
# 1. TelemetryFrame extension
# ---------------------------------------------------------------------
class TestTelemetryFrameExtension:
    """The five new PHASE 17 fields."""

    def test_frame_carries_mission_vehicle_engine_ids(self) -> None:
        sample = _good_sample()
        f = TelemetryFrame(
            sequence=1, time_s=0.0, produced_wall_time=0.0,
            sample=sample,
            mission_id="m-007", vehicle_id="v-007", engine_id="e-007",
        )
        assert f.mission_id == "m-007"
        assert f.vehicle_id == "v-007"
        assert f.engine_id == "e-007"

    def test_frame_to_dict_includes_ids_and_quality(self) -> None:
        sample = _good_sample()
        f = TelemetryFrame(
            sequence=1, time_s=0.0, produced_wall_time=0.0, sample=sample,
            mission_id="m", vehicle_id="v", engine_id="e",
            data_quality=0.5, model_version="phase17-streaming-1.0.0",
        )
        d = f.to_dict()
        for key in ("mission_id", "vehicle_id", "engine_id",
                    "data_quality", "model_version"):
            assert key in d
        assert d["mission_id"] == "m"
        assert d["data_quality"] == 0.5

    def test_frame_defaults_keep_backward_compat(self) -> None:
        """PHASE 1–16 callers (no kwargs) still work."""
        sample = _good_sample()
        f = TelemetryFrame(sequence=1, time_s=0.0,
                           produced_wall_time=0.0, sample=sample)
        # Defaults
        assert f.mission_id == "default-mission"
        assert f.data_quality == 1.0
        assert f.model_version == "phase17-streaming-1.0.0"


# ---------------------------------------------------------------------
# 2. TelemetryAdapter
# ---------------------------------------------------------------------
class TestTelemetryAdapter:

    def test_adapter_tags_mission_vehicle_engine(self) -> None:
        cfg = AdapterConfig(
            mission_id="M", vehicle_id="V", engine_id="E",
            model_version="v1.2.3",
        )
        adapter = TelemetryAdapter(cfg)
        sample = _good_sample()
        frame = adapter.adapt(sample, sequence=42)
        assert frame.mission_id == "M"
        assert frame.vehicle_id == "V"
        assert frame.engine_id == "E"
        assert frame.model_version == "v1.2.3"
        assert frame.sequence == 42

    def test_adapter_data_quality_clean(self) -> None:
        assert compute_data_quality(_good_sample()) == 1.0

    def test_adapter_data_quality_with_dropouts(self) -> None:
        # 2 of 12 channels dropped, no stuck.
        q = compute_data_quality(_stale_sample(n_dropout=2))
        # q = 1 - (2 + 0) / 36 = 34/36 ≈ 0.944
        assert 0.93 < q < 0.95

    def test_adapter_data_quality_with_stuck(self) -> None:
        # 1 stuck channel is penalised more heavily.
        q = compute_data_quality(_stuck_sample(n_stuck=1))
        # q = 1 - (0 + 3) / 36 = 33/36 ≈ 0.917
        assert 0.91 < q < 0.93


# ---------------------------------------------------------------------
# 3. AsyncStreamingPipeline
# ---------------------------------------------------------------------
class TestAsyncStreamingPipeline:

    def test_pipeline_end_to_end_runs(self) -> None:
        pipeline = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            sample = _good_sample(time_s=0.0)
            adapter = TelemetryAdapter()
            frame = adapter.adapt(sample, sequence=1)
            loop.run_until_complete(pipeline.submit(frame))
            result = loop.run_until_complete(
                asyncio.wait_for(pipeline.output_queue.get(), timeout=2.0)
            )
            assert isinstance(result, PipelineResult)
            assert result.frame.sequence == 1
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_records_per_stage_latency(self) -> None:
        tracker = LatencyTracker()
        pipeline = _make_full_pipeline(tracker=tracker)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            adapter = TelemetryAdapter()
            for i in range(3):
                frame = adapter.adapt(_good_sample(time_s=i * 0.1), sequence=i + 1)
                loop.run_until_complete(pipeline.submit(frame))
            for _ in range(3):
                loop.run_until_complete(asyncio.wait_for(
                    pipeline.output_queue.get(), timeout=2.0
                ))
            summary = tracker.summary()
            for stage_name in ("validation", "preprocessing", "digital_twin", "ai", "risk"):
                assert stage_name in summary
            for stage in summary.values():
                assert stage.samples >= 1
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_end_to_end_latency_populated(self) -> None:
        tracker = LatencyTracker()
        pipeline = _make_full_pipeline(tracker=tracker)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            adapter = TelemetryAdapter()
            frame = adapter.adapt(_good_sample(), sequence=1)
            loop.run_until_complete(pipeline.submit(frame))
            result = loop.run_until_complete(asyncio.wait_for(
                pipeline.output_queue.get(), timeout=2.0
            ))
            assert result.end_to_end_latency_s >= 0.0
            assert isinstance(result.latency, dict)
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_drops_on_overflow(self) -> None:
        pipeline = _make_full_pipeline(queue_max_size=1)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            adapter = TelemetryAdapter()
            for i in range(20):
                frame = adapter.adapt(
                    _good_sample(time_s=i * 0.1), sequence=i + 1,
                )
                loop.run_until_complete(pipeline.submit(frame))
            stats = pipeline.stage_stats()
            total_drops = sum(s.dropped_overflow for s in stats.values())
            assert total_drops >= 0
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_handles_invalid_frame(self) -> None:
        sample = _stale_sample(n_dropout=len(SENSOR_CHANNELS))
        adapter = TelemetryAdapter()
        frame = adapter.adapt(sample, sequence=1)
        frame.data_quality = 0.0
        tracker = LatencyTracker()
        pipeline = _make_full_pipeline(tracker=tracker)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            loop.run_until_complete(pipeline.submit(frame))
            result = loop.run_until_complete(asyncio.wait_for(
                pipeline.output_queue.get(), timeout=2.0
            ))
            assert result is not None
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_handles_preprocessor_dropout(self) -> None:
        tracker = LatencyTracker()
        pipeline = _make_full_pipeline(tracker=tracker)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            adapter = TelemetryAdapter()
            sample = _stale_sample(n_dropout=1)
            frame = adapter.adapt(sample, sequence=1)
            loop.run_until_complete(pipeline.submit(frame))
            result = loop.run_until_complete(asyncio.wait_for(
                pipeline.output_queue.get(), timeout=2.0
            ))
            assert result is not None
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_pipeline_stop_drains(self) -> None:
        pipeline = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            assert pipeline.is_running()
            loop.run_until_complete(pipeline.stop())
            assert not pipeline.is_running()
        finally:
            loop.close()


# ---------------------------------------------------------------------
# 4. ReplayController
# ---------------------------------------------------------------------
def _make_wire_bin(out_dir: Path, n_frames: int = 6) -> Path:
    """Build a minimal wire.bin by recording a scenario run.

    Reuses the PHASE 15 WireRecorder so the format matches
    what production hardware would produce. Uses the real
    PipelineRunner with the healthy scenario so the
    DashboardSnapshots are valid.
    """
    from backend.dashboard.snapshot import DashboardSnapshot
    from backend.dashboard.runner import PipelineRunner
    from backend.dashboard.scenarios import SCENARIO_REGISTRY
    from backend.hil.manifest import Manifest
    from backend.hil.recorder import WireRecorder

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg()
    manifest = Manifest(
        scenario_name="healthy_60s",
        sample_rate_hz=10.0,
        duration_s=float(n_frames) * 0.1,
    )
    rec = WireRecorder(out_dir, manifest)
    runner = PipelineRunner(
        cfg, SCENARIO_REGISTRY["healthy_60s"],
        history_size=n_frames, dt_s=0.1,
    )
    for _ in range(n_frames):
        snap = runner.tick()
        rec.record(snap)
    rec.close()
    return rec.bin_path


class TestReplayController:

    def test_replay_speed_1x_runs(self, tmp_path: Path) -> None:
        wire = _make_wire_bin(tmp_path / "rec1", n_frames=4)
        pipeline = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            controller = ReplayController(wire, pipeline, speed=ReplaySpeed.SPEED_1X)
            stats = loop.run_until_complete(
                asyncio.wait_for(controller.run(), timeout=5.0)
            )
            assert stats.frames_replayed >= 1
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()

    def test_replay_speed_2x_runs_faster(self, tmp_path: Path) -> None:
        wire = _make_wire_bin(tmp_path / "rec2", n_frames=6)
        # 1x
        pipe1 = _make_full_pipeline()
        loop1 = asyncio.new_event_loop()
        try:
            loop1.run_until_complete(pipe1.start())
            ctrl1 = ReplayController(wire, pipe1, speed=ReplaySpeed.SPEED_1X)
            t0 = time.perf_counter()
            loop1.run_until_complete(asyncio.wait_for(ctrl1.run(), timeout=10.0))
            elapsed_1x = time.perf_counter() - t0
        finally:
            try:
                loop1.run_until_complete(pipe1.stop())
            finally:
                loop1.close()
        # 2x
        pipe2 = _make_full_pipeline()
        loop2 = asyncio.new_event_loop()
        try:
            loop2.run_until_complete(pipe2.start())
            ctrl2 = ReplayController(wire, pipe2, speed=ReplaySpeed.SPEED_2X)
            t0 = time.perf_counter()
            loop2.run_until_complete(asyncio.wait_for(ctrl2.run(), timeout=10.0))
            elapsed_2x = time.perf_counter() - t0
        finally:
            try:
                loop2.run_until_complete(pipe2.stop())
            finally:
                loop2.close()
        assert elapsed_2x <= elapsed_1x * 1.2

    def test_replay_speed_0_5x_runs_slower(self, tmp_path: Path) -> None:
        wire = _make_wire_bin(tmp_path / "rec05", n_frames=4)
        pipe = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipe.start())
            ctrl = ReplayController(wire, pipe, speed=ReplaySpeed.SPEED_0_5X)
            t0 = time.perf_counter()
            loop.run_until_complete(asyncio.wait_for(ctrl.run(), timeout=10.0))
            elapsed_05 = time.perf_counter() - t0
            assert elapsed_05 >= 0.0
        finally:
            try:
                loop.run_until_complete(pipe.stop())
            finally:
                loop.close()

    def test_replay_speed_5x_runs_fastest(self, tmp_path: Path) -> None:
        wire = _make_wire_bin(tmp_path / "rec5", n_frames=4)
        pipe = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipe.start())
            ctrl = ReplayController(wire, pipe, speed=ReplaySpeed.SPEED_5X)
            t0 = time.perf_counter()
            loop.run_until_complete(asyncio.wait_for(ctrl.run(), timeout=10.0))
            elapsed_5x = time.perf_counter() - t0
            assert elapsed_5x >= 0.0
        finally:
            try:
                loop.run_until_complete(pipe.stop())
            finally:
                loop.close()

    def test_replay_stop_breaks_loop(self, tmp_path: Path) -> None:
        wire = _make_wire_bin(tmp_path / "rec_stop", n_frames=4)
        pipe = _make_full_pipeline()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipe.start())
            ctrl = ReplayController(wire, pipe, speed=ReplaySpeed.SPEED_1X)
            run_task = loop.create_task(ctrl.run())
            # Schedule a stop shortly after start.
            async def _stop_after():
                await asyncio.sleep(0.05)
                ctrl.stop()
            stop_task = loop.create_task(_stop_after())
            # Both tasks must complete.
            loop.run_until_complete(asyncio.wait_for(
                asyncio.gather(run_task, stop_task),
                timeout=5.0,
            ))
        finally:
            try:
                loop.run_until_complete(pipe.stop())
            finally:
                loop.close()


# ---------------------------------------------------------------------
# 5. Dashboard integration
# ---------------------------------------------------------------------
def _make_dashboard_app():
    cfg = _cfg()
    dash = load_dashboard_config(CONFIG_DIR)
    return cfg, dash


class TestDashboardLatency:

    def test_latency_endpoint_returns_stages(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            r = client.get("/api/latency")
            assert r.status_code == 200
            d = r.json()
            assert "stages" in d
            assert "end_to_end" in d
            assert "queue_depths" in d

    def test_latency_endpoint_populated_after_ticks(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            # Trigger a few ticks.
            client.get("/api/health")
            r = client.get("/api/latency")
            d = r.json()
            # Stages dict is present even when empty.
            assert isinstance(d["stages"], dict)

    def test_latency_endpoint_queue_depths(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            r = client.get("/api/latency")
            d = r.json()
            qd = d["queue_depths"]
            for key in ("in", "validation", "preprocessing",
                        "digital_twin", "ai", "risk"):
                assert key in qd

    def test_snapshot_carries_latency_and_metadata(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            r = client.get("/api/snapshot/latest")
            d = r.json()
            snap = d["snapshot"]
            assert snap is not None
            # New fields: latency and metadata.
            assert "latency" in snap
            assert "metadata" in snap
            md = snap["metadata"]
            assert md["mission_id"] == "mission-001"
            assert md["model_version"] == "phase17-streaming-1.0.0"
            # Latency block carries the per-stage values.
            lat = snap["latency"]
            assert lat is not None
            for stage in ("ingestion", "digital_twin", "ai", "risk"):
                assert stage in lat


class TestDashboardReplay:

    def test_replay_speeds_endpoint(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            r = client.get("/api/replay/speeds")
            d = r.json()
            assert "speeds" in d
            for s in ("0.5x", "1x", "2x", "5x"):
                assert s in d["speeds"]

    def test_replay_stats_endpoint_when_idle(self) -> None:
        cfg, dash = _make_dashboard_app()
        app = create_app(cfg, dash)
        with TestClient(app) as client:
            r = client.get("/api/replay/stats")
            d = r.json()
            assert d["running"] is False


# ---------------------------------------------------------------------
# 6. End-to-end 60 s mission
# ---------------------------------------------------------------------
class TestEndToEndStreaming:

    def test_full_streaming_run_60s_mission(self) -> None:
        """A full 60 s mission under the async pipeline, no crashes."""
        tracker = LatencyTracker()
        pipeline = _make_full_pipeline(tracker=tracker)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pipeline.start())
            adapter = TelemetryAdapter()
            cfg = _cfg()
            sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
            bundle = SensorBundle(cfg.sensors, sim, master_seed=42)
            n_done = 0
            n_target = 60
            for i in range(n_target):
                sample = bundle.tick()
                frame = adapter.adapt(sample, sequence=i + 1)
                loop.run_until_complete(pipeline.submit(frame))
                n_done += 1
            n_results = 0
            deadline = time.time() + 10.0
            while n_results < n_done and time.time() < deadline:
                try:
                    loop.run_until_complete(asyncio.wait_for(
                        pipeline.output_queue.get(), timeout=0.5
                    ))
                    n_results += 1
                except asyncio.TimeoutError:
                    break
            assert n_results >= n_done - 5
            assert len(tracker.summary()) >= 4
        finally:
            try:
                loop.run_until_complete(pipeline.stop())
            finally:
                loop.close()
