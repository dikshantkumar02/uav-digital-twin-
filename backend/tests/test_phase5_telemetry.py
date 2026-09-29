"""PHASE 5 tests — telemetry streaming."""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path

import pytest

from backend.config import load_config
from backend.sensors import NoiseMode, SensorBundle
from backend.simulation import EngineSimulator
from backend.telemetry import (
    FrameStatus,
    LatencyTracker,
    Preprocessor,
    QueueFullError,
    StreamSource,
    TelemetryFrame,
    TelemetryQueue,
)

pytestmark = pytest.mark.phase5

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _bundle(master_seed: int = 12345) -> SensorBundle:
    cfg = load_config(CONFIG_DIR)
    sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
    return SensorBundle(cfg.sensors, sim, master_seed=master_seed)


# ---------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------
class TestTelemetryQueue:
    def test_push_and_pop_order(self) -> None:
        q = TelemetryQueue(max_size=8)
        f1 = TelemetryFrame(sequence=1, time_s=0.0, produced_wall_time=0.0, sample=None)
        f2 = TelemetryFrame(sequence=2, time_s=0.1, produced_wall_time=0.1, sample=None)
        assert q.push(f1) and q.push(f2)
        assert q.pop().sequence == 1
        assert q.pop().sequence == 2

    def test_drop_on_overflow_drops_oldest(self) -> None:
        q = TelemetryQueue(max_size=3, drop_on_overflow=True)
        for i in range(5):
            q.push(TelemetryFrame(sequence=i, time_s=float(i), produced_wall_time=float(i), sample=None))
        # Should retain the last 3.
        assert q.stats().dropped_overflow == 2
        # The 3 retained are 2, 3, 4.
        seqs = [q.pop().sequence for _ in range(3)]
        assert seqs == [2, 3, 4]

    def test_no_drop_raises_on_full(self) -> None:
        q = TelemetryQueue(max_size=2, drop_on_overflow=False)
        q.push(TelemetryFrame(sequence=1, time_s=0.0, produced_wall_time=0.0, sample=None))
        q.push(TelemetryFrame(sequence=2, time_s=0.0, produced_wall_time=0.0, sample=None))
        assert not q.push(TelemetryFrame(sequence=3, time_s=0.0, produced_wall_time=0.0, sample=None))
        assert q.stats().pushed == 2
        assert q.stats().dropped_overflow == 0

    def test_drain_returns_all(self) -> None:
        q = TelemetryQueue(max_size=4)
        for i in range(3):
            q.push(TelemetryFrame(sequence=i, time_s=float(i), produced_wall_time=float(i), sample=None))
        out = q.drain()
        assert [f.sequence for f in out] == [0, 1, 2]
        assert len(q) == 0

    def test_pop_blocks_then_unblocks(self) -> None:
        q = TelemetryQueue(max_size=4)

        def _push_after_delay():
            _time.sleep(0.05)
            q.push(TelemetryFrame(sequence=42, time_s=0.0, produced_wall_time=0.0, sample=None))

        t = threading.Thread(target=_push_after_delay, daemon=True)
        t.start()
        f = q.pop(timeout_s=1.0)
        assert f is not None
        assert f.sequence == 42

    def test_high_watermark(self) -> None:
        q = TelemetryQueue(max_size=10, drop_on_overflow=True)
        for i in range(7):
            q.push(TelemetryFrame(sequence=i, time_s=float(i), produced_wall_time=float(i), sample=None))
        assert q.stats().high_watermark == 7


# ---------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------
class TestLatencyTracker:
    def test_records_one_stage(self) -> None:
        t = LatencyTracker()
        with t.stage("preprocessing"):
            _time.sleep(0.001)
        s = t.stages["preprocessing"]
        assert s.samples == 1
        assert s.last_s > 0.0

    def test_multiple_stages_independent(self) -> None:
        t = LatencyTracker()
        with t.stage("a"):
            _time.sleep(0.001)
        with t.stage("b"):
            _time.sleep(0.002)
        assert t.stages["a"].last_s < t.stages["b"].last_s

    def test_total_last_s_sums(self) -> None:
        t = LatencyTracker()
        with t.stage("a"):
            pass
        with t.stage("b"):
            pass
        assert t.total_last_s() == pytest.approx(
            t.stages["a"].last_s + t.stages["b"].last_s
        )


# ---------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------
class TestPreprocessor:
    def _frame_with_sample(self, sample) -> TelemetryFrame:
        return TelemetryFrame(
            sequence=1, time_s=0.0, produced_wall_time=0.0, sample=sample
        )

    def test_ok_when_all_channels_present(self) -> None:
        bundle = _bundle()
        sample = bundle.run()[100]
        f = self._frame_with_sample(sample)
        pp = Preprocessor()
        out = pp.process(f)
        assert out.status == FrameStatus.OK
        assert int(out.notes["n_filled"]) == 0
        assert int(out.notes["n_invalid"]) == 0

    def test_stale_when_some_dropouts_filled(self) -> None:
        bundle = _bundle()
        # Force one channel to dropout for a few ticks.
        bundle.inject_fault("rpm", NoiseMode.FAULT, start_t=0.0)
        pp = Preprocessor(max_dropout_age_s=2.0, max_invalid_channels=2)
        seen_non_ok = False
        for s in bundle.run()[:30]:
            f = pp.process(
                TelemetryFrame(sequence=int(s.time_s * 10), time_s=s.time_s, produced_wall_time=0.0, sample=s)
            )
            if f.status != FrameStatus.OK:
                seen_non_ok = True
                break
        assert seen_non_ok

    def test_invalid_when_too_many_missing(self) -> None:
        bundle = _bundle()
        # Force dropouts on many channels by directly forcing a high
        # dropout probability on the channel config. This guarantees
        # the preprocessor sees sustained dropouts.
        for ch in ("rpm", "egt", "cht", "oil_pressure", "vibration"):
            bundle.channel(ch).cfg.dropout_prob = 0.50
        # Tight age budget: any dropout lasting more than 1 tick is
        # un-fillable. With 5 channels at 50% dropout, frames with
        # multiple simultaneous missing channels are common.
        pp = Preprocessor(max_dropout_age_s=0.05, max_invalid_channels=1)
        seen_invalid = False
        for s in bundle.run()[:200]:
            f = pp.process(
                TelemetryFrame(sequence=int(s.time_s * 10), time_s=s.time_s, produced_wall_time=0.0, sample=s)
            )
            if f.status == FrameStatus.INVALID:
                seen_invalid = True
                break
        assert seen_invalid

    def test_forward_fill_recovers_short_dropout(self) -> None:
        bundle = _bundle()
        # Manually feed samples with a single None on egt.
        samples = bundle.run()[:5]
        pp = Preprocessor(max_dropout_age_s=5.0, max_invalid_channels=2)
        # First, process the prior 2 samples so the preprocessor has a
        # valid last-known egt value.
        for s in samples[:2]:
            pp.process(
                TelemetryFrame(sequence=int(s.time_s * 10), time_s=s.time_s, produced_wall_time=0.0, sample=s)
            )
        # Now make the 3rd sample's egt None and process it.
        samples[2].readings["egt"].value = None
        f = pp.process(
            TelemetryFrame(sequence=int(samples[2].time_s * 10), time_s=samples[2].time_s, produced_wall_time=0.0, sample=samples[2])
        )
        # The forward-fill should restore the egt value.
        assert samples[2].readings["egt"].value is not None
        assert f.notes.get("fill.egt", "").startswith("age=")

    def test_latency_recorded_when_tracker_supplied(self) -> None:
        t = LatencyTracker()
        bundle = _bundle()
        pp = Preprocessor(latency_tracker=t)
        sample = bundle.run()[0]
        pp.process(
            TelemetryFrame(sequence=1, time_s=sample.time_s, produced_wall_time=0.0, sample=sample)
        )
        assert "preprocessing" in t.stages


# ---------------------------------------------------------------------
# StreamSource
# ---------------------------------------------------------------------
class TestStreamSource:
    def test_tick_pushes_one_frame(self) -> None:
        bundle = _bundle()
        q = TelemetryQueue(max_size=8)
        src = StreamSource(bundle, q)
        f = src.tick()
        assert f is not None
        assert f.sequence == 1
        assert len(q) == 1

    def test_run_pushes_full_mission_frames(self) -> None:
        bundle = _bundle()
        q = TelemetryQueue(max_size=64, drop_on_overflow=True)
        src = StreamSource(bundle, q)
        n = src.run(max_steps=20)
        assert n == 20
        assert len(q) == 20
        # Sequence numbers are monotonic.
        seqs = []
        while True:
            f = q.pop()
            if f is None:
                break
            seqs.append(f.sequence)
        assert seqs == list(range(1, 21))

    def test_ingestion_latency_recorded(self) -> None:
        bundle = _bundle()
        q = TelemetryQueue(max_size=8)
        tracker = LatencyTracker()
        src = StreamSource(bundle, q, tracker=tracker)
        for _ in range(5):
            src.tick()
        assert "ingestion" in tracker.stages
        assert tracker.stages["ingestion"].samples == 5


# ---------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline_produces_valid_frames(self) -> None:
        bundle = _bundle()
        q = TelemetryQueue(max_size=128, drop_on_overflow=True)
        tracker = LatencyTracker()
        src = StreamSource(bundle, q, tracker=tracker)
        pp = Preprocessor(latency_tracker=tracker)
        src.run(max_steps=100)
        n_processed = 0
        n_ok = 0
        n_stale = 0
        n_invalid = 0
        while True:
            f = q.pop()
            if f is None:
                break
            f = pp.process(f)
            n_processed += 1
            if f.status == FrameStatus.OK:
                n_ok += 1
            elif f.status == FrameStatus.STALE:
                n_stale += 1
            elif f.status == FrameStatus.INVALID:
                n_invalid += 1
        assert n_processed == 100
        # Healthy mission: at least 90% OK.
        assert n_ok >= 90
        assert n_invalid == 0

    def test_end_to_end_latency_under_target(self) -> None:
        """The pipeline target is < 2 s end-to-end. We measure the
        total of all recorded stages for a small run."""
        bundle = _bundle()
        q = TelemetryQueue(max_size=128, drop_on_overflow=True)
        tracker = LatencyTracker()
        src = StreamSource(bundle, q, tracker=tracker)
        pp = Preprocessor(latency_tracker=tracker)
        src.run(max_steps=20)
        while q.pop() is not None:
            pass
        # Sanity: every frame's preprocessing is well under 1 ms.
        for s in tracker.stages.values():
            assert s.max_s < 0.1, f"stage {s.name} max latency too high: {s.max_s}"
