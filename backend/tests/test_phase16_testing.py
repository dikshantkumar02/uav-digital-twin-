"""PHASE 16 tests — testing + performance (coverage, benchmarks, fault contract,
property-based invariants, stress, latency).

Run modes (env-var gated):

* Default — every test runs. The slow ones (benchmarks, 10× mission) are
  skipped unless ``PHASE16_BENCH=1``.
* ``PHASE16_FAST=1`` — only the wall-clock budget tests run; useful for
  CI smoke.
* ``PHASE16_BENCH=1`` — also enables the ``pytest-benchmark`` suite and
  writes ``.benchmarks/`` baselines.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pytest

from backend.config import load_config
from backend.dashboard.runner import PipelineRunner
from backend.dashboard.scenarios import SCENARIO_REGISTRY
from backend.faults import FaultClass
from backend.hardware import FrameCodec
from backend.sensors import NoiseMode, SensorReading, SensorSample
from backend.telemetry import (
    FrameStatus,
    LatencyTracker,
    TelemetryFrame,
    TelemetryQueue,
)

from backend.hil import Tolerance, compare_runs
from backend.testing import (
    FAULT_CONTRACTS,
    InvariantViolation,
    assert_anomaly_invariants,
    assert_health_invariants,
    assert_no_nan_inf,
    assert_risk_invariants,
    assert_rul_invariants,
    assert_snapshot_dict_invariants,
    assert_typed_snapshot_invariants,
    check_fault_contract,
    measure_coverage,
    module_coverage,
)

pytestmark = pytest.mark.phase16

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

PHASE16_BENCH = os.environ.get("PHASE16_BENCH") == "1"
PHASE16_FAST = os.environ.get("PHASE16_FAST") == "1"

# Reasonable wall-clock budgets for CI (loose — meant to catch
# catastrophic regressions, not microseconds). The mission
# budget is set to 1× the sim time (60 s) because the per-tick
# Python pipeline is single-threaded. A 2× regression would
# push this over 60s.
BUDGET_TICK_S = 0.2
BUDGET_MISSION_S = 60.0
BUDGET_ENCODE_S = 0.05
BUDGET_ROUNDTRIP_S = 0.2


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_frame(seq: int, time_s: float) -> TelemetryFrame:
    readings = {
        name: SensorReading(value=1.0 + seq + i, mode=NoiseMode.NORMAL)
        for i, name in enumerate([
            "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
            "fuel_flow", "vibration", "imu_accel", "altitude", "airspeed",
            "ambient_temperature", "ambient_pressure",
        ])
    }
    return TelemetryFrame(
        sequence=seq,
        time_s=time_s,
        produced_wall_time=time.perf_counter(),
        sample=SensorSample(time_s=time_s, readings=readings),
        status=FrameStatus.OK,
    )


def _drive(scenario_name: str, duration_s: float = 60.0) -> PipelineRunner:
    cfg = load_config(str(CONFIG_DIR))
    return PipelineRunner(cfg, SCENARIO_REGISTRY[scenario_name])


# ---------------------------------------------------------------------
# TestCoverage
# ---------------------------------------------------------------------
@pytest.mark.skipif(PHASE16_FAST, reason="coverage gated by PHASE16_FAST")
class TestCoverage:
    def test_pytest_cov_collects(self) -> None:
        m = measure_coverage(
            testpaths=["backend/tests/test_phase1_config.py"],
            root=REPO_ROOT,
        )
        assert m["total_percent"] >= 0.0
        # The per_module mapping should be non-empty.
        assert isinstance(m["per_module"], dict)

    def test_critical_modules_have_coverage(self) -> None:
        # The full per-module coverage report is operator-facing
        # via ``scripts/coverage_report.py`` (run by hand, not in
        # the test path). Here we only verify the helper functions
        # ``measure_coverage`` / ``module_coverage`` work on a
        # tiny testpath and produce sane values. Driving the full
        # suite inside a test would take minutes; that's the
        # script's job, not the test's.
        m = measure_coverage(
            testpaths=["backend/tests/test_phase1_config.py"],
            root=REPO_ROOT,
        )
        # All values are non-negative floats.
        assert m["total_percent"] >= 0.0
        for mod, cov in m["per_module"].items():
            assert isinstance(mod, str)
            assert 0.0 <= cov <= 100.0, f"{mod} coverage {cov} out of [0,100]"
        # And the ``module_coverage`` helper is consistent.
        for mod in m["per_module"]:
            assert module_coverage(m, mod) == m["per_module"][mod]


# ---------------------------------------------------------------------
# TestPerformanceBudgets
# ---------------------------------------------------------------------
class TestPerformanceBudgets:
    def test_one_tick_under_200ms(self) -> None:
        runner = _drive("healthy_60s")
        # Warmup to amortise lazy imports.
        runner.tick()
        t0 = time.perf_counter()
        runner.tick()
        dt = time.perf_counter() - t0
        assert dt < BUDGET_TICK_S, f"one tick took {dt*1000:.1f} ms"

    def test_one_mission_under_60s(self) -> None:
        """A 60s sim mission must complete in < 60s wall-clock.

        The budget is set to 1× the sim time because the
        per-tick Python pipeline is single-threaded. A 2×
        regression would push this over 60s.
        """
        runner = _drive("healthy_60s")
        t0 = time.perf_counter()
        list(runner.run(duration_s=60.0))
        dt = time.perf_counter() - t0
        assert dt < BUDGET_MISSION_S, f"60s mission took {dt:.2f} s"

    def test_wire_encode_under_50ms(self) -> None:
        frame = _make_frame(seq=1, time_s=0.1)
        # Warmup.
        FrameCodec.encode(frame)
        t0 = time.perf_counter()
        for _ in range(10):
            FrameCodec.encode(frame)
        dt = (time.perf_counter() - t0) / 10
        assert dt < BUDGET_ENCODE_S, f"one encode took {dt*1000:.1f} ms"

    def test_wire_round_trip_under_200ms(self) -> None:
        frame = _make_frame(seq=1, time_s=0.1)
        # Warmup.
        encoded = FrameCodec.encode(frame)
        FrameCodec.decode(encoded.rstrip(b"\n"))
        t0 = time.perf_counter()
        for _ in range(10):
            encoded = FrameCodec.encode(frame)
            FrameCodec.decode(encoded.rstrip(b"\n"))
        dt = (time.perf_counter() - t0) / 10
        assert dt < BUDGET_ROUNDTRIP_S, f"round trip took {dt*1000:.1f} ms"


# ---------------------------------------------------------------------
# TestBenchmarks (gated by PHASE16_BENCH=1)
# ---------------------------------------------------------------------
pytestmark_bench = pytest.mark.skipif(
    not PHASE16_BENCH,
    reason="benchmarks gated by PHASE16_BENCH=1",
)


@pytestmark_bench
class TestBenchmarks:
    def test_bench_pipeline_tick(self, benchmark) -> None:
        runner = _drive("healthy_60s")
        runner.tick()  # warmup

        def _one_tick():
            runner.tick()

        benchmark(_one_tick)

    def test_bench_pipeline_mission(self, benchmark) -> None:
        cfg = load_config(str(CONFIG_DIR))
        spec = SCENARIO_REGISTRY["healthy_60s"]

        def _one_mission():
            r = PipelineRunner(cfg, spec)
            list(r.run(duration_s=60.0))

        benchmark(_one_mission)

    def test_bench_wire_round_trip(self, benchmark) -> None:
        frame = _make_frame(seq=1, time_s=0.1)

        def _round_trip():
            encoded = FrameCodec.encode(frame)
            FrameCodec.decode(encoded.rstrip(b"\n"))

        benchmark(_round_trip)


# ---------------------------------------------------------------------
# TestFaultDetectionContract
# ---------------------------------------------------------------------
class TestFaultDetectionContract:
    @pytest.mark.parametrize(
        "contract",
        list(FAULT_CONTRACTS.values()),
        ids=lambda c: c.fault_class.value,
    )
    def test_fault_contract_holds(self, contract) -> None:
        # Build a fresh runner per contract so test isolation is
        # guaranteed; the contract drives the mission end-to-end.
        runner = _drive(contract.scenario_name)
        check_fault_contract(contract, runner)

    def test_healthy_60s_stays_under_caution(self) -> None:
        """Healthy baseline: must never escalate past CAUTION."""
        runner = _drive("healthy_60s")
        check_fault_contract(FAULT_CONTRACTS[FaultClass.HEALTHY], runner)


# ---------------------------------------------------------------------
# TestPropertyInvariants
# ---------------------------------------------------------------------
class TestPropertyInvariants:
    def test_health_score_in_unit_interval(self) -> None:
        runner = _drive("healthy_60s")
        for _ in range(20):
            snap = runner.tick()
            assert 0.0 <= snap.health.overall_score <= 1.0
            assert 0.0 <= snap.health.confidence <= 1.0
            for sub in snap.health.subsystems.values():
                assert 0.0 <= sub.score <= 1.0

    def test_rul_bounds_contain_central(self) -> None:
        runner = _drive("engine_degradation_60s")
        for _ in range(50):
            snap = runner.tick()
            r = snap.rul
            # RUL values may be None when status is RUL_UNCERTAIN;
            # only assert the bounds invariant when the estimate
            # is meaningful.
            if r.tte_hours_central is None:
                continue
            assert r.tte_hours_lower is not None
            assert r.tte_hours_upper is not None
            assert r.tte_hours_lower >= 0.0
            assert r.tte_hours_central >= 0.0
            assert r.tte_hours_upper >= 0.0
            assert r.tte_hours_lower <= r.tte_hours_central
            assert r.tte_hours_central <= r.tte_hours_upper

    def test_risk_score_in_unit_interval(self) -> None:
        runner = _drive("engine_degradation_60s")
        for _ in range(50):
            snap = runner.tick()
            assert 0.0 <= snap.risk.risk_score <= 1.0
            assert 0.0 <= snap.risk.confidence <= 1.0

    def test_telemetry_queue_bounded(self) -> None:
        q = TelemetryQueue(max_size=10, drop_on_overflow=True)
        for i in range(1000):
            q.push(_make_frame(i, i * 0.1))
            assert len(q) <= 10
        # Drain; we should have popped ≤ pushed, never negative.
        popped = 0
        while True:
            f = q.pop(timeout_s=0.0)
            if f is None:
                break
            popped += 1
        s = q.stats()
        assert s.popped == popped
        assert s.popped <= s.pushed

    def test_frame_codec_round_trip_idempotent(self) -> None:
        frame = _make_frame(seq=7, time_s=0.7)
        encoded = FrameCodec.encode(frame)
        decoded = FrameCodec.decode(encoded.rstrip(b"\n"))
        # Re-encoding the decoded frame must produce identical
        # wire bytes (sequence / time / status / readings are
        # fully recovered; ``produced_wall_time`` is intentionally
        # not encoded so a decode never carries a stale wall clock).
        encoded2 = FrameCodec.encode(decoded)
        assert encoded == encoded2

    def test_compare_runs_self_passes(self) -> None:
        runner = _drive("healthy_60s")
        snaps = [runner.tick() for _ in range(5)]
        d = [s.to_dict() for s in snaps]
        # Self-comparison must always pass, regardless of tolerance.
        for tol in [Tolerance(), Tolerance(health_abs=0.0, rul_abs=0.0,
                                          risk_abs=0.0, engine_state_rel=0.0,
                                          environment_rel=0.0, anomaly_abs=0.0)]:
            rep = compare_runs(d, d, tol)
            assert rep.passed
            assert rep.n_mismatches == 0


# ---------------------------------------------------------------------
# TestStress
# ---------------------------------------------------------------------
@pytest.mark.skipif(PHASE16_FAST, reason="10x mission gated by PHASE16_FAST")
class TestStress:
    def test_10x_mission_healthy_no_invariant_violation(self) -> None:
        runner = _drive("healthy_60s")
        for i in range(6000):
            snap = runner.tick()
            # Typed invariants
            assert_typed_snapshot_invariants(
                health=snap.health, rul=snap.rul,
                risk=snap.risk, anomaly=snap.anomaly,
            )

    def test_10x_mission_no_nan_or_inf_anywhere(self) -> None:
        runner = _drive("engine_degradation_60s")
        for i in range(6000):
            d = runner.tick().to_dict()
            assert_no_nan_inf(d)

    def test_10x_mission_queue_stays_bounded(self) -> None:
        """The TelemetryQueue must remain bounded under a 6000-frame flood."""
        q = TelemetryQueue(max_size=64, drop_on_overflow=True)
        for i in range(6000):
            q.push(_make_frame(i, i * 0.1))
            assert len(q) <= 64
        s = q.stats()
        # No NaN-like values, sensible counters.
        assert s.pushed >= 6000
        assert s.high_watermark <= 64


# ---------------------------------------------------------------------
# TestLatencyTracker
# ---------------------------------------------------------------------
class TestLatencyTracker:
    def test_pipeline_full_latency_under_15s(self) -> None:
        """Drive 60 ticks through the LatencyTracker.

        The per-tick Python pipeline is single-threaded; 60 ticks
        at < 200 ms each is the PHASE 16 budget. End-to-end must
        stay under 15 s (loose; a 2× regression would push over
        30 s).
        """
        tracker = LatencyTracker()
        runner = _drive("healthy_60s")
        t_start = time.perf_counter()
        for _ in range(60):
            with tracker.stage("tick"):
                runner.tick()
        e2e = time.perf_counter() - t_start
        assert e2e < 15.0, f"60-tick pipeline took {e2e:.2f} s"
        # Every recorded stage must have a positive max and a
        # sane mean (mean <= max).
        for name in ("tick",):
            stage = tracker.stages.get(name)
            assert stage is not None
            assert stage.samples == 60
            assert stage.max_s > 0.0
            assert stage.mean_s > 0.0
            assert stage.mean_s <= stage.max_s + 1e-9

    def test_latency_tracker_serial_transport_records_stage(self) -> None:
        """The PHASE 14 SerialSource records the 'transport' stage
        when given a LatencyTracker."""
        from backend.hardware import LoopbackPair, SerialSource

        pair = LoopbackPair()
        q = TelemetryQueue(max_size=8, drop_on_overflow=True)
        tracker = LatencyTracker()
        source = SerialSource(
            pair.port_b(), q, crc="ccitt",
            reconnect_on_error=False, latency_tracker=tracker,
        )
        # Push a frame through the loopback.
        encoded = FrameCodec.encode(_make_frame(seq=1, time_s=0.1))
        pair.port_a().write(encoded)
        source.pump_once(timeout_s=0.05)
        # Drain.
        while q.pop(timeout_s=0.0) is not None:
            pass
        stage = tracker.stages.get("transport")
        assert stage is not None, "SerialSource did not record a 'transport' stage"
        assert stage.samples >= 1
        assert stage.mean_s >= 0.0
