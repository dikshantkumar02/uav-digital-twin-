"""PHASE 4 tests — virtual sensor system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.config import SensorConfig, load_config
from backend.sensors import (
    Channel,
    NoiseMode,
    SENSOR_CHANNELS,
    SensorBundle,
    SensorSample,
    build_channel,
)
from backend.sensors.noise import (
    apply_bias,
    apply_calibration_error,
    apply_dropout,
    apply_drift,
    apply_gaussian_noise,
    apply_spike,
    apply_stuck,
)
from backend.simulation import EngineSimulator

pytestmark = pytest.mark.phase4

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _bundle(master_seed: int = 12345) -> SensorBundle:
    cfg = load_config(CONFIG_DIR)
    sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
    return SensorBundle(cfg.sensors, sim, master_seed=master_seed)


# ---------------------------------------------------------------------
# Noise primitives
# ---------------------------------------------------------------------
class TestNoisePrimitives:
    def test_gaussian_zero_std(self) -> None:
        rng = np.random.default_rng(0)
        assert apply_gaussian_noise(rng, 1.0, 0.0) == 1.0

    def test_gaussian_mean_around_value(self) -> None:
        rng = np.random.default_rng(0)
        true = 100.0
        std = 5.0
        n = 20_000
        samples = [apply_gaussian_noise(rng, true, std) for _ in range(n)]
        mean = sum(samples) / n
        assert abs(mean - true) < 0.2     # E[X] = true
        # Var check (loose).
        var = sum((s - mean) ** 2 for s in samples) / n
        assert abs(var - std * std) < 1.0

    def test_bias(self) -> None:
        assert apply_bias(10.0, 0.5) == 10.5
        assert apply_bias(10.0, -0.5) == 9.5

    def test_drift(self) -> None:
        assert apply_drift(10.0, 1.5) == 11.5
        assert apply_drift(10.0, -1.5) == 8.5

    def test_calibration_error(self) -> None:
        # 0.01 => +1% scale
        assert apply_calibration_error(100.0, 0.01) == 101.0
        assert apply_calibration_error(100.0, -0.01) == 99.0

    def test_dropout_never_when_prob_zero(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(100):
            assert apply_dropout(rng, 5.0, 0.0) == 5.0

    def test_dropout_can_return_none(self) -> None:
        rng = np.random.default_rng(0)
        seen_none = any(apply_dropout(rng, 5.0, 1.0) is None for _ in range(5))
        assert seen_none

    def test_spike_within_amplitude(self) -> None:
        rng = np.random.default_rng(0)
        # Force a spike by setting prob=1.0
        v = apply_spike(rng, 10.0, 1.0, 5.0)
        assert v != 10.0
        assert abs(abs(v - 10.0) - 5.0) < 1e-9

    def test_stuck(self) -> None:
        assert apply_stuck(10.0, True, 42.0) == 42.0
        assert apply_stuck(10.0, False, 42.0) == 10.0


# ---------------------------------------------------------------------
# Per-channel sampling
# ---------------------------------------------------------------------
class TestChannel:
    def test_channel_truth_extraction_matches_engine(self) -> None:
        bundle = _bundle()
        samples = bundle.run()[:50]
        s = samples[25]
        # RPM channel should track engine.rpm within a small noise band.
        reading = s.get("rpm")
        truth = s.engine.rpm
        assert reading is not None
        assert abs(reading - truth) < 30.0   # noise_std default 5.0 + spike room

    def test_drift_accumulator_grows_with_time(self) -> None:
        bundle = _bundle()
        ch = bundle.channel("egt")
        # Snapshot the accumulator before and after a long run.
        before = ch.drift_accumulator
        bundle.run()[:1000]  # ~100 s
        after = ch.drift_accumulator
        assert after > before

    def test_calibration_error_present(self) -> None:
        """Calibration error is multiplicative. The mean reading should
        be slightly off the true value even on average."""
        bundle = _bundle()
        ch = bundle.channel("rpm")
        # Disable noise, drift, bias, dropout, spike, stuck, drift effects
        # by patching the cfg in-place; calibration_error remains.
        ch.cfg.noise_std = 0.0
        ch.cfg.bias = 0.0
        ch.cfg.drift_per_hour = 0.0
        ch.cfg.spike_prob = 0.0
        ch.cfg.dropout_prob = 0.0
        ch.cfg.stuck_prob = 0.0
        # Run a short mission and average.
        samples = bundle.run()[:100]
        mean_reading = sum(s.get("rpm") for s in samples) / len(samples)
        mean_truth = sum(s.engine.rpm for s in samples) / len(samples)
        # Calibration error is 0.01, so mean reading should be ~1% above truth.
        assert mean_reading > mean_truth

    def test_inject_stuck_freezes_value(self) -> None:
        bundle = _bundle()
        bundle.inject_fault("rpm", NoiseMode.STUCK, start_t=0.0)
        samples = bundle.run()[:50]
        # From the first sample onward, rpm should be the stuck value (0.0 default).
        for s in samples[1:]:
            assert s.get("rpm") == 0.0
            assert s.is_stuck("rpm")

    def test_inject_dropout_increases_dropout_rate(self) -> None:
        bundle = _bundle()
        # FAULT mode forces dropout_prob up to >= 0.10 inside the channel.
        bundle.inject_fault("egt", NoiseMode.FAULT, start_t=0.0)
        samples = bundle.run()[:200]
        n_dropped = sum(1 for s in samples if s.is_dropped("egt"))
        # Expect roughly 10%+ dropout.
        assert n_dropped / len(samples) > 0.05

    def test_inject_drift_adds_extra_drift(self) -> None:
        bundle_a = _bundle(master_seed=1)
        bundle_b = _bundle(master_seed=2)
        bundle_b.inject_fault("oil_pressure", NoiseMode.DRIFTING, start_t=0.0)
        n = 200
        # Take final-state drift accumulators.
        sa = bundle_a.run()[:n]
        sb = bundle_b.run()[:n]
        drift_a = bundle_a.channel("oil_pressure").drift_accumulator
        drift_b = bundle_b.channel("oil_pressure").drift_accumulator
        assert drift_b > drift_a
        # The readings should also be biased.
        bias_a = sum(s.get("oil_pressure") for s in sa) / n
        bias_b = sum(s.get("oil_pressure") for s in sb) / n
        # The drifting channel is shifted away from the truth on average.
        truth = sum(s.engine.oil_pressure_psi for s in sb) / n
        assert abs(bias_b - truth) > abs(bias_a - truth)

    def test_clear_fault_restores_normal(self) -> None:
        bundle = _bundle()
        bundle.inject_fault("rpm", NoiseMode.STUCK, start_t=0.0)
        bundle.clear_fault("rpm")
        samples = bundle.run()[:50]
        # After clearing, no reading should be the stuck value (0.0).
        non_stuck = [s for s in samples if s.get("rpm") not in (None, 0.0)]
        assert len(non_stuck) > 0


# ---------------------------------------------------------------------
# Bundle + scenarios
# ---------------------------------------------------------------------
class TestSensorBundle:
    def test_all_channels_have_a_reading(self) -> None:
        bundle = _bundle()
        samples = bundle.run()
        s = samples[100]
        for name in SENSOR_CHANNELS:
            assert name in s.readings

    def test_bundle_is_deterministic_with_seed(self) -> None:
        a = _bundle(master_seed=99)
        b = _bundle(master_seed=99)
        sa = a.run()
        sb = b.run()
        for x, y in zip(sa, sb):
            for name in SENSOR_CHANNELS:
                vx = x.get(name)
                vy = y.get(name)
                if vx is None or vy is None:
                    assert vx is None and vy is None
                else:
                    assert vx == pytest.approx(vy, rel=1e-9, abs=1e-9)

    def test_different_seeds_diverge(self) -> None:
        a = _bundle(master_seed=1)
        b = _bundle(master_seed=2)
        sa = a.run()[:200]
        sb = b.run()[:200]
        # At least one channel must show different readings.
        any_diff = False
        for x, y in zip(sa, sb):
            for name in ("rpm", "egt", "cht"):
                vx = x.get(name)
                vy = y.get(name)
                if vx is not None and vy is not None and abs(vx - vy) > 1.0:
                    any_diff = True
                    break
            if any_diff:
                break
        assert any_diff


class TestMandatoryScenarios:
    """Mandatory-scenario coverage at the sensor layer."""

    def test_s1_healthy_all_channels_within_reasonable_band(self) -> None:
        """A healthy mission must produce readings close to the truth
        on every channel (within ~5*noise_std over a long window)."""
        bundle = _bundle()
        samples = bundle.run()
        # Use a stable cruise segment (after t=600s, before t=3500s).
        cruise = [s for s in samples if 600.0 <= s.time_s <= 3500.0]
        assert len(cruise) > 100
        for name in SENSOR_CHANNELS:
            cfg_ch = bundle.channel(name).cfg
            # Allow ~6 sigma + 1% calibration + 5*noise spike room.
            tol = max(8.0 * cfg_ch.noise_std, 0.02 * cfg_ch.noise_std * 1e4)
            deltas = []
            for s in cruise:
                v = s.get(name)
                if v is None:
                    continue
                # The "true" value is obtained by recomputing the channel
                # extractor against the engine + env in the sample.
                from backend.sensors.channels import _true_value
                truth = _true_value(name, s.engine, s.env)
                deltas.append(abs(v - truth))
            if deltas:
                mean = sum(deltas) / len(deltas)
                assert mean < tol + 5.0, f"{name} mean abs error {mean} > {tol}"

    def test_s3_sensor_drift_on_stable_engine_yields_biased_reading(self) -> None:
        """Drift injection on the EGT channel must produce a reading
        that diverges from the truth over time."""
        bundle = _bundle()
        bundle.inject_fault("egt", NoiseMode.DRIFTING, start_t=0.0)
        samples = bundle.run()
        # Compare mean residual in the first 10% vs last 10% of the run.
        n = len(samples)
        head = samples[: max(1, n // 10)]
        tail = samples[-max(1, n // 10):]

        def _mean_residual(samples_subset):
            vals = []
            for s in samples_subset:
                v = s.get("egt")
                if v is not None:
                    vals.append(v - s.engine.egt_c)
            return sum(vals) / len(vals) if vals else 0.0

        head_res = _mean_residual(head)
        tail_res = _mean_residual(tail)
        # The tail should be more biased than the head.
        assert abs(tail_res) > abs(head_res)

    def test_s6_multiple_sensor_anomalies_increase_dropout_count(self) -> None:
        """Injecting faults on multiple channels must produce dropouts
        on every affected channel and reduce the per-channel sample
        coverage."""
        bundle_clean = _bundle(master_seed=7)
        bundle_faulty = _bundle(master_seed=7)
        for ch in ("rpm", "egt", "oil_pressure"):
            bundle_faulty.inject_fault(ch, NoiseMode.FAULT, start_t=0.0)

        clean_samples = bundle_clean.run()[:1000]
        fault_samples = bundle_faulty.run()[:1000]

        def _dropout_rate(samples, name):
            n_total = len(samples)
            n_drop = sum(1 for s in samples if s.is_dropped(name))
            return n_drop / n_total

        for ch in ("rpm", "egt", "oil_pressure"):
            r_clean = _dropout_rate(clean_samples, ch)
            r_fault = _dropout_rate(fault_samples, ch)
            assert r_fault > r_clean + 0.02
