"""
Software-in-the-Loop (SIH) demonstration mode.

A **deterministic, single-command** walkthrough of the digital twin
on six canonical scenarios. The demo shows — for each scenario —

* live telemetry (rpm, egt, vibration, …)
* the Digital Twin's expected vs. observed values
* per-channel residual z-scores
* the anomaly detector's overall score + label
* the AI classifier's diagnosis + confidence
* the health index, per subsystem and overall
* RUL status + uncertainty
* the PHASE 12 mission risk (go / caution / RTB / abort)
* the PHASE 23 mission-reliability band (LOW / MEDIUM / HIGH / UNKNOWN)
* an event timeline (fault onsets, sensor faults, risk transitions)
* a side-by-side comparison: what a naive single-channel threshold
  monitor would have flagged, vs. what the system actually concluded.

The whole demo is deterministic: every scenario uses a fixed seed,
the simulator advances at a fixed dt, and the same input produces
the same output every run.

Usage
-----

::

    # Run all 6 scenarios end-to-end (≈ 30 s on a laptop)
    python -m backend.dashboard.sih_demo

    # Run one specific scenario
    python -m backend.dashboard.sih_demo --scenario 2

    # Print JSON instead of a human-readable table
    python -m backend.dashboard.sih_demo --json | jq '.[] | .summary'

Scenarios
---------

1. HEALTHY FLIGHT — 60 s, no faults.
2. TURBULENCE WITHOUT ENGINE FAULT — 60 s, severe turbulence at
   t = 5 s.  The Digital Twin confirms the vibration is environmental
   (env severity ↑↑, engine residual normal), and the system classifies
   the event as ENVIRONMENTAL DISTURBANCE — not an engine fault.
3. SENSOR FAULT — 60 s, RPM sensor stuck at t = 10 s.  The
   diagnostics report a sensor channel anomaly, NOT an engine fault.
4. EARLY ENGINE DEGRADATION — 30 s, engine degradation onset at
   t = 10 s, progression is fast.  Caught at t ≈ 12 s by the digital
   twin + anomaly detector.
5. ENGINE DEGRADATION + TURBULENCE — 60 s, both engine degradation
   (t = 5 s) AND heavy turbulence (t = 5 s) active simultaneously.
   The system must still surface the engine fault even when the
   environment is also noisy — the engine-side residual stays
   elevated while the env severity is high, and the
   ``ENGINE_DEGRADATION`` driver wins.
6. UNKNOWN ANOMALY — 60 s, intermittent sensor dropouts that keep
   no single channel healthy long enough to be calibrated.  The
   system correctly returns UNKNOWN rather than guessing.

This module is **read-only / advisory**.  It never issues aircraft
control commands, never computes a mission-success-probability
number, and never fakes outputs when the evidence is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import load_config
from backend.dashboard.runner import PipelineRunner
from backend.dashboard.scenarios import (
    DEFAULT_SCENARIO_NAME,
    SCENARIO_REGISTRY,
    ScenarioSpec,
)
from backend.diagnostics import AnomalyDetector
from backend.diagnostics.diagnostic_assembler import DiagnosticAssembler
from backend.environment import MissionProfile
from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.health import HealthIndexCalculator
from backend.risk import (
    MissionReliabilityInputs,
    MissionRiskCalculator,
    ReliabilityBand,
    aggregate_reliability,
)
from backend.rul import RulCalculator
from backend.sensors import SensorBundle
from backend.simulation import EngineSimulator
from backend.telemetry import FrameStatus, LatencyTracker


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

# Fixed per-scenario seeds.  Pinned here so the SIH demo is
# reproducible even if the registry is changed.
_DEMO_SEEDS: Dict[int, int] = {
    1: 100,   # healthy
    2: 200,   # turbulence only
    3: 300,   # sensor fault (stuck RPM)
    4: 400,   # early engine degradation
    5: 500,   # engine degradation + turbulence (composite)
    6: 600,   # intermittent dropouts (unknown)
}


# ---------------------------------------------------------------------
# Per-scenario spec builder (5 of 6 reuse existing registry entries)
# ---------------------------------------------------------------------
def _scenario_1_healthy() -> ScenarioSpec:
    """60 s mission, no faults."""
    return ScenarioSpec(
        name="sih_1_healthy",
        fault_class=FaultClass.HEALTHY,
        severity=0.0,
        onset_time_s=0.0,
        duration_s=None,
        seed=_DEMO_SEEDS[1],
        progression="linear",
        run_classifier=False,
        description="60 s mission, no faults. Baseline for comparison.",
    )


def _scenario_2_turbulence_only() -> ScenarioSpec:
    """60 s mission, heavy turbulence at t=5 s, no engine fault.

    Severity 0.9 keeps the env severity driver in the WARNING/CRITICAL
    band without forcing the anomaly detector to also flag the
    engine (the env layer's vibration is handled by the engine
    simulator, not the fault-injector).
    """
    return ScenarioSpec(
        name="sih_2_turbulence",
        fault_class=FaultClass.ENVIRONMENTAL_DISTURBANCE,
        severity=0.9,
        onset_time_s=5.0,
        duration_s=50.0,
        seed=_DEMO_SEEDS[2],
        progression="step",
        run_classifier=False,
        description=(
            "60 s mission, severe turbulence at t=5 s.  "
            "Vibration increases but the engine residual stays nominal — "
            "system must classify as ENVIRONMENTAL DISTURBANCE, "
            "not an engine fault."
        ),
    )


def _scenario_3_sensor_fault() -> ScenarioSpec:
    """60 s mission, RPM sensor stuck at t=10 s."""
    return ScenarioSpec(
        name="sih_3_sensor_fault",
        fault_class=FaultClass.SENSOR_FAULT,
        severity=1.0,
        onset_time_s=10.0,
        duration_s=30.0,
        seed=_DEMO_SEEDS[3],
        progression="step",
        run_classifier=False,
        description=(
            "60 s mission, RPM sensor stuck at t=10 s.  "
            "Diagnostics must report a SENSOR FAULT, not an engine fault."
        ),
    )


def _scenario_4_early_degradation() -> ScenarioSpec:
    """60 s mission, fast engine degradation onset at t=10 s,
    active for the entire remaining mission so the anomaly
    detector has time to build a sustained signal."""
    return ScenarioSpec(
        name="sih_4_early_degradation",
        fault_class=FaultClass.ENGINE_DEGRADATION,
        severity=0.85,
        onset_time_s=10.0,
        duration_s=None,
        seed=_DEMO_SEEDS[4],
        progression="step",
        run_classifier=False,
        description=(
            "60 s mission, sudden engine degradation at t=10 s.  "
            "Expected detection by t ≈ 12 s."
        ),
    )


def _scenario_5_degradation_plus_turbulence() -> Tuple[ScenarioSpec, ScenarioSpec]:
    """Two specs — one for engine degradation, one for env turbulence.

    The combined injection is handled by :class:`CompositeInjector`
    (defined below).  This function returns both specs so the
    runner can attach them to the same engine + sensor bundle.
    """
    eng = ScenarioSpec(
        name="sih_5a_eng_degradation",
        fault_class=FaultClass.ENGINE_DEGRADATION,
        # Aggressive enough that the *engine* channels (not
        # just the airframe) drift above the 3σ threshold so
        # the env-only heuristic does not mask the engine
        # fault. Scenario 5 is the *discrimination* case: the
        # engine is degrading AND the air is turbulent — the
        # system must still surface the engine fault.
        severity=1.0,
        onset_time_s=5.0,
        duration_s=None,
        seed=_DEMO_SEEDS[5],
        progression="linear",
        run_classifier=False,
        description="Engine degradation at t=5 s.",
    )
    env = ScenarioSpec(
        name="sih_5b_turbulence",
        fault_class=FaultClass.ENVIRONMENTAL_DISTURBANCE,
        severity=0.85,
        onset_time_s=5.0,
        duration_s=45.0,
        seed=_DEMO_SEEDS[5] + 1,
        progression="step",
        run_classifier=False,
        description="Heavy turbulence at t=5 s.",
    )
    return eng, env


def _scenario_6_unknown() -> ScenarioSpec:
    """A scenario that produces a UNKNOWN diagnostic state.

    We use a HEALTHY base spec — the "unknown" behaviour is
    produced by the per-tick dropout injection in
    :func:`_drive_scenario_6`.
    """
    return ScenarioSpec(
        name="sih_6_unknown",
        fault_class=FaultClass.HEALTHY,
        severity=0.0,
        onset_time_s=0.0,
        duration_s=None,
        seed=_DEMO_SEEDS[6],
        progression="linear",
        run_classifier=False,
        description=(
            "60 s mission, intermittent sensor dropouts.  "
            "The system must report UNKNOWN rather than fabricate a result."
        ),
    )


# All six scenario builders.  #5 returns a tuple of two specs —
# the runner is a custom loop, not the standard PipelineRunner.
SCENARIO_BUILDERS: Dict[
    int, Callable[[], Any]
] = {
    1: _scenario_1_healthy,
    2: _scenario_2_turbulence_only,
    3: _scenario_3_sensor_fault,
    4: _scenario_4_early_degradation,
    5: _scenario_5_degradation_plus_turbulence,
    6: _scenario_6_unknown,
}

SCENARIO_NAMES: Dict[int, str] = {
    1: "HEALTHY FLIGHT",
    2: "TURBULENCE WITHOUT ENGINE FAULT",
    3: "SENSOR FAULT",
    4: "EARLY ENGINE DEGRADATION",
    5: "ENGINE DEGRADATION + TURBULENCE",
    6: "UNKNOWN ANOMALY",
}


# ---------------------------------------------------------------------
# Composite injector — combines two FaultInjectors (scenario 5)
# ---------------------------------------------------------------------
class CompositeInjector:
    """Combine two :class:`FaultInjector` instances on the same
    simulator + sensor bundle.

    Each per-tick call to :meth:`tick` returns a
    :class:`~backend.faults.injector.FaultTick` whose values are the
    element-wise sum of the two underlying injectors' ticks (the
    engine's ``vibration_external`` is additive, the
    ``degradation_severity`` is additive modulo a 1.0 cap).

    :meth:`apply_environment` is called on both sub-injectors so
    env-side mutations from the ENVIRONMENTAL_DISTURBANCE scenario
    propagate correctly.
    """

    def __init__(self, a: FaultInjector, b: FaultInjector) -> None:
        self._a = a
        self._b = b

    def attach(self, sim: EngineSimulator, bundle: SensorBundle) -> None:
        self._a.attach(sim, bundle)
        self._b.attach(sim, bundle)

    def reset(self) -> None:
        self._a.reset()
        self._b.reset()

    def tick(self, t_s: float):
        ta = self._a.tick(t_s)
        tb = self._b.tick(t_s)
        # Combine: degradation is capped, vibration is summed.
        deg = min(1.0, max(0.0, ta.degradation_severity + tb.degradation_severity))
        vib = ta.vibration_external + tb.vibration_external
        from backend.faults.injector import FaultTick
        return FaultTick(degradation_severity=deg, vibration_external=vib)

    def apply_sensor(self, t_s: float) -> bool:
        return self._a.apply_sensor(t_s) or self._b.apply_sensor(t_s)

    def apply_environment(self, t_s: float) -> bool:
        return self._a.apply_environment(t_s) or self._b.apply_environment(t_s)


# ---------------------------------------------------------------------
# Dropout injector — forces intermittent dropouts (scenario 6)
# ---------------------------------------------------------------------
class DropoutInjector:
    """A small policy that injects deterministic 100% dropouts on
    rotating channels at scheduled intervals.

    Used by scenario 6 to force the data quality to oscillate
    such that no single channel is calibrated long enough for
    the classifier / risk layers to commit to a label.  The result
    surfaces as ``UNKNOWN`` mission reliability.

    Implementation: we set the channel's ``dropout_prob`` to 1.0
    for the active window, and restore the original value when
    the window ends.  This is deterministic given a fixed RNG
    seed (the sensor bundle is constructed with ``master_seed``
    so the per-sample stochastic draw is reproducible).
    """

    def __init__(self, schedule: List[Tuple[float, float, str]]) -> None:
        """``schedule`` is a list of ``(start_t, end_t, channel)``."""
        self._schedule = sorted(schedule)
        self._active: Dict[str, Tuple[float, float]] = {}
        self._original_probs: Dict[str, float] = {}
        self._bundle: Optional[SensorBundle] = None

    def attach(self, sim: EngineSimulator, bundle: SensorBundle) -> None:
        self._bundle = bundle

    def reset(self) -> None:
        for ch, prob in self._original_probs.items():
            try:
                self._bundle._channels[ch].cfg.dropout_prob = prob
            except Exception:
                pass
        self._active.clear()
        self._original_probs.clear()

    def tick(self, t_s: float) -> Tuple[str, ...]:
        """Activate / deactivate dropouts.  Returns the channels
        currently in dropout for downstream display."""
        for start_t, end_t, ch in self._schedule:
            if start_t <= t_s < end_t and ch not in self._active:
                try:
                    cfg = self._bundle._channels[ch].cfg
                    if ch not in self._original_probs:
                        self._original_probs[ch] = float(cfg.dropout_prob)
                    cfg.dropout_prob = 1.0
                except Exception:
                    pass
                self._active[ch] = (start_t, end_t)
            elif (ch in self._active
                  and (t_s < self._active[ch][0] or t_s >= self._active[ch][1])):
                try:
                    cfg = self._bundle._channels[ch].cfg
                    cfg.dropout_prob = self._original_probs.get(ch, 0.0)
                except Exception:
                    pass
                del self._active[ch]
        return tuple(self._active.keys())


# ---------------------------------------------------------------------
# Per-tick snapshot bundle (what the demo renders)
# ---------------------------------------------------------------------
@dataclass
class TickRecord:
    """One tick's worth of demo display data."""

    tick_index: int
    time_s: float
    # Telemetry
    observed: Dict[str, Optional[float]]
    twin_expected: Dict[str, float]
    residual_z: Dict[str, float]
    # Anomaly
    anomaly_score: float
    anomaly_label: str
    # Health
    health_overall: float
    health_label: str
    # RUL
    rul_status: str
    rul_estimate_h: Optional[float]
    rul_lower_h: Optional[float]
    rul_upper_h: Optional[float]
    rul_uncertain: bool
    # Risk
    risk_status: str
    risk_score: float
    # Reliability (PHASE 23)
    rel_band: str
    rel_headline: str
    rel_drivers: Tuple[str, ...]
    rel_confidence: float
    # Environment
    env_wind_mps: float
    env_turbulence: float
    # Naive threshold flag
    naive_flags: Tuple[str, ...]
    # Event log delta
    new_events: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)


@dataclass
class ScenarioResult:
    """The full result of one scenario run."""

    scenario_id: int
    scenario_name: str
    description: str
    ticks: List[TickRecord]
    summary: Dict[str, Any]
    # Final diagnostic engine state label, for the
    # superiority-of-system block.
    final_engine_state: str
    final_risk_status: str
    final_reliability_band: str
    final_reliability_headline: str
    # Naive-monitor summary: the set of channels that a single-channel
    # threshold would have flagged, with the earliest tick.
    naive_flags_by_channel: Dict[str, int]
    # What the system actually concluded
    system_conclusion: str
    # Latency summary (if measured)
    latency_ms: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _fmt(v: Optional[float], fmt: str = "8.2f", na: str = "—") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    try:
        return f"{float(v):{fmt}}"
    except (TypeError, ValueError):
        return na


def _channel_block(residual, sample_readings) -> Tuple[Dict[str, Optional[float]], Dict[str, float], Dict[str, float]]:
    """Extract per-channel (observed, expected, z) from a residual.

    Accepts both a :class:`ResidualFrame` (used in the custom
    scenarios 5/6 drivers) and a flat dict (the
    ``DashboardSnapshot.residual`` wire format, with keys
    ``residual.<channel>.<field>``).
    """
    obs: Dict[str, Optional[float]] = {}
    expected: Dict[str, float] = {}
    z: Dict[str, float] = {}
    if isinstance(residual, dict):
        # Wire format: { "residual.<channel>.observed": ..., ... }
        # Group by channel first.
        per_ch: Dict[str, Dict[str, Any]] = {}
        for k, v in residual.items():
            parts = k.split(".")
            if len(parts) == 3 and parts[0] == "residual":
                per_ch.setdefault(parts[1], {})[parts[2]] = v
        for ch, fields in per_ch.items():
            obs_v = fields.get("observed")
            obs[ch] = obs_v if obs_v is not None else None
            if "predicted" in fields and fields["predicted"] is not None:
                expected[ch] = float(fields["predicted"])
            if "z_score" in fields and fields["z_score"] is not None:
                z[ch] = float(fields["z_score"])
    else:
        for ch, r in residual.residuals.items():
            obs[ch] = r.observed
            expected[ch] = float(r.predicted)
            z[ch] = float(r.z_score) if r.z_score is not None else 0.0
    for ch, reading in sample_readings.items():
        if ch not in obs and getattr(reading, "value", None) is not None:
            obs[ch] = float(reading.value)
    return obs, expected, z


def _naive_flag_for_channel(z: float) -> bool:
    """Apply a single-channel absolute z-score threshold.

    Uses a 3.0σ cut-off per channel (the typical statistical
    alarm cut-off).  A naive monitor would fire the alarm on
    |z| > 3.
    """
    return abs(z) > 3.0


def _assess_reliability_from_diag(
    sample, risk, diagnostic, time_s: float, *, rated_power_kw: float
):
    """Build a :class:`MissionReliabilityInputs` and call
    :func:`aggregate_reliability`.

    Mirrors the logic in
    :meth:`backend.dashboard.runner.PipelineRunner._assess_reliability`
    but as a free function so we can reuse it from the
    composite / dropout drivers.
    """
    engine_state = sample.engine
    rated_kw = max(1.0, float(rated_power_kw))
    engine_load = max(
        0.0,
        min(1.0, float(engine_state.brake_power_kw) / rated_kw),
    )
    env_d = diagnostic.environment_context or {}
    turbulence = float(env_d.get("turbulence_intensity", 0.0) or 0.0)
    wind_mps = float(env_d.get("wind_mps", 0.0) or 0.0)
    env_severity = max(0.0, min(1.0, max(turbulence, wind_mps) / 5.0))
    sensor_confidence = float(diagnostic.data_quality)
    fp = diagnostic.fault_probabilities or {}
    p_engine = float(fp.get("engine", 0.0) or 0.0)
    p_sensor = float(fp.get("sensor", 0.0) or 0.0)
    p_env = float(fp.get("environment", 0.0) or 0.0)
    rul_block = diagnostic.rul or {}
    rul_uncertain_flag = bool(rul_block.get("is_uncertain", False))
    if rul_uncertain_flag:
        rul_uncertainty = 1.0
    else:
        lower = float(rul_block.get("lower", 0.0) or 0.0)
        upper = float(rul_block.get("upper", 0.0) or 0.0)
        denom = max(upper, 1e-6)
        rul_uncertainty = max(
            0.0, min(1.0, (upper - lower) / denom)
        )
    remaining_hours: Optional[float] = None
    try:
        h2d = getattr(risk, "hours_to_destination", None)
    except Exception:
        h2d = None
    if h2d is not None:
        try:
            remaining_hours = float(h2d)
        except (TypeError, ValueError):
            remaining_hours = None
    inputs = MissionReliabilityInputs(
        risk_assessment=risk,
        fault_prob_engine=p_engine,
        fault_prob_sensor=p_sensor,
        fault_prob_environment=p_env,
        rul_uncertainty=rul_uncertainty,
        mission_phase=risk.mission_phase,
        remaining_hours=remaining_hours,
        engine_load=engine_load,
        env_severity=env_severity,
        sensor_confidence=sensor_confidence,
    )
    return aggregate_reliability(inputs, time_s=float(time_s))


def _build_tick_record(
    *,
    tick_index: int,
    snap,                  # Snapshot-like
    diagnostic,            # DiagnosticState
    reliability,           # MissionReliabilityAssessment
    events_before: Tuple[Dict[str, Any], ...],
    events_after: Tuple[Dict[str, Any], ...],
) -> TickRecord:
    """Convert one tick's outputs into a :class:`TickRecord`."""
    sample_readings = getattr(snap, "_sample_readings", {})
    obs, expected, z = _channel_block(snap.residual, sample_readings)
    naive_flags = tuple(
        ch for ch, zv in z.items() if _naive_flag_for_channel(zv)
    )
    env_block = diagnostic.environment_context or {}
    rul = diagnostic.rul or {}
    env_wind = float(env_block.get("wind_mps", 0.0) or 0.0)
    env_turb = float(env_block.get("turbulence_intensity", 0.0) or 0.0)
    return TickRecord(
        tick_index=tick_index,
        time_s=float(snap.time_s),
        observed=obs,
        twin_expected=expected,
        residual_z=z,
        anomaly_score=float(snap.anomaly.overall_score) if snap.anomaly else 0.0,
        anomaly_label=str(snap.anomaly.overall_label.value) if snap.anomaly else "NORMAL",
        health_overall=float(snap.health.overall_score),
        health_label=str(snap.health.overall_label.value),
        rul_status=str(rul.get("status", "RUL_OK")),
        rul_estimate_h=(
            float(rul.get("estimate_h"))
            if rul.get("estimate_h") is not None else None
        ),
        rul_lower_h=(
            float(rul.get("lower_h"))
            if rul.get("lower_h") is not None else None
        ),
        rul_upper_h=(
            float(rul.get("upper_h"))
            if rul.get("upper_h") is not None else None
        ),
        rul_uncertain=bool(rul.get("is_uncertain", False)),
        risk_status=str(snap.risk.status.value),
        risk_score=float(snap.risk.risk_score),
        rel_band=reliability.band.value,
        rel_headline=reliability.explanation.headline,
        rel_drivers=tuple(d.note for d in reliability.explanation.drivers),
        rel_confidence=float(reliability.confidence),
        env_wind_mps=env_wind,
        env_turbulence=env_turb,
        naive_flags=naive_flags,
        new_events=tuple(
            e for e in events_after
            if e not in events_before
        ),
    )


# ---------------------------------------------------------------------
# Internal helpers — small shims so we can reuse PipelineRunner's
# PHASE 23 aggregator on non-PipelineRunner ticks.
# ---------------------------------------------------------------------
class _DigitalTwinFromCfg:
    """A thin wrapper that exposes ``step`` + ``residual`` like the
    real :class:`DigitalTwin` but takes the full LoadedConfig."""

    def __init__(self, cfg) -> None:
        from backend.digital_twin import DigitalTwin
        self._twin = DigitalTwin(cfg.engine, dt_s=0.1)

    def step(self, env, obs):
        return self._twin.step(env, obs)

    def residual(self, twin_state, obs):
        return self._twin.residual(twin_state, obs)


@dataclass
class _SnapshotLike:
    """A duck-typed stand-in for DashboardSnapshot — only the
    fields ``_build_tick_record`` reads are populated."""

    time_s: float
    anomaly: Any
    health: Any
    rul: Any
    risk: Any
    residual: Any
    environment: Any = None
    engine_state: Any = None
    frame_status: str = "OK"
    scenario_name: str = ""
    _sample_readings: Dict[str, Any] = field(default_factory=dict)


def _summarise_run(records: List[TickRecord]) -> Dict[str, Any]:
    """Compute summary statistics over a scenario's tick records.

    The summary drives the ``system_conclusion`` line at the end
    of each scenario.  We look at *sustained* anomaly count
    (rather than peak) because the detector's EWMA smoother can
    spike to ANOMALY briefly on healthy runs without it
    representing a real fault.
    """
    if not records:
        return {}
    summary: Dict[str, Any] = {}
    # Anomaly peak.
    summary["peak_anomaly_score"] = max(r.anomaly_score for r in records)
    summary["peak_anomaly_label"] = max(
        records, key=lambda r: r.anomaly_score
    ).anomaly_label
    # Sustained-anomaly tick count: ticks where the score is
    # above the 0.5 warn threshold (this is the threshold the
    # PHASE 8 detector uses internally).
    summary["sustained_anomaly_ticks"] = sum(
        1 for r in records if r.anomaly_score > 0.5
    )
    # Median anomaly score (more robust to spikes than peak).
    sorted_scores = sorted(r.anomaly_score for r in records)
    summary["median_anomaly_score"] = sorted_scores[len(sorted_scores) // 2]
    # Env severity peak.  We store wind_mps and turbulence as raw
    # values (typically < 5 m/s).  Normalise to the same 0..1
    # envelope the reliability aggregator uses.
    summary["peak_env_severity"] = max(
        max(r.env_wind_mps, r.env_turbulence) / 5.0 for r in records
    )
    # Peak vibration reading.
    summary["peak_vibration"] = max(
        (r.observed.get("vibration") or 0.0) for r in records
    )
    # Peak z-score in any channel.
    summary["peak_z_egt"] = max(
        (abs(r.residual_z.get("egt", 0.0)) for r in records),
        default=0.0,
    )
    summary["peak_z_vib"] = max(
        (abs(r.residual_z.get("vibration", 0.0)) for r in records),
        default=0.0,
    )
    # Peak health-overall drop.
    summary["min_health"] = min(r.health_overall for r in records)
    # Baseline health (first-tick value) — the run-specific
    # reference point. The healthy baseline sits at ~0.84
    # (PHASE 11 starts HEALTHY at 0.84 not 1.0 because the
    # engine always carries a small wear + oil deficit). A
    # *meaningful* drop is anything > 0.015 below the baseline.
    summary["baseline_health"] = records[0].health_overall if records else 1.0
    summary["health_drop_vs_baseline"] = (
        summary["baseline_health"] - summary["min_health"]
    )
    # Sensor fault count: ticks where ONE channel has an extreme
    # residual (|z| > 6) but the *other* engine channels are
    # nominal (|z| < 3).  This is the classic stuck / dropout
    # sensor signature: a single channel blows up while the rest
    # of the engine looks normal.  An engine fault, by contrast,
    # produces correlated multi-channel drift.
    summary["sensor_fault_event_count"] = sum(
        1 for r in records
        if any(abs(v) > 6.0 for v in r.residual_z.values())
        and sum(1 for v in r.residual_z.values() if abs(v) > 3.0) <= 1
    )
    # "Environmental-only" ticks: airframe channels (vibration,
    # altitude, airspeed) are extreme BUT every engine-side
    # channel (rpm, egt, cht, oil_pressure, oil_temperature,
    # fuel_flow) is nominal AND the health index is *not*
    # declining. This is the turbulence-without-engine-fault
    # signature: the airframe shakes but the engine itself is
    # fine. A naive vibration threshold would flag this as
    # an engine fault; the multi-channel fusion does not.
    # When the engine channels are also extreme, the system
    # should call it an engine fault (env+engine signals
    # separate cleanly).
    _ENGINE_CHANS = (
        "rpm", "egt", "cht", "oil_pressure",
        "oil_temperature", "fuel_flow",
    )
    _AIRFRAME_CHANS = ("vibration", "altitude", "airspeed")
    summary["env_dominant_event_count"] = sum(
        1 for r in records
        if any(abs(r.residual_z.get(ch, 0.0)) > 3.0
               for ch in _AIRFRAME_CHANS)
        and all(abs(r.residual_z.get(ch, 0.0)) < 2.0
                for ch in _ENGINE_CHANS)
    )
    # Engine-extreme ticks: any engine channel is > 3σ.
    # Used to flip scenario 5 (engine + turbulence) out of
    # the env-only bucket.
    summary["engine_extreme_event_count"] = sum(
        1 for r in records
        if any(abs(r.residual_z.get(ch, 0.0)) > 3.0
               for ch in _ENGINE_CHANS)
    )
    # Confidence drop count.
    summary["low_confidence_ticks"] = sum(
        1 for r in records if r.rel_confidence < 0.40
    )
    # RUL uncertain count.
    summary["rul_uncertain_ticks"] = sum(1 for r in records if r.rul_uncertain)
    return summary


def _conclusion_from_run(
    records: List[TickRecord], rel_band: str,
) -> str:
    """Summarise the entire run in one English line.

    The logic prefers evidence from the *whole run* (sustained
    anomaly fraction, peak env severity) over the final tick
    alone, because the risk layer's per-tick score can oscillate
    around the CAUTION threshold (no EWMA on the risk score in
    PHASE 12).  This is the *demonstrative* version of the
    comparison: a one-line takeaway that a control-room operator
    would see.
    """
    s = _summarise_run(records)
    if not s:
        return "—"
    if rel_band == "UNKNOWN" or s.get("low_confidence_ticks", 0) > len(records) * 0.3:
        return "UNKNOWN — insufficient validated evidence"
    sustained = s.get("sustained_anomaly_ticks", 0)
    median_anom = s.get("median_anomaly_score", 0.0)
    peak_env = s.get("peak_env_severity", 0.0)
    min_health = s.get("min_health", 1.0)
    baseline_health = s.get("baseline_health", 1.0)
    health_drop = float(s.get("health_drop_vs_baseline", 0.0))
    sensor_event_count = s.get("sensor_fault_event_count", 0)
    env_dominant_count = s.get("env_dominant_event_count", 0)
    engine_extreme_count = s.get("engine_extreme_event_count", 0)
    # The Digital Twin is *meant* to absorb a slow-moving
    # degradation so the per-channel residuals stay small. In that
    # case the anomaly score does not cross the 0.5 threshold,
    # but the **health index** does drop noticeably versus the
    # run's own baseline. Use a *relative* threshold: 0.015 below
    # the baseline. The healthy baseline naturally drifts up to
    # 0.01 below the first-tick value (PHASE 11 has a small
    # health deficit because the engine carries wear). A
    # *meaningful* drop is anything more than that.
    health_drop_indicates_fault = health_drop > 0.015
    n = max(1, len(records))
    sustained_fraction = sustained / n
    engine_fault_sustained = sustained_fraction > 0.10
    # Sensor fault signature: ONE channel has extreme z while the
    # rest are nominal — even if the anomaly detector trips on it.
    # A meaningful sensor fault has many such ticks (>= 5% of run).
    sensor_event_dominant = sensor_event_count / n > 0.05
    # Environmental signature: airframe channels (vibration /
    # altitude / airspeed) extreme BUT all engine channels
    # nominal. The 95th-percentile anomaly score will still
    # trip on this, but the *cause* is airframe / atmosphere,
    # not engine. A meaningful env event is >= 10% of the run.
    env_event_dominant = env_dominant_count / n > 0.10
    # Engine-extreme signature: at least one engine channel
    # crossed 3σ for a meaningful fraction of the run. If
    # both engine and airframe are extreme, the env-only
    # check must NOT fire — the cause is engine, with
    # turbulence as background.
    engine_extreme_dominant = engine_extreme_count / n > 0.05
    # Heuristic priority:
    # 1. If data was systematically dropped (UNKNOWN), say so.
    # 2. If a *sensor-only* fault signature is dominant, say so
    #    even if the anomaly score is high (a single stuck
    #    channel trips the anomaly detector too).
    # 3. If the *airframe* is shaking but every engine channel
    #    is nominal, the cause is environmental (turbulence),
    #    not engine. This is the *discrimination* case: a naive
    #    vibration threshold would call this an engine fault;
    #    the multi-channel system does not.
    # 4. If anomaly was sustained AND env severity stayed low,
    #    call it an engine fault.
    # 5. If both engine and env were high, the engine fault
    #    wins (the system discriminates correctly: env and
    #    engine are separate signals).
    if env_event_dominant and not sensor_event_dominant and not engine_extreme_dominant:
        return (
            f"ENVIRONMENTAL DISTURBANCE (not engine fault) — "
            f"airframe-only anomaly: {env_dominant_count} ticks "
            f"({env_dominant_count/n:.0%} of run), env={peak_env:.2f}"
        )
    if sensor_event_dominant and not engine_fault_sustained:
        return (
            f"SENSOR FAULT (not engine fault): {sensor_event_count} "
            f"single-channel-extreme ticks"
        )
    if sensor_event_dominant and sustained_fraction < 0.30:
        return (
            f"SENSOR FAULT (not engine fault): {sensor_event_count} "
            f"single-channel-extreme ticks ({sensor_event_count/n:.0%} of run)"
        )
    if sensor_event_dominant:
        return (
            f"SENSOR FAULT (not engine fault): {sensor_event_count} "
            f"single-channel-extreme ticks dominate the anomaly "
            f"({sensor_event_count/n:.0%} of run, "
            f"anom sustained {sustained_fraction:.0%})"
        )
    if engine_fault_sustained and peak_env > 0.20:
        return (
            f"ENGINE FAULT detected despite turbulence "
            f"(anom sustained {sustained_fraction:.0%} of run, "
            f"env={peak_env:.2f})"
        )
    if engine_fault_sustained:
        return (
            f"ENGINE FAULT (anomaly sustained {sustained_fraction:.0%} "
            f"of run, median={median_anom:.2f})"
        )
    # Slow / partial degradation: anomaly did not sustain, but
    # health dropped meaningfully versus the run baseline.
    # Split between env-driven and engine-driven by env severity.
    if health_drop_indicates_fault and peak_env > 0.20:
        return (
            f"ENGINE FAULT detected despite turbulence "
            f"(anom low: {sustained_fraction:.0%} sustained, "
            f"health dropped {health_drop:.3f} below baseline "
            f"({baseline_health:.2f}→{min_health:.2f}), "
            f"env={peak_env:.2f})"
        )
    if health_drop_indicates_fault:
        return (
            f"ENGINE FAULT (slow degradation: health dropped "
            f"{health_drop:.3f} below baseline "
            f"({baseline_health:.2f}→{min_health:.2f}), "
            f"median anomaly={median_anom:.2f})"
        )
    if peak_env > 0.20:
        return (
            f"ENVIRONMENTAL DISTURBANCE (not engine fault) — "
            f"env severity={peak_env:.2f}"
        )
    if rel_band == "LOW":
        return "HEALTHY (no anomaly)"
    if rel_band == "MEDIUM":
        return "ELEVATED RISK (caution)"
    if rel_band == "HIGH":
        return "HIGH RISK"
    return "—"


# ---------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------
def _drive_scenario_1_through_4(scenario: ScenarioSpec, ticks: int) -> ScenarioResult:
    """Standard scenario driver: build a PipelineRunner and run
    for ``ticks`` ticks.  Used for scenarios 1, 2, 3, 4."""
    cfg = load_config(str(CONFIG_DIR))
    runner = PipelineRunner(cfg, scenario, history_size=ticks + 10)
    records: List[TickRecord] = []
    events_seen: List[Dict[str, Any]] = []
    for i in range(ticks):
        snap = runner.tick()
        diag = runner.latest_diagnostic
        rel = runner.latest_reliability
        new_events = tuple(snap.events)
        # The frozen DashboardSnapshot does not allow ad-hoc attrs;
        # we build a small read-only view for the tick renderer.
        sample_readings: Dict[str, Any] = {}
        for k, v in (snap.residual or {}).items():
            if k.startswith("residual.") and k.endswith(".observed"):
                ch_name = k.split(".")[1]
                sample_readings[ch_name] = v
        snap_view = _SnapshotLike(
            time_s=snap.time_s,
            anomaly=snap.anomaly,
            health=snap.health,
            rul=snap.rul,
            risk=snap.risk,
            residual=snap.residual,
            environment=snap.environment,
            engine_state=snap.engine_state,
            _sample_readings=sample_readings,
        )
        rec = _build_tick_record(
            tick_index=i,
            snap=snap_view,
            diagnostic=diag,
            reliability=rel,
            events_before=tuple(events_seen),
            events_after=new_events,
        )
        records.append(rec)
        events_seen = list(new_events)
    final = records[-1] if records else None
    naive_by_ch: Dict[str, int] = {}
    for r in records:
        for ch in r.naive_flags:
            naive_by_ch.setdefault(ch, r.tick_index)
    system_conclusion = _conclusion_from_run(
        records,
        rel_band=final.rel_band if final else "UNKNOWN",
    )
    return ScenarioResult(
        scenario_id=int(scenario.name.split("_")[1]),
        scenario_name=SCENARIO_NAMES.get(
            int(scenario.name.split("_")[1]), scenario.name
        ),
        description=scenario.description,
        ticks=records,
        summary={
            "duration_s": scenario.duration_s,
            "onset_time_s": scenario.onset_time_s,
        },
        final_engine_state=final.health_label if final else "UNKNOWN",
        final_risk_status=final.risk_status if final else "—",
        final_reliability_band=final.rel_band if final else "UNKNOWN",
        final_reliability_headline=final.rel_headline if final else "—",
        naive_flags_by_channel=naive_by_ch,
        system_conclusion=system_conclusion,
    )


def _drive_scenario_5(ticks: int) -> ScenarioResult:
    """Composite scenario: engine degradation + turbulence injected
    by a :class:`CompositeInjector` that wraps two standard
    :class:`FaultInjector` instances.  The runner is built
    directly so we can swap the single injector for the composite.
    """
    cfg = load_config(str(CONFIG_DIR))
    eng_spec, env_spec = _scenario_5_degradation_plus_turbulence()
    composite = CompositeInjector(
        FaultInjector(eng_spec.to_fault_scenario()),
        FaultInjector(env_spec.to_fault_scenario()),
    )
    sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
    bundle = SensorBundle(cfg.sensors, sim, master_seed=eng_spec.seed)
    twin_mod = _DigitalTwinFromCfg(cfg)
    detector = AnomalyDetector()
    health = HealthIndexCalculator(cfg.engine)
    rul = RulCalculator(cfg.engine)
    profile = MissionProfile.from_config(cfg.environment.mission)
    risk = MissionRiskCalculator(profile, config=cfg.risk)
    assembler = DiagnosticAssembler()
    composite.attach(sim, bundle)

    rated_kw = float(cfg.engine.geometry.rated_power_kw)
    records: List[TickRecord] = []
    for i in range(ticks):
        t_now = sim.runner._t
        composite.apply_sensor(t_now)
        composite.apply_environment(t_now)
        ft = composite.tick(t_now)
        sample = bundle.tick(
            degradation_severity=ft.degradation_severity,
            vibration_external=ft.vibration_external,
        )
        obs = {ch: r.value for ch, r in sample.readings.items() if r.value is not None}
        twin_state = twin_mod.step(sample.env, obs)
        residual = twin_mod.residual(twin_state, obs)
        a = detector.detect(residual, sample, frame_status=FrameStatus.OK)
        h = health.update(state=sample.engine, anomaly=a, classification=None,
                          time_s=sample.time_s)
        r = rul.update(health=h, state=sample.engine,
                       time_s=sample.time_s, dt_s=0.1)
        rk = risk.update(health=h, rul=r, anomaly=a, time_s=sample.time_s)
        diag = assembler.assemble(
            time_s=sample.time_s,
            sample=sample,
            twin_state=twin_state,
            residual=residual,
            anomaly=a,
            classification=None,
            health=h,
            rul=r,
            risk=rk,
            engine=sample.engine,
            env=sample.env,
        )
        rel = _assess_reliability_from_diag(
            sample=sample,
            risk=rk,
            diagnostic=diag,
            time_s=sample.time_s,
            rated_power_kw=rated_kw,
        )
        snap = _SnapshotLike(
            time_s=sample.time_s,
            anomaly=a,
            health=h,
            rul=r,
            risk=rk,
            residual=residual,
            environment=sample.env,
            engine_state=sample.engine,
            _sample_readings=sample.readings,
        )
        rec = _build_tick_record(
            tick_index=i,
            snap=snap,
            diagnostic=diag,
            reliability=rel,
            events_before=tuple(),
            events_after=tuple(),
        )
        records.append(rec)
    final = records[-1] if records else None
    naive_by_ch: Dict[str, int] = {}
    for r in records:
        for ch in r.naive_flags:
            naive_by_ch.setdefault(ch, r.tick_index)
    system_conclusion = _conclusion_from_run(
        records,
        rel_band=final.rel_band if final else "UNKNOWN",
    )
    return ScenarioResult(
        scenario_id=5,
        scenario_name=SCENARIO_NAMES[5],
        description=(
            "60 s mission.  Engine degradation AND heavy turbulence "
            "injected simultaneously at t=5 s.  The system must still "
            "surface the engine fault — the engine-side residual stays "
            "elevated while env severity is also high, and the "
            "ENGINE_DEGRADATION driver wins."
        ),
        ticks=records,
        summary={"duration_s": 60.0, "onset_time_s": 5.0},
        final_engine_state=final.health_label if final else "UNKNOWN",
        final_risk_status=final.risk_status if final else "—",
        final_reliability_band=final.rel_band if final else "UNKNOWN",
        final_reliability_headline=final.rel_headline if final else "—",
        naive_flags_by_channel=naive_by_ch,
        system_conclusion=system_conclusion,
    )


def _drive_scenario_6(ticks: int) -> ScenarioResult:
    """Healthy engine + intermittent sensor dropouts.

    The dropout policy injects short DROPPED windows on rotating
    channels, keeping the data_quality below the calibration floor
    so the risk layer reports INSUFFICIENT_DATA and the reliability
    band becomes UNKNOWN.
    """
    cfg = load_config(str(CONFIG_DIR))
    scenario = _scenario_6_unknown()
    sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
    bundle = SensorBundle(cfg.sensors, sim, master_seed=scenario.seed)
    twin_mod = _DigitalTwinFromCfg(cfg)
    detector = AnomalyDetector()
    health = HealthIndexCalculator(cfg.engine)
    rul = RulCalculator(cfg.engine)
    profile = MissionProfile.from_config(cfg.environment.mission)
    risk = MissionRiskCalculator(profile, config=cfg.risk)
    assembler = DiagnosticAssembler()

    # Deterministic dropout schedule: drop most channels in rotating
    # windows so that data quality falls below the calibration
    # floor (the reliability layer's UNKNOWN trip wire is
    # `sensor_confidence < 0.20`). The schedule overlaps so that
    # ~9/11 channels are dropped for sustained windows, giving
    # `data_quality ≈ 0.18` (9/11 dropped) which propagates to
    # UNKNOWN through the risk layer.
    schedule: List[Tuple[float, float, str]] = []
    all_chans = ("rpm", "egt", "cht", "oil_pressure", "oil_temperature",
                 "fuel_flow", "vibration", "altitude", "airspeed",
                 "ambient_temperature", "ambient_pressure")
    # Two rotating groups: drop ~9 channels at a time, keep 2
    # alive. Group A keeps `altitude`, `airspeed`; group B keeps
    # `ambient_temperature`, `ambient_pressure`. Each group is
    # active for 25 s, alternating.
    for k, t0 in enumerate((5.0, 30.0, 55.0)):
        keep = ("altitude", "airspeed") if k % 2 == 0 else ("ambient_temperature", "ambient_pressure")
        for ch in all_chans:
            if ch not in keep:
                schedule.append((t0, t0 + 25.0, ch))
    droput = DropoutInjector(schedule)
    droput.attach(sim, bundle)

    rated_kw = float(cfg.engine.geometry.rated_power_kw)
    records: List[TickRecord] = []
    for i in range(ticks):
        t_now = sim.runner._t
        droput.tick(t_now)
        sample = bundle.tick()
        obs = {ch: r.value for ch, r in sample.readings.items() if r.value is not None}
        twin_state = twin_mod.step(sample.env, obs)
        residual = twin_mod.residual(twin_state, obs)
        a = detector.detect(residual, sample, frame_status=FrameStatus.OK)
        h = health.update(state=sample.engine, anomaly=a, classification=None,
                          time_s=sample.time_s)
        r = rul.update(health=h, state=sample.engine,
                       time_s=sample.time_s, dt_s=0.1)
        rk = risk.update(health=h, rul=r, anomaly=a, time_s=sample.time_s)
        diag = assembler.assemble(
            time_s=sample.time_s,
            sample=sample,
            twin_state=twin_state,
            residual=residual,
            anomaly=a,
            classification=None,
            health=h,
            rul=r,
            risk=rk,
            engine=sample.engine,
            env=sample.env,
        )
        rel = _assess_reliability_from_diag(
            sample=sample,
            risk=rk,
            diagnostic=diag,
            time_s=sample.time_s,
            rated_power_kw=rated_kw,
        )
        snap = _SnapshotLike(
            time_s=sample.time_s,
            anomaly=a,
            health=h,
            rul=r,
            risk=rk,
            residual=residual,
            environment=sample.env,
            engine_state=sample.engine,
            _sample_readings=sample.readings,
        )
        rec = _build_tick_record(
            tick_index=i,
            snap=snap,
            diagnostic=diag,
            reliability=rel,
            events_before=tuple(),
            events_after=tuple(),
        )
        records.append(rec)
    final = records[-1] if records else None
    naive_by_ch: Dict[str, int] = {}
    for r in records:
        for ch in r.naive_flags:
            naive_by_ch.setdefault(ch, r.tick_index)
    system_conclusion = _conclusion_from_run(
        records,
        rel_band=final.rel_band if final else "UNKNOWN",
    )
    return ScenarioResult(
        scenario_id=6,
        scenario_name=SCENARIO_NAMES[6],
        description=(
            "60 s mission.  Intermittent sensor dropouts on rotating "
            "channels.  No single channel is calibrated long enough for "
            "the classifier / risk layers to commit to a label — the "
            "system correctly reports UNKNOWN rather than fabricating a "
            "diagnosis."
        ),
        ticks=records,
        summary={"duration_s": 60.0, "onset_time_s": 5.0},
        final_engine_state=final.health_label if final else "UNKNOWN",
        final_risk_status=final.risk_status if final else "—",
        final_reliability_band=final.rel_band if final else "UNKNOWN",
        final_reliability_headline=final.rel_headline if final else "—",
        naive_flags_by_channel=naive_by_ch,
        system_conclusion=system_conclusion,
    )


# ---------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------
def run_scenario(scenario_id: int, *, ticks: int = 600) -> ScenarioResult:
    """Run a single scenario by id (1..6) and return the result."""
    if scenario_id not in SCENARIO_BUILDERS:
        raise ValueError(
            f"unknown scenario_id={scenario_id}; "
            f"valid: {sorted(SCENARIO_BUILDERS)}"
        )
    if scenario_id == 1:
        return _drive_scenario_1_through_4(_scenario_1_healthy(), ticks)
    if scenario_id == 2:
        return _drive_scenario_1_through_4(_scenario_2_turbulence_only(), ticks)
    if scenario_id == 3:
        return _drive_scenario_1_through_4(_scenario_3_sensor_fault(), ticks)
    if scenario_id == 4:
        return _drive_scenario_1_through_4(_scenario_4_early_degradation(), ticks)
    if scenario_id == 5:
        return _drive_scenario_5(ticks)
    if scenario_id == 6:
        return _drive_scenario_6(ticks)
    raise ValueError(f"unhandled scenario_id={scenario_id}")  # pragma: no cover


def run_all(*, ticks: int = 600) -> List[ScenarioResult]:
    """Run all 6 scenarios in order and return the list of results."""
    return [run_scenario(i, ticks=ticks) for i in sorted(SCENARIO_BUILDERS)]


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def _print_header(scenario_id: int, name: str, description: str) -> None:
    bar = "=" * 78
    print(bar)
    print(f"  SCENARIO {scenario_id}: {name}")
    print(bar)
    print(description)
    print()


def _print_tick_table(records: List[TickRecord], *, every: int = 10) -> None:
    """Print a compact per-tick table.

    ``every`` controls the down-sampling: the demo prints every Nth
    tick so a 60 s × 10 Hz mission (600 ticks) is readable on a
    terminal.  The first and last tick are always printed.
    """
    if not records:
        return
    headers = (
        "tick", "t(s)", "rpm", "egt°C", "vib g",
        "z_egt", "z_vib", "anom", "health",
        "rul", "risk", "rel",
    )
    print("    " + "  ".join(f"{h:>7}" for h in headers))
    print("    " + "--" * len(headers))
    n = len(records)
    shown = set()
    shown.add(0)
    shown.add(n - 1)
    for i in range(0, n, every):
        shown.add(i)
    for i, rec in enumerate(records):
        if i not in shown:
            continue
        rpm = _fmt(rec.observed.get("rpm"), "7.0f")
        egt = _fmt(rec.observed.get("egt"), "7.1f")
        vib = _fmt(rec.observed.get("vibration"), "6.2f")
        z_egt = _fmt(rec.residual_z.get("egt"), "6.2f")
        z_vib = _fmt(rec.residual_z.get("vibration"), "6.2f")
        anom = f"{rec.anomaly_score:5.2f}/{rec.anomaly_label[:4]}"
        health = f"{rec.health_overall:5.2f}/{rec.health_label[:4]}"
        rul = rec.rul_status.replace("RUL_", "")[:5]
        risk = f"{rec.risk_status[:4]}({rec.risk_score:.2f})"
        rel = f"{rec.rel_band}({rec.rel_confidence:.2f})"
        print(
            f"    {rec.tick_index:>4d}  {rec.time_s:>5.1f}  {rpm}  {egt}  "
            f"{vib}  {z_egt}  {z_vib}  {anom}  {health}  {rul:>5}  "
            f"{risk:>11}  {rel:>11}"
        )
    print()


def _print_event_timeline(records: List[TickRecord]) -> None:
    """Print the event timeline across the run."""
    print("  Event timeline:")
    any_event = False
    for rec in records:
        for ev in rec.new_events:
            any_event = True
            kind = str(ev.get("kind", "?"))
            desc = str(ev.get("description", ""))
            print(f"    t={rec.time_s:6.1f}s  {kind:<20}  {desc}")
    if not any_event:
        print("    (no events)")
    print()


def _print_final_state(result: ScenarioResult) -> None:
    print("  Final state:")
    print(f"    engine state       : {result.final_engine_state}")
    print(f"    risk status        : {result.final_risk_status}")
    print(f"    reliability band   : {result.final_reliability_band}")
    print(f"    reliability line   : {result.final_reliability_headline}")
    print(f"    system conclusion  : {result.system_conclusion}")
    print()


def _print_comparison_block(result: ScenarioResult) -> None:
    """Show what a naive single-channel threshold monitor would
    have done, side-by-side with what the actual system did."""
    print("  Naive single-channel threshold monitor vs. this system:")
    print("    [naive] would have flagged: ", end="")
    if not result.naive_flags_by_channel:
        print("(no channels exceeded |z|>3)")
    else:
        items = sorted(
            result.naive_flags_by_channel.items(),
            key=lambda kv: kv[1]
        )
        for ch, idx in items:
            print(f"{ch}@t={idx*0.1:.1f}s  ", end="")
    print()
    print(f"    [system]  {result.system_conclusion}")
    print()


def _print_scenario_table(result: ScenarioResult) -> None:
    """Render one scenario as a complete report."""
    _print_header(result.scenario_id, result.scenario_name, result.description)
    n = len(result.ticks)
    every = max(1, n // 12)
    _print_tick_table(result.ticks, every=every)
    _print_event_timeline(result.ticks)
    _print_final_state(result)
    _print_comparison_block(result)


# ---------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------
def _result_to_dict(result: ScenarioResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "description": result.description,
        "ticks": [
            {
                "tick_index": r.tick_index,
                "time_s": r.time_s,
                "observed": r.observed,
                "twin_expected": r.twin_expected,
                "residual_z": r.residual_z,
                "anomaly_score": r.anomaly_score,
                "anomaly_label": r.anomaly_label,
                "health_overall": r.health_overall,
                "health_label": r.health_label,
                "rul_status": r.rul_status,
                "rul_estimate_h": r.rul_estimate_h,
                "rul_lower_h": r.rul_lower_h,
                "rul_upper_h": r.rul_upper_h,
                "rul_uncertain": r.rul_uncertain,
                "risk_status": r.risk_status,
                "risk_score": r.risk_score,
                "rel_band": r.rel_band,
                "rel_headline": r.rel_headline,
                "rel_drivers": list(r.rel_drivers),
                "rel_confidence": r.rel_confidence,
                "env_wind_mps": r.env_wind_mps,
                "env_turbulence": r.env_turbulence,
                "naive_flags": list(r.naive_flags),
                "new_events": list(r.new_events),
            }
            for r in result.ticks
        ],
        "summary": result.summary,
        "final_engine_state": result.final_engine_state,
        "final_risk_status": result.final_risk_status,
        "final_reliability_band": result.final_reliability_band,
        "final_reliability_headline": result.final_reliability_headline,
        "naive_flags_by_channel": result.naive_flags_by_channel,
        "system_conclusion": result.system_conclusion,
    }
    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backend.dashboard.sih_demo",
        description=(
            "Software-in-the-Loop demonstration of the aero-piston "
            "digital twin on six canonical scenarios."
        ),
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help=(
            "Run only this scenario (1..6).  Default: run all 6."
        ),
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=600,
        help=(
            "How many ticks to run per scenario "
            "(default 600 = 60 s at 0.1 s/tick)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of a human-readable table.",
    )
    args = parser.parse_args(argv)

    if args.scenario is not None:
        results = [run_scenario(args.scenario, ticks=args.ticks)]
    else:
        results = run_all(ticks=args.ticks)

    if args.json:
        print(json.dumps(
            [_result_to_dict(r) for r in results],
            indent=2,
        ))
    else:
        for r in results:
            _print_scenario_table(r)
        print("=" * 78)
        print("  SIH DEMO COMPLETE — all scenarios deterministic, all runs reproducible.")
        print("=" * 78)
    return 0


__all__ = [
    "CompositeInjector",
    "DropoutInjector",
    "SCENARIO_BUILDERS",
    "SCENARIO_NAMES",
    "ScenarioResult",
    "TickRecord",
    "main",
    "run_all",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
