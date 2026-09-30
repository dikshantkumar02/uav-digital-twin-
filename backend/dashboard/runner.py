"""
Pipeline runner (PHASE 13).

The :class:`PipelineRunner` wires every per-tick calculator from
PHASES 2-12 into a single ``tick()`` API. It is the dashboard's
"engine room": each call to :meth:`tick` advances the simulator
one step, runs the full chain, and returns a
:class:`~backend.dashboard.snapshot.DashboardSnapshot`.

The runner is pure with respect to wall time — it is deterministic
for a given scenario, so the test suite can call ``tick()`` in a
tight loop and assert against exact outputs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, List, Optional

from backend.config import LoadedConfig
from backend.diagnostics import AnomalyDetector
from backend.digital_twin import DigitalTwin
from backend.environment import MissionProfile
from backend.faults import FaultInjector
from backend.health import HealthIndexCalculator
from backend.ml import FaultClassification, FaultClassifier
from backend.risk import (
    MissionReliabilityInputs,
    MissionRiskCalculator,
    aggregate_reliability,
)
from backend.rul import RulCalculator
from backend.sensors import SensorBundle, SensorSample
from backend.simulation import EngineSimulator
from backend.telemetry import FrameStatus, LatencyTracker

from .alerts import derive_alerts
from .scenarios import ScenarioSpec
from .snapshot import DashboardSnapshot


# Default dt — matches the rest of the system.
DEFAULT_DT_S = 0.1

# Ring-buffer / history size — overridden by DashboardConfig.history_size.
DEFAULT_HISTORY_SIZE = 600

# PHASE 19 — ring-buffer cap on the *events* log surfaced through
# the snapshot. The events list is short on purpose: the
# dashboard's bottom strip shows the most recent few entries, so
# keeping a long history would be a memory cost without a UI
# payoff.
MAX_EVENT_LOG = 32


@dataclass(frozen=True)
class _PipelineState:
    """The per-tick outputs of one step of the pipeline."""

    sample: SensorSample
    health: object  # HealthIndex (avoids forward ref in type signature)
    rul: object     # RulEstimate
    risk: object    # RiskAssessment
    anomaly: object  # AnomalyAssessment
    classification: Optional[FaultClassification]


class PipelineRunner:
    """Drives the full per-tick chain from a single :class:`ScenarioSpec`.

    Usage::

        runner = PipelineRunner(cfg, SCENARIO_REGISTRY["engine_degradation_60s"])
        snap = runner.tick()
        for snap in runner.run(duration_s=60.0):
            ...

    The runner is single-threaded and deterministic. It owns a ring
    buffer of the last ``history_size`` snapshots so a late-loading
    page is never empty.
    """

    def __init__(
        self,
        cfg: LoadedConfig,
        scenario: ScenarioSpec,
        *,
        history_size: int = DEFAULT_HISTORY_SIZE,
        dt_s: float = DEFAULT_DT_S,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self._cfg = cfg
        self._scenario = scenario
        self._dt_s = float(dt_s)
        self._history: Deque[DashboardSnapshot] = deque(maxlen=int(history_size))
        self._tick_count = 0
        self._last: Optional[DashboardSnapshot] = None
        # PHASE 22 — unified diagnostic state. The same
        # ``DiagnosticAssembler`` instance is reused across ticks
        # so the residual-trend sliding window accumulates within
        # a scenario but is reset between scenarios.
        from backend.diagnostics.diagnostic_assembler import (
            DiagnosticAssembler,
        )
        from backend.diagnostics.diagnostic_state import DiagnosticState
        from backend.risk.reliability import MissionReliabilityAssessment
        self._diagnostic_assembler = DiagnosticAssembler()
        self._latest_diagnostic: Optional[DiagnosticState] = None
        self._diagnostic_history: Deque[DiagnosticState] = deque(
            maxlen=int(history_size)
        )
        # PHASE 23 — mission reliability view. The assessment is
        # a pure-function output of the diagnostic state; no
        # instance state lives on this object beyond the cache
        # and the ring buffer.
        self._latest_reliability: Optional[MissionReliabilityAssessment] = None
        self._reliability_history: Deque[MissionReliabilityAssessment] = deque(
            maxlen=int(history_size)
        )
        # Cached so the per-tick computation doesn't re-read YAML.
        self._engine_rated_power_kw: float = float(
            cfg.engine.geometry.rated_power_kw
        )
        # PHASE 17: optional shared LatencyTracker — when provided,
        # the runner records per-stage timings into it. None
        # preserves the existing PHASE 13 zero-overhead behaviour.
        self._tracker = latency_tracker

        # --- Simulators + sensors ---
        self._sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=dt_s)
        self._bundle = SensorBundle(cfg.sensors, self._sim, master_seed=scenario.seed)
        # Deterministic: a fresh seed per scenario.
        self._sim.reset()

        # --- Fault injector (PHASE 7) ---
        self._injector = FaultInjector(scenario.to_fault_scenario())
        self._injector.attach(self._sim, self._bundle)

        # --- Digital twin (PHASE 6) ---
        self._twin = DigitalTwin(cfg.engine, dt_s=dt_s)

        # --- Anomaly detector (PHASE 8) ---
        self._anomaly = AnomalyDetector()

        # --- Optional fault classifier (PHASE 9) ---
        self._classifier: Optional[FaultClassifier] = None
        if scenario.run_classifier:
            self._classifier = FaultClassifier()

        # --- Mission profile (PHASE 2) — needed by the risk calculator ---
        profile = MissionProfile.from_config(cfg.environment.mission)

        # --- Health / RUL / Risk (PHASES 10 / 11 / 12) ---
        self._health = HealthIndexCalculator(cfg.engine)
        self._rul = RulCalculator(cfg.engine)
        self._risk = MissionRiskCalculator(profile, config=cfg.risk)

        # --- PHASE 19 — control-room dashboard state -------------------
        # Event log: short rolling list of scenario events
        # (fault onsets, mission-phase transitions, sensor faults).
        # Kept small and in-memory; the dashboard's bottom strip
        # shows the most recent few entries.
        self._events: List[dict] = []
        # The previous tick's fault-active set is used to detect
        # *transitions* (e.g. a new fault class onsets, or a
        # previously active fault class resolves). Transitions
        # are the only events that get logged.
        self._prev_active_fault: Optional[str] = None
        self._prev_sensor_fault: bool = False
        self._prev_phase: Optional[str] = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def scenario(self) -> ScenarioSpec:
        return self._scenario

    @property
    def history(self) -> Deque[DashboardSnapshot]:
        return self._history

    @property
    def last_snapshot(self) -> Optional[DashboardSnapshot]:
        return self._last

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def dt_s(self) -> float:
        return self._dt_s

    @property
    def is_finished(self) -> bool:
        """True when the mission profile has been fully consumed."""
        return self._sim.runner._t >= self._sim.runner.duration_s - 1e-9

    @property
    def latency_tracker(self) -> Optional[LatencyTracker]:
        return self._tracker

    @property
    def diagnostic_assembler(self) -> "DiagnosticAssembler":
        """The :class:`DiagnosticAssembler` used by this runner.

        Exposed so tests and advanced consumers can introspect
        the trend-history state.
        """
        return self._diagnostic_assembler

    @property
    def latest_diagnostic(self) -> Optional["DiagnosticState"]:
        """The most recent :class:`DiagnosticState` (or ``None`` if
        no tick has run yet)."""
        return self._latest_diagnostic

    @property
    def diagnostic_history(self) -> Deque["DiagnosticState"]:
        """Rolling window of recent diagnostic states (same length
        as the snapshot history)."""
        return self._diagnostic_history

    # ------------------------------------------------------------------
    # PHASE 23 — mission reliability view
    # ------------------------------------------------------------------
    @property
    def latest_reliability(self) -> Optional["MissionReliabilityAssessment"]:
        """The most recent :class:`MissionReliabilityAssessment`
        (or ``None`` if no tick has run yet)."""
        return self._latest_reliability

    @property
    def reliability_history(self) -> Deque["MissionReliabilityAssessment"]:
        """Rolling window of recent reliability assessments (same
        length as the snapshot history)."""
        return self._reliability_history

    # ------------------------------------------------------------------
    # Tick — the canonical API
    # ------------------------------------------------------------------
    def tick(self) -> DashboardSnapshot:
        """Advance the simulator + pipeline by one ``dt`` and return
        the per-tick :class:`DashboardSnapshot`.

        Deterministic with respect to the current scenario.
        """
        # Local imports to avoid forcing import-time cycles in tests.
        import time as _time

        tr = self._tracker
        t_now = self._sim.runner._t
        # 1) Fault injector advisory (per-tick + sensor side-channel + env).
        fault = self._injector.tick(t_now)
        self._injector.apply_sensor(t_now)
        self._injector.apply_environment(t_now)

        # 2) Sensor bundle advances the simulator and reads every channel.
        t0_ingest = _time.perf_counter()
        sample = self._bundle.tick(
            degradation_severity=fault.degradation_severity,
            vibration_external=fault.vibration_external,
        )
        if tr is not None:
            tr.record("ingestion", _time.perf_counter() - t0_ingest)

        # 3) Digital twin: closed-loop step + residual frame.
        obs = {
            ch: r.value
            for ch, r in sample.readings.items()
            if r.value is not None
        }
        t0_twin = _time.perf_counter()
        twin_state = self._twin.step(sample.env, obs)
        residual = self._twin.residual(twin_state, obs)
        if tr is not None:
            tr.record("digital_twin", _time.perf_counter() - t0_twin)

        # 4) Anomaly detection (PHASE 8). Default frame_status is OK.
        t0_ai = _time.perf_counter()
        anomaly = self._anomaly.detect(
            residual, sample, frame_status=FrameStatus.OK
        )

        # 5) Optional PHASE 9 fault classification (windowed).
        classification: Optional[FaultClassification] = None
        if self._classifier is not None:
            self._classifier.window.append(
                time_s=sample.time_s,
                residual=residual,
                sample=sample,
                env=sample.env,
            )
            classification = self._classifier.classify()

        # 6) Health index (PHASE 10) — drives the SENSORS subsystem.
        health = self._health.update(
            state=sample.engine,
            anomaly=anomaly,
            classification=classification,
            time_s=sample.time_s,
        )

        # 7) RUL (PHASE 11) — driven by health + engine truth.
        rul = self._rul.update(
            health=health,
            state=sample.engine,
            time_s=sample.time_s,
            dt_s=self._dt_s,
        )
        if tr is not None:
            tr.record("ai", _time.perf_counter() - t0_ai)

        # 8) Mission risk (PHASE 12) — drives the dashboard's go/no-go.
        t0_risk = _time.perf_counter()
        risk = self._risk.update(
            health=health,
            rul=rul,
            anomaly=anomaly,
            time_s=sample.time_s,
        )
        if tr is not None:
            tr.record("risk", _time.perf_counter() - t0_risk)

        # 9) Per-frame latency block — what the dashboard renders.
        latency_block: Optional[dict] = None
        if tr is not None:
            latency_block = {
                name: float(stage.last_s)
                for name, stage in tr.summary().items()
            }
            latency_block["end_to_end_s"] = float(sum(
                s.last_s for s in tr.summary().values()
            ))

        # 10) PHASE 19 — model_confidence (aggregated).
        model_conf = self._aggregate_model_confidence(risk, health, anomaly, classification)

        # 10b) PHASE 22 — assemble the unified diagnostic state early
        # so its ``data_quality`` can feed the alert path. The
        # assembler maintains a residual-trend sliding window
        # internally; the assembly is deterministic, so calling it
        # here (and again at the end with reliability attached) is
        # safe.
        self._latest_diagnostic = self._diagnostic_assembler.assemble(
            time_s=sample.time_s,
            sample=sample,
            twin_state=twin_state,
            residual=residual,
            anomaly=anomaly,
            classification=classification,
            health=health,
            rul=rul,
            risk=risk,
            engine=sample.engine,
            env=sample.env,
        )
        actual_data_quality = float(self._latest_diagnostic.data_quality)

        # 11) PHASE 19 — events log: detect transitions and append.
        self._update_events(
            sample=sample,
            fault=fault,
            risk=risk,
            residual=residual,
        )

        # 12) PHASE 19 — residual + twin_expected blocks (flat dicts).
        residual_dict = residual.to_dict()
        twin_expected = {
            ch: float(r.predicted)
            for ch, r in residual.residuals.items()
        }

        # 13) PHASE 19 — derive alerts (pure function, post-tick).
        snap_payload = {
            "scenario_name": self._scenario.name,
            "time_s": sample.time_s,
            "risk": risk.to_dict(),
            "anomaly": anomaly.to_dict(),
            "fault_classification": (
                classification.to_dict() if classification is not None else None
            ),
            "environment": sample.env.to_dict(),
            "residual": residual_dict,
            "metadata": {
                "data_quality": actual_data_quality,
                "model_version": "phase17-streaming-1.0.0",
            },
            "model_confidence": model_conf,
            # The health index drives the engine-wear banner;
            # the alert helper needs the full wire dict to
            # see subsystems.
            "health": health.to_dict(),
            # RUL is the second engine-wear signal.
            "rul": rul.to_dict(),
        }
        # Inject a per-channel sensor_health view derived from
        # the bundle's readings so the SENSOR_FAULT banner has
        # something to summarise. The view is a small
        # ``{channel: {"mode": "..."}}`` dict — same shape the
        # alerts helper expects.
        snap_payload["sample"] = {
            "sensor_health": self._sample_sensor_health(sample),
        }
        alerts = derive_alerts(snap_payload)

        snap = DashboardSnapshot(
            time_s=sample.time_s,
            tick_index=self._tick_count,
            scenario_name=self._scenario.name,
            frame_status=FrameStatus.OK.value,
            engine_state=sample.engine,
            environment=sample.env,
            health=health,
            rul=rul,
            risk=risk,
            anomaly=anomaly,
            fault_classification=classification,
            latency=latency_block,
            mission_id="mission-001",
            vehicle_id="vehicle-001",
            engine_id="engine-001",
            data_quality=actual_data_quality,
            model_version="phase17-streaming-1.0.0",
            # PHASE 19
            residual=residual_dict,
            twin_expected=twin_expected,
            alerts=tuple(alerts),
            events=tuple(self._events),
            model_confidence=model_conf,
        )
        self._history.append(snap)
        self._last = snap
        # PHASE 22 — diagnostic state was assembled in step 10b
        # (early) so its ``data_quality`` could feed the alert
        # path. The assembler is deterministic given the same
        # inputs, so re-running it here would be wasteful; we
        # instead attach the reliability view to the existing
        # state. We use ``dataclasses.replace`` because
        # ``DiagnosticState`` is frozen.
        self._latest_reliability = self._assess_reliability(
            sample=sample,
            risk=risk,
            diagnostic=self._latest_diagnostic,
            time_s=sample.time_s,
        )
        from dataclasses import replace as _dc_replace
        self._latest_diagnostic = _dc_replace(
            self._latest_diagnostic,
            reliability=self._latest_reliability,
        )
        self._diagnostic_history.append(self._latest_diagnostic)
        self._reliability_history.append(self._latest_reliability)
        self._tick_count += 1
        return snap

    # ------------------------------------------------------------------
    # Run — convenience generator
    # ------------------------------------------------------------------
    def run(self, duration_s: float) -> Iterator[DashboardSnapshot]:
        """Yield one :class:`DashboardSnapshot` per tick for ``duration_s``
        of sim time. Stops early if the mission profile is exhausted.
        """
        if duration_s <= 0:
            return
        n_steps = int(math.ceil(duration_s / self._dt_s))
        for _ in range(n_steps):
            if self.is_finished:
                break
            yield self.tick()

    # ------------------------------------------------------------------
    # Reset / scenario switch
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all calculators, the simulator, the sensor bundle, and
        clear the history. The active scenario is preserved.
        """
        self._sim.reset()
        self._bundle.reset()
        self._twin.reset()
        self._anomaly.reset()
        if self._classifier is not None:
            self._classifier.reset()
        self._health.reset()
        self._rul.reset()
        self._risk.reset()
        self._injector.reset()
        self._history.clear()
        self._last = None
        self._tick_count = 0
        # PHASE 19 — clear event log + transition tracking.
        self._events.clear()
        self._prev_active_fault = None
        self._prev_sensor_fault = False
        self._prev_phase = None
        # PHASE 22 — clear the diagnostic state and reset the
        # assembler's residual-trend sliding window.
        self._diagnostic_assembler.reset()
        self._latest_diagnostic = None
        self._diagnostic_history.clear()
        # PHASE 23 — clear the reliability cache and history.
        self._latest_reliability = None
        self._reliability_history.clear()

    def set_scenario(self, scenario: ScenarioSpec) -> None:
        """Switch to a new scenario. Resets the entire pipeline.

        Re-instantiates the simulator, sensor bundle, fault injector,
        and calculators so the new scenario starts from a clean
        state. This is the operation invoked by ``POST /api/scenario``.
        """
        # Tear down the old scenario.
        # (We rebuild the entire pipeline so scenario-specific state
        # is fully replaced — including a fresh simulator that the
        # fault injector can re-attach to.)
        self._scenario = scenario
        # Reset the cache so a fresh load_config() is needed if the
        # user calls it — but the LoadedConfig is immutable so we
        # just rebuild the pipeline.
        self._sim = EngineSimulator(
            self._cfg.engine, self._cfg.environment, dt_s=self._dt_s
        )
        self._bundle = SensorBundle(
            self._cfg.sensors, self._sim, master_seed=scenario.seed
        )
        self._injector = FaultInjector(scenario.to_fault_scenario())
        self._injector.attach(self._sim, self._bundle)
        self._twin = DigitalTwin(self._cfg.engine, dt_s=self._dt_s)
        self._anomaly = AnomalyDetector()
        self._classifier = (
            FaultClassifier() if scenario.run_classifier else None
        )
        profile = MissionProfile.from_config(self._cfg.environment.mission)
        self._health = HealthIndexCalculator(self._cfg.engine)
        self._rul = RulCalculator(self._cfg.engine)
        self._risk = MissionRiskCalculator(profile, config=self._cfg.risk)
        self._history.clear()
        self._last = None
        self._tick_count = 0
        # PHASE 19 — clear event log + transition tracking so a
        # scenario switch starts from a clean event feed.
        self._events.clear()
        self._prev_active_fault = None
        self._prev_sensor_fault = False
        self._prev_phase = None

    def list_scenarios(self) -> List[dict]:
        """Wire-format list of all scenarios (for ``/api/scenarios``)."""
        from .scenarios import SCENARIO_REGISTRY

        return [s.to_dict() for s in SCENARIO_REGISTRY.values()]

    # ------------------------------------------------------------------
    # PHASE 19 — model_confidence aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_model_confidence(
        risk: object,
        health: object,
        anomaly: object,
        classification: Optional[object],
    ) -> float:
        """Average the available confidence signals into a single 0..1.

        Each source contributes its own confidence. The aggregate is
        the mean of the present values; if no calculator is
        calibrated, the result is ``0.0`` (not "high") so the
        dashboard surfaces that honestly.
        """
        values: List[float] = []
        try:
            rc = float(getattr(risk, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            rc = 0.0
        try:
            hc = float(getattr(health, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            hc = 0.0
        try:
            ac = float(getattr(anomaly, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            ac = 0.0
        # We require at least one calibrated source before we
        # report a non-zero aggregate. ``0.0`` = nothing calibrated.
        if rc > 0.0:
            values.append(rc)
        if hc > 0.0:
            values.append(hc)
        if ac > 0.0:
            values.append(ac)
        if classification is not None:
            try:
                cc = float(getattr(classification, "confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                cc = 0.0
            if cc > 0.0:
                values.append(cc)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    # ------------------------------------------------------------------
    # PHASE 19 — events log
    # ------------------------------------------------------------------
    def _record_event(self, kind: str, description: str, time_s: float) -> None:
        """Append a single event to the rolling log."""
        self._events.append({
            "time_s": float(time_s),
            "kind": str(kind),
            "description": str(description),
        })
        # Trim from the front when we exceed the cap.
        if len(self._events) > MAX_EVENT_LOG:
            del self._events[0 : len(self._events) - MAX_EVENT_LOG]

    def _update_events(
        self,
        *,
        sample: SensorSample,
        fault: object,
        risk: object,
        residual: object,
    ) -> None:
        """Detect transitions and append a structured event per transition.

        Three transitions are tracked:
          1. A new fault class onsets (``FAULT_INJECTED``).
          2. A previously active fault class clears (``FAULT_CLEARED``).
          3. A sensor fault becomes active or clears (``SENSOR_FAULT``).
        The risk module's status transition is also surfaced.
        """
        t = float(sample.time_s)

        # 1) Fault-class onsets / clearances
        active_fault = None
        try:
            active_fault = str(getattr(fault, "fault_class", "") or "")
        except Exception:
            active_fault = None
        if active_fault and active_fault != "NONE" and active_fault != self._prev_active_fault:
            self._record_event(
                "FAULT_INJECTED",
                f"fault_class={active_fault}",
                t,
            )
        elif self._prev_active_fault and (
            not active_fault or active_fault == "NONE"
        ):
            self._record_event(
                "FAULT_CLEARED",
                f"fault_class={self._prev_active_fault}",
                t,
            )
        self._prev_active_fault = active_fault

        # 2) Sensor-fault transition (any channel mode != NORMAL).
        sensor_fault_active = any(
            str(getattr(r, "mode", "")).upper() != "NORMAL"
            for r in sample.readings.values()
        )
        if sensor_fault_active and not self._prev_sensor_fault:
            bad = [
                ch
                for ch, r in sample.readings.items()
                if str(getattr(r, "mode", "")).upper() != "NORMAL"
            ]
            self._record_event(
                "SENSOR_FAULT",
                "channels: " + ", ".join(bad) if bad else "channel: —",
                t,
            )
        self._prev_sensor_fault = sensor_fault_active

        # 3) Risk status transition
        risk_status = str(getattr(risk, "status", "") or "")
        risk_severity = str(getattr(risk, "severity", "") or "")
        if risk_status and risk_status not in ("GO", "NOMINAL", "NORMAL"):
            # Only log when status changed (compared to last entry).
            last = self._events[-1] if self._events else None
            if last is None or last.get("kind") != "RISK_TRANSITION":
                self._record_event(
                    "RISK_TRANSITION",
                    f"status={risk_status} severity={risk_severity}",
                    t,
                )

    # ------------------------------------------------------------------
    # PHASE 19 — per-channel sensor-health view
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_sensor_health(sample: SensorSample) -> dict:
        """Return ``{channel: {"mode": "..."}}`` from a ``SensorSample``.

        Used by :func:`backend.dashboard.alerts.derive_alerts` to
        detect STUCK / DROPPED / DRIFTING / SPIKE / FAULT
        conditions. Channels whose reading is missing or normal
        are omitted from the dict to keep the wire payload small.
        """
        out: dict = {}
        for ch, reading in sample.readings.items():
            if reading is None:
                continue
            mode_obj = getattr(reading, "mode", None)
            # The mode is a NoiseMode(str, Enum); pull the
            # value (e.g. "STUCK") rather than the repr
            # ("NoiseMode.STUCK") so the alert banner reads
            # cleanly.
            if hasattr(mode_obj, "value"):
                mode = str(mode_obj.value).upper()
            else:
                mode = str(mode_obj or "NORMAL").upper()
            if mode == "NORMAL":
                continue
            out[str(ch)] = {"mode": mode}
        return out

    # ------------------------------------------------------------------
    # PHASE 23 — mission reliability assessment
    # ------------------------------------------------------------------
    def _assess_reliability(
        self,
        *,
        sample: SensorSample,
        risk: object,
        diagnostic: "DiagnosticState",
        time_s: float,
    ) -> "MissionReliabilityAssessment":
        """Assemble a :class:`MissionReliabilityInputs` bundle from
        the per-tick outputs and run the pure aggregator.

        All numeric signals are derived from upstream values the
        pipeline has already computed (PHASE 22 diagnostic state,
        PHASE 12 risk assessment, the sensor sample, the engine
        state). We do not query any extra I/O here.
        """
        # Engine load proxy: brake power / rated power, clipped.
        engine_state = sample.engine
        rated_kw = max(1.0, float(self._engine_rated_power_kw))
        engine_load = max(
            0.0,
            min(1.0, float(engine_state.brake_power_kw) / rated_kw),
        )

        # Environmental severity: composite of turbulence + gust,
        # normalised against a conservative 5 m/s envelope.
        env_d = diagnostic.environment_context or {}
        turbulence = float(env_d.get("turbulence_intensity", 0.0) or 0.0)
        wind_mps = float(env_d.get("wind_mps", 0.0) or 0.0)
        env_severity = max(0.0, min(1.0, max(turbulence, wind_mps) / 5.0))

        # Sensor confidence: PHASE 22 data_quality is already the
        # fraction of channels that produced a non-dropped reading.
        sensor_confidence = float(diagnostic.data_quality)

        # Per-category fault probabilities: PHASE 22 groups them
        # into engine / sensor / environment buckets. When the
        # classifier is uncalibrated, ``unknown=1.0`` and the other
        # buckets are 0.0, which is exactly what we want — the
        # ``unknown`` bucket is intentionally not consumed here so
        # the band honestly reflects the available evidence.
        fp = diagnostic.fault_probabilities or {}
        p_engine = float(fp.get("engine", 0.0) or 0.0)
        p_sensor = float(fp.get("sensor", 0.0) or 0.0)
        p_env = float(fp.get("environment", 0.0) or 0.0)

        # RUL uncertainty: 1.0 when RUL is uncertain, else the
        # fractional spread of the lower/upper band.
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

        # Remaining mission duration (hours): reuse the PHASE 12
        # ``hours_to_destination`` if present, else None.
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


__all__ = [
    "DEFAULT_DT_S",
    "DEFAULT_HISTORY_SIZE",
    "PipelineRunner",
]
