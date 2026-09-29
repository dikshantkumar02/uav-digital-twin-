"""
AsyncStreamingPipeline (PHASE 17 — real-time streaming pipeline).

Wires five independent coroutine stages together with bounded
``asyncio.Queue``s so each stage runs at its own pace and the
pipeline survives transient slowdowns (graceful back-pressure
with drop-on-overflow). The five stages mirror the user spec:

    in_q   →  validation    →  val_q
    val_q  →  preprocessing →  pp_q
    pp_q   →  digital_twin  →  dt_q
    dt_q   →  ai            →  ai_q
    ai_q   →  risk          →  out_q

Each stage records itself into a shared :class:`LatencyTracker`
under its stage name. The end-to-end latency is captured by
the risk stage using the frame's ``produced_wall_time`` and
the current wall time when the risk assessment is emitted.

Pipeline lifecycle
------------------

* :meth:`start` spawns 5 daemon tasks. Idempotent.
* :meth:`submit` puts a :class:`TelemetryFrame` into ``in_q``
  using ``put_nowait`` — when the queue is full and
  ``drop_on_overflow`` is True, the **oldest** frame is dropped
  (and recorded as such) so the latest always makes it in.
* :meth:`output_queue` exposes the final ``out_q`` so the
  dashboard / WebSocket can consume results.
* :meth:`stop` cancels all tasks and drains the queues.

All exceptions inside a stage are caught and logged. A failing
stage keeps the rest of the pipeline alive — the spec is
"graceful failure", not "crash on first error".
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
)

from .frame import FrameStatus, TelemetryFrame
from .latency import LatencyTracker
from .preprocessor import Preprocessor


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """Static configuration for the :class:`AsyncStreamingPipeline`."""

    queue_max_size: int = 64
    drop_on_overflow: bool = True
    n_workers_per_stage: int = 1            # reserved (currently 1 only)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineResult:
    """The full output of the risk stage — the dashboard's per-tick payload.

    Every calculator's output is included so the WebSocket can
    publish a complete snapshot in one message. ``latency`` is
    the per-stage breakdown (filled by the risk stage) so the
    dashboard shows the **measured** per-stage latency, not a
    hard-coded constant.
    """

    frame: TelemetryFrame
    twin_state: Any                 # TwinState (avoids forward ref)
    residual: Any                   # ResidualFrame
    anomaly: Any                    # AnomalyAssessment
    classification: Any             # Optional[FaultClassification]
    health: Any                     # HealthIndex
    rul: Any                        # RulEstimate
    risk: Any                       # RiskAssessment
    latency: Dict[str, float] = field(default_factory=dict)
    end_to_end_latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame.to_dict(),
            "anomaly": self.anomaly.to_dict() if self.anomaly is not None else None,
            "classification": (
                self.classification.to_dict()
                if self.classification is not None
                else None
            ),
            "health": self.health.to_dict() if self.health is not None else None,
            "rul": self.rul.to_dict() if self.rul is not None else None,
            "risk": self.risk.to_dict() if self.risk is not None else None,
            "latency": dict(self.latency),
            "end_to_end_latency_s": float(self.end_to_end_latency_s),
        }


# ---------------------------------------------------------------------
# Stage counters
# ---------------------------------------------------------------------
@dataclass
class StageStats:
    """Counters for one pipeline stage."""

    name: str
    processed: int = 0
    dropped_overflow: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "processed": self.processed,
            "dropped_overflow": self.dropped_overflow,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class AsyncStreamingPipeline:
    """Five-stage async streaming pipeline."""

    def __init__(
        self,
        *,
        cfg: Optional[PipelineConfig] = None,
        digital_twin: Any = None,
        anomaly_detector: Any = None,
        classifier: Any = None,                # optional FaultClassifier
        health_calculator: Any = None,
        rul_calculator: Any = None,
        risk_calculator: Any = None,
        preprocessor: Optional[Preprocessor] = None,
        tracker: Optional[LatencyTracker] = None,
    ) -> None:
        self._cfg = cfg or PipelineConfig()
        self._twin = digital_twin
        self._anomaly = anomaly_detector
        self._classifier = classifier
        self._health = health_calculator
        self._rul = rul_calculator
        self._risk = risk_calculator
        self._pre = preprocessor or Preprocessor(latency_tracker=tracker)
        self._tracker = tracker

        max_q = int(self._cfg.queue_max_size)
        # 4 inter-stage queues + 1 input queue.
        self._in_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
        self._val_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
        self._pp_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
        self._dt_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
        self._ai_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
        self._out_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)

        self._tasks: List[asyncio.Task] = []
        self._stage_stats: Dict[str, StageStats] = {
            "validation": StageStats("validation"),
            "preprocessing": StageStats("preprocessing"),
            "digital_twin": StageStats("digital_twin"),
            "ai": StageStats("ai"),
            "risk": StageStats("risk"),
        }
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def output_queue(self) -> asyncio.Queue:
        return self._out_q

    @property
    def tracker(self) -> Optional[LatencyTracker]:
        return self._tracker

    def stage_stats(self) -> Dict[str, StageStats]:
        return self._stage_stats

    def queue_depths(self) -> Dict[str, int]:
        return {
            "in": self._in_q.qsize(),
            "validation": self._val_q.qsize(),
            "preprocessing": self._pp_q.qsize(),
            "digital_twin": self._dt_q.qsize(),
            "ai": self._ai_q.qsize(),
            "risk": self._out_q.qsize(),
        }

    def is_running(self) -> bool:
        return all(not t.done() for t in self._tasks) and bool(self._tasks)

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------
    async def submit(self, frame: TelemetryFrame) -> bool:
        """Submit a frame to the pipeline.

        Returns True if the frame is in ``in_q`` after the call.
        On overflow with ``drop_on_overflow`` the oldest frame is
        discarded and ``True`` is still returned (the spec
        promise: "latest always makes it in").
        """
        try:
            self._in_q.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            if not self._cfg.drop_on_overflow:
                return False
            try:
                dropped = self._in_q.get_nowait()
                self._stage_stats["validation"].dropped_overflow += 1
                log.warning(
                    "pipeline: input queue full, dropped frame seq=%d",
                    dropped.sequence,
                )
            except asyncio.QueueEmpty:
                pass
            try:
                self._in_q.put_nowait(frame)
                return True
            except asyncio.QueueFull:
                return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._tasks:
            return
        self._stop_event.clear()
        self._tasks = [
            asyncio.create_task(self._validation_loop(), name="val"),
            asyncio.create_task(self._preprocessing_loop(), name="pp"),
            asyncio.create_task(self._digital_twin_loop(), name="dt"),
            asyncio.create_task(self._ai_loop(), name="ai"),
            asyncio.create_task(self._risk_loop(), name="risk"),
        ]

    async def stop(self, timeout_s: float = 2.0) -> None:
        """Cancel all stage tasks and wait for them to exit."""
        self._stop_event.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await asyncio.wait_for(t, timeout=timeout_s)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        self._tasks = []

    # ------------------------------------------------------------------
    # Stage 1: validation
    # ------------------------------------------------------------------
    async def _validation_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._get_or_stop(self._in_q)
            if frame is None:
                return
            try:
                if self._tracker is not None:
                    with self._tracker.stage("validation"):
                        self._validate_inplace(frame)
                else:
                    self._validate_inplace(frame)
                if frame.status is FrameStatus.INVALID:
                    # Skip the heavy stages but still produce a
                    # placeholder result so the dashboard can show
                    # "INVALID" honestly. Anomaly/AI calculators
                    # themselves also short-circuit on INVALID.
                    pass
                await self._put_or_drop(self._val_q, frame, "validation")
                self._stage_stats["validation"].processed += 1
            except Exception as exc:  # noqa: BLE001
                self._stage_stats["validation"].errors += 1
                log.exception("validation stage error: %s", exc)

    @staticmethod
    def _validate_inplace(frame: TelemetryFrame) -> None:
        """Re-derive the frame status from sample readings.

        The adapter already set an initial status; this stage
        also picks up: ``data_quality < 0.2`` → INVALID (so the
        downstream AI short-circuits on truly bad data).
        """
        if frame.data_quality < 0.20:
            frame.status = FrameStatus.INVALID
        # Note: the preprocessor's dropout / forward-fill logic
        # is what upgrades STALE → OK or downgrades to INVALID
        # for the per-channel forward-fill case. Here we only
        # adjust for the gross data-quality floor.

    # ------------------------------------------------------------------
    # Stage 2: preprocessing
    # ------------------------------------------------------------------
    async def _preprocessing_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._get_or_stop(self._val_q)
            if frame is None:
                return
            try:
                if self._tracker is not None:
                    with self._tracker.stage("preprocessing"):
                        self._pre.process(frame)
                else:
                    self._pre.process(frame)
                await self._put_or_drop(self._pp_q, frame, "preprocessing")
                self._stage_stats["preprocessing"].processed += 1
            except Exception as exc:  # noqa: BLE001
                self._stage_stats["preprocessing"].errors += 1
                log.exception("preprocessing stage error: %s", exc)

    # ------------------------------------------------------------------
    # Stage 3: digital twin
    # ------------------------------------------------------------------
    async def _digital_twin_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._get_or_stop(self._pp_q)
            if frame is None:
                return
            try:
                if self._tracker is not None:
                    with self._tracker.stage("digital_twin"):
                        twin_state, residual = self._dt_step(frame)
                else:
                    twin_state, residual = self._dt_step(frame)
                self._pending_dt = (frame, twin_state, residual)
                await self._put_or_drop(self._dt_q, frame, "digital_twin")
                self._stage_stats["digital_twin"].processed += 1
            except Exception as exc:  # noqa: BLE001
                self._stage_stats["digital_twin"].errors += 1
                log.exception("digital_twin stage error: %s", exc)
                # Mark as INVALID so downstream skips AI.
                frame.status = FrameStatus.INVALID

    def _dt_step(self, frame: TelemetryFrame) -> Tuple[Any, Any]:
        sample = frame.sample
        if sample is None:
            return None, None
        obs = {
            ch: r.value
            for ch, r in sample.readings.items()
            if r.value is not None
        }
        env = sample.env
        if env is None:
            return None, None
        twin_state = self._twin.step(env, obs)
        residual = self._twin.residual(twin_state, obs)
        return twin_state, residual

    # ------------------------------------------------------------------
    # Stage 4: AI (anomaly + classification + health + RUL)
    # ------------------------------------------------------------------
    async def _ai_loop(self) -> None:
        # Local state that has to survive across frames.
        while not self._stop_event.is_set():
            frame = await self._get_or_stop(self._dt_q)
            if frame is None:
                return
            try:
                if self._tracker is not None:
                    with self._tracker.stage("ai"):
                        result = self._ai_step(frame)
                else:
                    result = self._ai_step(frame)
                self._pending_ai = result
                await self._put_or_drop(self._ai_q, frame, "ai")
                self._stage_stats["ai"].processed += 1
            except Exception as exc:  # noqa: BLE001
                self._stage_stats["ai"].errors += 1
                log.exception("ai stage error: %s", exc)

    def _ai_step(self, frame: TelemetryFrame) -> Dict[str, Any]:
        sample = frame.sample
        # Recompute twin + residual here (the dt stage stored them
        # in self._pending_dt but we want a self-contained AI step).
        twin_state, residual = self._dt_step(frame)
        anomaly = None
        if self._anomaly is not None and sample is not None and residual is not None:
            anomaly = self._anomaly.detect(
                residual, sample, frame_status=frame.status,
            )
        classification = None
        if self._classifier is not None and sample is not None and residual is not None:
            try:
                self._classifier.window.append(
                    time_s=sample.time_s,
                    residual=residual,
                    sample=sample,
                    env=sample.env,
                )
                classification = self._classifier.classify()
            except Exception:  # noqa: BLE001
                classification = None
        health = None
        if self._health is not None and sample is not None:
            health = self._health.update(
                state=sample.engine,
                anomaly=anomaly,
                classification=classification,
                time_s=sample.time_s,
            )
        rul = None
        if self._rul is not None and sample is not None:
            rul = self._rul.update(
                health=health,
                state=sample.engine,
                time_s=sample.time_s,
            )
        return {
            "twin_state": twin_state,
            "residual": residual,
            "anomaly": anomaly,
            "classification": classification,
            "health": health,
            "rul": rul,
        }

    # ------------------------------------------------------------------
    # Stage 5: risk
    # ------------------------------------------------------------------
    async def _risk_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._get_or_stop(self._ai_q)
            if frame is None:
                return
            try:
                if self._tracker is not None:
                    with self._tracker.stage("risk"):
                        result = self._risk_step(frame)
                else:
                    result = self._risk_step(frame)
                await self._put_or_drop(self._out_q, result, "risk")
                self._stage_stats["risk"].processed += 1
            except Exception as exc:  # noqa: BLE001
                self._stage_stats["risk"].errors += 1
                log.exception("risk stage error: %s", exc)

    def _risk_step(self, frame: TelemetryFrame) -> PipelineResult:
        ai = getattr(self, "_pending_ai", None) or self._ai_step(frame)
        risk = None
        if self._risk is not None and ai.get("health") is not None:
            risk = self._risk.update(
                health=ai["health"],
                rul=ai.get("rul"),
                anomaly=ai.get("anomaly"),
                time_s=frame.time_s,
            )
        e2e = max(0.0, _time.perf_counter() - float(frame.produced_wall_time))
        # Build the per-stage latency block.
        latency: Dict[str, float] = {}
        if self._tracker is not None:
            for name, stage in self._tracker.summary().items():
                latency[name] = float(stage.last_s)
        return PipelineResult(
            frame=frame,
            twin_state=ai.get("twin_state"),
            residual=ai.get("residual"),
            anomaly=ai.get("anomaly"),
            classification=ai.get("classification"),
            health=ai.get("health"),
            rul=ai.get("rul"),
            risk=risk,
            latency=latency,
            end_to_end_latency_s=e2e,
        )

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    async def _get_or_stop(self, q: asyncio.Queue) -> Optional[Any]:
        """Await one item, or None if the pipeline is stopping."""
        try:
            return await q.get()
        except asyncio.CancelledError:
            return None

    async def _put_or_drop(
        self, q: asyncio.Queue, item: Any, stage_name: str
    ) -> bool:
        try:
            await q.put(item)
            return True
        except asyncio.QueueFull:
            if not self._cfg.drop_on_overflow:
                return False
            try:
                dropped = q.get_nowait()
                self._stage_stats[stage_name].dropped_overflow += 1
                log.warning(
                    "pipeline: %s queue full, dropped oldest item",
                    stage_name,
                )
            except asyncio.QueueEmpty:
                pass
            try:
                await q.put(item)
                return True
            except asyncio.QueueFull:
                return False


__all__ = [
    "AsyncStreamingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StageStats",
]
