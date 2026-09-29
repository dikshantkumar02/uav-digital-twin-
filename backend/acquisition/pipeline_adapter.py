"""
Pipeline integration (PHASE 18 — embedded telemetry acquisition).

The :class:`AcquisitionPipelineAdapter` plugs the acquisition
layer into the PHASE 17 :class:`~backend.telemetry.AsyncStreamingPipeline`.

It reads :class:`TelemetrySample` objects from a
:class:`SensorAbstraction`, converts them to the legacy
:class:`~backend.sensors.SensorSample` shape the pipeline
already understands, runs them through a
:class:`~backend.telemetry.TelemetryAdapter` to set the
spec-mandated ``mission_id`` / ``vehicle_id`` / ``engine_id``
/ ``data_quality`` / ``model_version`` fields, and submits the
resulting :class:`~backend.telemetry.TelemetryFrame` to the
pipeline.

The end-to-end chain is::

    SimulatedAcquisition
    → SensorAbstraction (calibration + sampling + health)
    → AcquisitionPipelineAdapter
    → TelemetryAdapter (data-quality scoring)
    → AsyncStreamingPipeline
    → PipelineResult
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from backend.sensors import (
    NoiseMode,
    SensorReading,
    SensorSample,
    SENSOR_CHANNELS,
)
from backend.telemetry import (
    AdapterConfig,
    AsyncStreamingPipeline,
    TelemetryAdapter,
)

from .schema import SensorHealth, TELEMETRY_FIELDS, TelemetrySample
from .sensor import SensorAbstraction


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------
# Channels the spec defines that overlap with PHASE 4's existing
# SENSOR_CHANNELS. The IMU is collapsed to ``imu_accel`` for the
# pipeline (the dt + AI stages use the vertical axis); throttle
# is dropped (the engine state already carries it).
_PIPELINE_CHANNELS: tuple[str, ...] = (
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temperature",
    "fuel_flow",
    "vibration",
    "imu_accel",
    "altitude",
    "airspeed",
    "ambient_temperature",
    "ambient_pressure",
)


def _to_pipeline_sample(sample: TelemetrySample) -> SensorSample:
    """Convert an acquisition sample to the pipeline's :class:`SensorSample`.

    IMU: the pipeline carries a single ``imu_accel`` channel; we
    use the magnitude of (accel_x, accel_y, accel_z) so the
    downstream digital-twin + AI don't have to know about the
    new schema.

    Throttle: the engine state already carries the truth value,
    so the pipeline doesn't need it.
    """
    imu_mag = (
        sample.accel_x ** 2
        + sample.accel_y ** 2
        + sample.accel_z ** 2
    ) ** 0.5
    raw = {
        "rpm": sample.rpm,
        "egt": sample.egt,
        "cht": sample.cht,
        "oil_pressure": sample.oil_pressure,
        "oil_temperature": sample.oil_temperature,
        "fuel_flow": sample.fuel_flow,
        "vibration": sample.vibration,
        "imu_accel": imu_mag,
        "altitude": sample.altitude,
        "airspeed": sample.airspeed,
        "ambient_temperature": sample.ambient_temperature,
        "ambient_pressure": sample.ambient_pressure,
    }
    # Convert from the acquisition's health bitfield to a PHASE 4
    # NoiseMode: stuck / dropout / fault / normal.
    if int(sample.sensor_health) & int(SensorHealth.SENSOR_FAULT):
        # A dropout in the underlying simulator surfaces as a
        # STUCK reading (the sensor is faulted). The pipeline
        # already knows how to discount STUCK channels.
        mode = NoiseMode.STUCK
    else:
        mode = NoiseMode.NORMAL
    readings = {
        name: SensorReading(
            value=float(raw[name]),
            mode=mode,
        )
        for name in _PIPELINE_CHANNELS
    }
    return SensorSample(
        time_s=sample.timestamp_ms / 1000.0,
        readings=readings,
        engine=None,
        env=None,
    )


# ---------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------
@dataclass
class AcquisitionPipelineAdapter:
    """Drive a :class:`AsyncStreamingPipeline` from a sensor abstraction.

    The adapter is intentionally simple: it owns the abstraction
    (so it can pull samples), the pipeline (so it can submit
    them), and a :class:`TelemetryAdapter` (so the resulting
    frames carry the spec's mission/vehicle/engine IDs and a
    data-quality score).
    """

    abstraction: SensorAbstraction
    pipeline: AsyncStreamingPipeline
    adapter: TelemetryAdapter

    def __init__(
        self,
        abstraction: SensorAbstraction,
        pipeline: AsyncStreamingPipeline,
        adapter: Optional[TelemetryAdapter] = None,
        *,
        adapter_config: Optional[AdapterConfig] = None,
    ) -> None:
        self.abstraction = abstraction
        self.pipeline = pipeline
        self.adapter = adapter or TelemetryAdapter(
            cfg=adapter_config or AdapterConfig()
        )
        self._sequence: int = 0
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------
    async def step(self) -> bool:
        """Pull one sample from the abstraction and submit it.

        Returns ``True`` if a frame was submitted, ``False`` if
        the source was closed / produced no data.
        """
        if self._stopped or not self.abstraction.is_open():
            return False
        sample = self.abstraction.read(timeout_s=0.0)
        if sample is None:
            return False
        self._sequence += 1
        legacy = _to_pipeline_sample(sample)
        frame = self.adapter.adapt(legacy, sequence=self._sequence)
        return bool(await self.pipeline.submit(frame))

    def step_sync(self) -> bool:
        """Synchronous wrapper for ``step`` — drives a single sample.

        Used in tests where the event loop is owned by the test
        rather than the adapter. Returns True if a frame was
        submitted.
        """
        return _run_sync(self.step())

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    async def run(
        self,
        n_samples: Optional[int] = None,
        sleep_s: float = 0.0,
    ) -> int:
        """Submit samples until ``n_samples`` is reached or ``stop()`` is called.

        When ``n_samples`` is None the loop runs until
        :meth:`stop` is called. ``sleep_s`` is the per-iteration
        pause — useful in tests to let the pipeline drain.
        """
        produced = 0
        while not self._stopped:
            if n_samples is not None and produced >= n_samples:
                break
            ok = await self.step()
            if ok:
                produced += 1
            if sleep_s > 0.0:
                await asyncio.sleep(sleep_s)
        return produced

    def stop(self) -> None:
        """Signal the run loop to exit at the next iteration."""
        self._stopped = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def sequence(self) -> int:
        return self._sequence


def _run_sync(coro):
    """Helper: run a coroutine in a fresh event loop.

    Used by :meth:`AcquisitionPipelineAdapter.step_sync` so
    tests can drive the adapter without owning a loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


__all__ = ["AcquisitionPipelineAdapter", "_to_pipeline_sample"]
