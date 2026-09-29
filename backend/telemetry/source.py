"""
Stream source — drives the sensor bundle and pushes frames onto a queue.

This is the producer side of the pipeline. The consumer side is the
preprocessor (PHASE 5) and the digital twin (PHASE 6). The source is
the only place that knows about wall-clock vs. simulated time; the
rest of the system sees the ``TelemetryFrame`` API only.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import List, Optional

from backend.sensors import SensorBundle

from .frame import TelemetryFrame
from .latency import LatencyTracker
from .queue import TelemetryQueue


@dataclass
class StreamSource:
    """Drives a sensor bundle onto a telemetry queue."""

    bundle: SensorBundle
    queue: TelemetryQueue
    tracker: Optional[LatencyTracker] = None
    _sequence: int = 0
    _closed: bool = False

    # ------------------------------------------------------------------
    def tick(
        self,
        degradation_severity: float = 0.0,
        vibration_external: float = 0.0,
    ) -> Optional[TelemetryFrame]:
        """Advance the simulator by one step, build a frame, push it."""
        if self._closed:
            return None
        t0 = _time.perf_counter()
        sample = self.bundle.tick(
            degradation_severity=degradation_severity,
            vibration_external=vibration_external,
        )
        if self.tracker is not None:
            # Sensor read time goes under "ingestion".
            self.tracker.record("ingestion", _time.perf_counter() - t0)
        self._sequence += 1
        frame = TelemetryFrame(
            sequence=self._sequence,
            time_s=sample.time_s,
            produced_wall_time=_time.perf_counter(),
            sample=sample,
        )
        if not self.queue.push(frame):
            # Queue full + no drop policy: caller decides what to do.
            return None
        return frame

    def run(
        self,
        degradation_severity: float = 0.0,
        max_steps: Optional[int] = None,
        vibration_external: Optional[List[float]] = None,
    ) -> int:
        """Run the full mission; push a frame for every tick.

        Returns the number of frames pushed (i.e. not dropped).
        """
        n_pushed = 0
        for i, sample in enumerate(
            self.bundle.run(
                degradation_severity=degradation_severity,
                vibration_external=vibration_external,
            )
        ):
            if max_steps is not None and i >= max_steps:
                break
            t0 = _time.perf_counter()
            self._sequence += 1
            frame = TelemetryFrame(
                sequence=self._sequence,
                time_s=sample.time_s,
                produced_wall_time=_time.perf_counter(),
                sample=sample,
            )
            if self.tracker is not None:
                self.tracker.record("ingestion", _time.perf_counter() - t0)
            if self.queue.push(frame):
                n_pushed += 1
        return n_pushed

    def close(self) -> None:
        self._closed = True
        self.queue.close()
