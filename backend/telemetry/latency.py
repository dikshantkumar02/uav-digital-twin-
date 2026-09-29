"""
End-to-end latency tracking.

Latency is measured across named stages. Each stage records
``start_wall_time`` and ``end_wall_time`` for the frames it touches,
and emits a :class:`StageLatency` summary. The pipeline target is
< 2 s end-to-end; intermediate stages must each be well under that.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StageLatency:
    """Summary of one pipeline stage."""

    name: str
    samples: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    last_s: float = 0.0

    def record(self, dt_s: float) -> None:
        self.samples += 1
        self.total_s += dt_s
        if dt_s > self.max_s:
            self.max_s = dt_s
        self.last_s = dt_s

    @property
    def mean_s(self) -> float:
        return self.total_s / self.samples if self.samples else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "samples": self.samples,
            "mean_s": self.mean_s,
            "max_s": self.max_s,
            "last_s": self.last_s,
        }


class LatencyTracker:
    """Track latency across named pipeline stages.

    Usage::

        tracker = LatencyTracker()
        with tracker.stage("preprocessing"):
            ...do work...
        # later:
        with tracker.stage("digital_twin"):
            ...
    """

    def __init__(self) -> None:
        self._stages: Dict[str, StageLatency] = {}
        self._open: Dict[str, float] = {}

    # ------------------------------------------------------------------
    @property
    def stages(self) -> Dict[str, StageLatency]:
        return self._stages

    def stage(self, name: str) -> "_Stage":
        return _Stage(self, name)

    def record(self, name: str, dt_s: float) -> None:
        s = self._stages.get(name)
        if s is None:
            s = StageLatency(name=name)
            self._stages[name] = s
        s.record(dt_s)

    def summary(self) -> Dict[str, StageLatency]:
        return dict(self._stages)

    def total_last_s(self) -> float:
        """Sum of ``last_s`` across all stages recorded so far."""
        return sum(s.last_s for s in self._stages.values())

    def reset(self) -> None:
        self._stages.clear()
        self._open.clear()


class _Stage:
    """Context manager that records a single stage's elapsed time."""

    def __init__(self, tracker: LatencyTracker, name: str) -> None:
        self._tracker = tracker
        self._name = name
        self._t0: float = 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = _time.perf_counter() - self._t0
        self._tracker.record(self._name, dt)
