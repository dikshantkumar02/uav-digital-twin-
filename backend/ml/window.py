"""
Window buffer (PHASE 9).

A small ring buffer that holds the last ``size`` ticks of inputs to
the classifier. Each "tick" is a tuple of
``(residual_frame, sensor_sample, env_state)`` so the feature
extractor can compute per-window statistics.

The buffer is **stateful** (it's a ring, not a list) so the
classifier can keep state across ticks without re-allocating.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from backend.digital_twin import ResidualFrame
from backend.environment import EnvironmentState
from backend.sensors import SensorSample


@dataclass
class WindowTick:
    """One tick of inputs to the feature extractor."""

    time_s: float
    residual: ResidualFrame
    sample: Optional[SensorSample]
    env: Optional[EnvironmentState]


class WindowBuffer:
    """Fixed-size ring buffer of :class:`WindowTick`."""

    def __init__(self, size: int = 50) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._buf: Deque[WindowTick] = deque(maxlen=int(size))
        self._size = int(size)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def append(
        self,
        time_s: float,
        residual: ResidualFrame,
        sample: Optional[SensorSample] = None,
        env: Optional[EnvironmentState] = None,
    ) -> None:
        self._buf.append(
            WindowTick(time_s=float(time_s), residual=residual, sample=sample, env=env)
        )

    def ticks(self) -> List[WindowTick]:
        """Return the current window as a list (oldest first)."""
        return list(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) == self._size


__all__ = ["WindowBuffer", "WindowTick"]
