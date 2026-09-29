"""
In-memory telemetry queue with bounded size.

Provides blocking + non-blocking push, latest-N peek, and frame-count
metrics. Used between the stream source and the preprocessor; the
preprocessor + downstream consumers also push through the same queue
abstraction in tests (the dashboard in PHASE 13 will use its own).
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .frame import FrameStatus, TelemetryFrame


class QueueFullError(RuntimeError):
    """Raised on a non-blocking put into a full queue."""


@dataclass
class QueueStats:
    """Running counters for queue behaviour."""

    pushed: int = 0
    popped: int = 0
    dropped_overflow: int = 0
    high_watermark: int = 0

    def to_dict(self) -> dict:
        return {
            "pushed": self.pushed,
            "popped": self.popped,
            "dropped_overflow": self.dropped_overflow,
            "high_watermark": self.high_watermark,
        }


class TelemetryQueue:
    """Thread-safe bounded FIFO queue of :class:`TelemetryFrame`."""

    def __init__(self, max_size: int = 1024, drop_on_overflow: bool = True) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._max = int(max_size)
        self._drop = bool(drop_on_overflow)
        self._q: Deque[TelemetryFrame] = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stats = QueueStats()
        self._closed = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def max_size(self) -> int:
        return self._max

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                pushed=self._stats.pushed,
                popped=self._stats.popped,
                dropped_overflow=self._stats.dropped_overflow,
                high_watermark=self._stats.high_watermark,
            )

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------
    def push(self, frame: TelemetryFrame) -> bool:
        """Push a frame.

        If the queue is full and ``drop_on_overflow`` is True, the oldest
        frame is discarded to make room and ``dropped_overflow`` is
        incremented. Returns True if the frame is in the queue after
        the call. Otherwise returns False (queue full + no drop policy).
        """
        with self._not_empty:
            if self._closed:
                return False
            if len(self._q) >= self._max:
                if self._drop:
                    old = self._q.popleft()
                    old.status = FrameStatus.DROPPED
                    self._stats.dropped_overflow += 1
                    self._stats.pushed += 1   # counted as a "push" attempt that dropped an old
                    self._q.append(frame)
                    self._stats.high_watermark = max(self._stats.high_watermark, len(self._q))
                    self._not_empty.notify()
                    return True
                return False
            self._q.append(frame)
            self._stats.pushed += 1
            self._stats.high_watermark = max(self._stats.high_watermark, len(self._q))
            self._not_empty.notify()
            return True

    def push_nowait(self, frame: TelemetryFrame) -> bool:
        """Alias for ``push``; kept for clarity at call sites."""
        return self.push(frame)

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------
    def pop(self, timeout_s: Optional[float] = None) -> Optional[TelemetryFrame]:
        """Pop the oldest frame. Returns None on close / timeout."""
        with self._not_empty:
            if not self._q and timeout_s is None:
                return None
            if not self._q and not self._closed:
                self._not_empty.wait(timeout=timeout_s)
            if not self._q:
                return None
            frame = self._q.popleft()
            self._stats.popped += 1
            return frame

    def drain(self) -> List[TelemetryFrame]:
        """Drain the queue into a list (oldest first)."""
        out: List[TelemetryFrame] = []
        with self._lock:
            out.extend(self._q)
            self._q.clear()
            self._stats.popped += len(out)
            return out

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
