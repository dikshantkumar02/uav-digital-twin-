"""
Telemetry preprocessor.

Responsibilities:

1. **Dropout handling** — if a channel's value is None, forward-fill
   from the most recent valid value, but only if that value is no
   older than ``max_age_s``. Otherwise the frame's status is downgraded.
2. **Validation** — count the number of channels with no valid value
   within the age budget. If more than ``max_invalid_channels`` are
   unusable, the frame is marked ``INVALID`` and downstream consumers
   must not act on it.
3. **Latency** — wrap the work in a ``LatencyTracker`` stage.
4. **Per-frame notes** — record which channels were filled, which were
   lost, and the max channel age.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .frame import FrameStatus, TelemetryFrame
from .latency import LatencyTracker


@dataclass
class _ChannelState:
    last_value: Optional[float] = None
    last_valid_time_s: Optional[float] = None
    age_s: float = 0.0


class Preprocessor:
    """Channel-aligned dropout-tolerant frame processor."""

    def __init__(
        self,
        max_dropout_age_s: float = 2.0,
        max_invalid_channels: int = 3,
        latency_tracker: Optional[LatencyTracker] = None,
    ) -> None:
        if max_dropout_age_s < 0:
            raise ValueError("max_dropout_age_s must be >= 0")
        if max_invalid_channels < 0:
            raise ValueError("max_invalid_channels must be >= 0")
        self._max_age = float(max_dropout_age_s)
        self._max_invalid = int(max_invalid_channels)
        self._tracker = latency_tracker
        self._state: Dict[str, _ChannelState] = {}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._state.clear()

    def process(self, frame: TelemetryFrame) -> TelemetryFrame:
        """Process one frame, mutating it in place and returning it."""
        if self._tracker is not None:
            with self._tracker.stage("preprocessing"):
                self._process_inplace(frame)
        else:
            self._process_inplace(frame)
        frame.preprocessed_wall_time = _time.perf_counter()
        return frame

    # ------------------------------------------------------------------
    def _process_inplace(self, frame: TelemetryFrame) -> None:
        sample = frame.sample
        if sample is None:
            frame.status = FrameStatus.INVALID
            return

        n_invalid = 0
        n_filled = 0
        max_age = 0.0
        for name, reading in sample.readings.items():
            st = self._state.setdefault(name, _ChannelState())
            if reading.value is None:
                # Dropout. Try forward-fill.
                if st.last_value is not None and st.last_valid_time_s is not None:
                    age = frame.time_s - st.last_valid_time_s
                    if age <= self._max_age:
                        reading.value = st.last_value
                        reading.mode = reading.mode  # leave the original mode
                        n_filled += 1
                        st.age_s = age
                        max_age = max(max_age, age)
                        frame.notes[f"fill.{name}"] = f"age={age:.2f}s"
                        continue
                # No usable value within the age budget.
                n_invalid += 1
                st.age_s = float("inf")
            else:
                st.last_value = float(reading.value)
                st.last_valid_time_s = frame.time_s
                st.age_s = 0.0

        frame.notes["n_filled"] = str(n_filled)
        frame.notes["n_invalid"] = str(n_invalid)
        frame.notes["max_age_s"] = f"{max_age:.3f}"

        if n_invalid > self._max_invalid:
            frame.status = FrameStatus.INVALID
        elif n_filled > 0 or n_invalid > 0:
            frame.status = FrameStatus.STALE
        else:
            frame.status = FrameStatus.OK
