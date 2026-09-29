"""
ReplayController (PHASE 17 — real-time streaming pipeline).

Streams a recorded ``wire.bin`` (PHASE 15 HIL recording) into an
:class:`~backend.telemetry.pipeline.AsyncStreamingPipeline` at
0.5×, 1×, 2×, or 5× the original mission pacing.

The controller does **not** use the PHASE 15 :class:`WireReplayer`
daemon-thread machinery — that path writes raw bytes into a
:class:`LoopbackPort` for a :class:`SerialSource` to drain. For
the async pipeline we want **explicit per-frame pacing** so the
recorded frame's ``time_s`` delta directly drives ``asyncio.sleep``.

Pacing
------

For every frame in the recording, the controller:

1. Decodes the wire bytes into a :class:`TelemetryFrame` using
   :class:`FrameCodec.decode`.
2. Computes the delta against the previous frame's ``time_s``.
3. Sleeps ``delta / speed_multiplier`` so a 1× replay
   reproduces the original pacing.
4. Submits the frame to the async pipeline.

If the recorded frame stream is exhausted, the controller stops
gracefully. Calling :meth:`stop` from outside cancels the
in-flight sleep so a long replay can be terminated early.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.hardware import FrameCodec
from backend.telemetry import AsyncStreamingPipeline, TelemetryFrame


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Speeds
# ---------------------------------------------------------------------
class ReplaySpeed(str, Enum):
    """Replay speed multipliers."""

    SPEED_0_5X = "0.5x"
    SPEED_1X = "1x"
    SPEED_2X = "2x"
    SPEED_5X = "5x"

    @property
    def multiplier(self) -> float:
        return {
            ReplaySpeed.SPEED_0_5X: 0.5,
            ReplaySpeed.SPEED_1X: 1.0,
            ReplaySpeed.SPEED_2X: 2.0,
            ReplaySpeed.SPEED_5X: 5.0,
        }[self]


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------
@dataclass
class ReplayStats:
    """Counters for a completed replay run."""

    frames_replayed: int = 0
    frames_dropped: int = 0
    elapsed_wall_s: float = 0.0
    speed: ReplaySpeed = ReplaySpeed.SPEED_1X

    def to_dict(self) -> dict:
        return {
            "frames_replayed": self.frames_replayed,
            "frames_dropped": self.frames_dropped,
            "elapsed_wall_s": float(self.elapsed_wall_s),
            "speed": self.speed.value,
        }


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------
class ReplayController:
    """Replay a recorded wire.bin at a chosen speed.

    Parameters
    ----------
    wire_path:
        Path to the recorded ``wire.bin`` from PHASE 15.
    pipeline:
        The :class:`AsyncStreamingPipeline` to feed.
    speed:
        Playback speed multiplier.
    """

    def __init__(
        self,
        wire_path: Path,
        pipeline: AsyncStreamingPipeline,
        *,
        speed: ReplaySpeed = ReplaySpeed.SPEED_1X,
    ) -> None:
        self._path = Path(wire_path)
        self._pipeline = pipeline
        self._speed = ReplaySpeed(speed)
        self._stop_event = asyncio.Event()
        self._stats = ReplayStats(speed=self._speed)

    @property
    def stats(self) -> ReplayStats:
        return self._stats

    @property
    def speed(self) -> ReplaySpeed:
        return self._speed

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Async runner
    # ------------------------------------------------------------------
    async def run(self) -> ReplayStats:
        """Replay the recording into the pipeline. Returns stats."""
        if not self._path.is_file():
            raise FileNotFoundError(f"wire.bin not found: {self._path}")
        t0 = _time.perf_counter()
        last_time_s: Optional[float] = None
        multiplier = self._speed.multiplier
        # We read the wire.bin line-by-line and decode each line
        # directly — this is the same path the SerialSource uses,
        # minus the port indirection.
        with self._path.open("rb") as fh:
            for raw_line in fh:
                if self._stop_event.is_set():
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    frame = FrameCodec.decode(line, crc="ccitt")
                except Exception as exc:  # noqa: BLE001
                    log.warning("replay: decode error: %s", exc)
                    self._stats.frames_dropped += 1
                    continue
                # Pacing.
                if last_time_s is not None:
                    delta = float(frame.time_s) - float(last_time_s)
                    if delta < 0.0:
                        delta = 0.0
                    sleep_for = delta / multiplier if multiplier > 0 else 0.0
                    if sleep_for > 0.0:
                        try:
                            await asyncio.wait_for(
                                self._stop_event.wait(), timeout=sleep_for
                            )
                            # If wait() returned, the stop event fired.
                            break
                        except asyncio.TimeoutError:
                            pass
                last_time_s = float(frame.time_s)
                # Submit to the pipeline.
                ok = await self._pipeline.submit(frame)
                if ok:
                    self._stats.frames_replayed += 1
                else:
                    self._stats.frames_dropped += 1
        self._stats.elapsed_wall_s = _time.perf_counter() - t0
        return self._stats


__all__ = ["ReplayController", "ReplaySpeed", "ReplayStats"]
