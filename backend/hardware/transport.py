"""
Serial transport — reads bytes from a :class:`Port`, decodes them
via :class:`FrameCodec`, and pushes the resulting
:class:`TelemetryFrame` objects onto a :class:`TelemetryQueue`.

The transport owns a single background daemon thread that does the
read → framer → decode → push loop. The downstream pipeline
(``Preprocessor`` → digital twin → …) is unchanged from PHASE 5+
and is the same code path the synthetic :class:`StreamSource` uses.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from dataclasses import dataclass
from typing import Iterable, Optional, Set

from backend.telemetry import LatencyTracker, TelemetryQueue

from .framing import LineFramer
from .ports import Port
from .protocol import (
    FrameCodec,
    FrameCrcError,
    FrameFormatError,
    FrameValueError,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------
@dataclass
class TransportStats:
    """Running counters for transport health."""

    bytes_read: int = 0
    frames_pushed: int = 0
    crc_errors: int = 0
    format_errors: int = 0
    value_errors: int = 0
    reconnects: int = 0

    def to_dict(self) -> dict:
        return {
            "bytes_read": self.bytes_read,
            "frames_pushed": self.frames_pushed,
            "crc_errors": self.crc_errors,
            "format_errors": self.format_errors,
            "value_errors": self.value_errors,
            "reconnects": self.reconnects,
        }


# ---------------------------------------------------------------------
# SerialSource
# ---------------------------------------------------------------------
class SerialSource:
    """Read frames from a :class:`Port` and push them onto a queue.

    Parameters
    ----------
    port:
        A :class:`Port` (or any object that quacks like one). For
        real hardware, use :class:`SerialPort`. For tests, use
        :class:`LoopbackPort` or :class:`NullPort`.
    queue:
        The :class:`TelemetryQueue` to push decoded frames onto.
    expected_channels:
        Optional allowlist of channel names. ``None`` accepts any
        channel in ``SENSOR_CHANNELS``. An empty set rejects all.
    latency_tracker:
        Optional :class:`LatencyTracker` — the decode + push stage
        is recorded under the ``"transport"`` name.
    reconnect_on_error:
        If True, transient read errors (USB unplug, port closed by
        peer) cause the reader to sleep ``reconnect_backoff_s`` and
        try to re-open. If False, the first error stops the thread.
    reconnect_backoff_s:
        Sleep between reconnect attempts.
    read_chunk_bytes:
        Max bytes per ``Port.read()`` call.
    crc:
        ``"ccitt"`` (default) or ``"none"`` — passed to
        :class:`FrameCodec`.
    """

    def __init__(
        self,
        port: Port,
        queue: TelemetryQueue,
        *,
        expected_channels: Optional[Iterable[str]] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        reconnect_on_error: bool = True,
        reconnect_backoff_s: float = 1.0,
        read_chunk_bytes: int = 4096,
        crc: str = "ccitt",
        poll_sleep_s: float = 0.005,
    ) -> None:
        self._port = port
        self._queue = queue
        self._allowed: Optional[Set[str]] = (
            None if expected_channels is None else set(expected_channels)
        )
        self._tracker = latency_tracker
        self._reconnect_on_error = bool(reconnect_on_error)
        self._reconnect_backoff_s = float(reconnect_backoff_s)
        self._read_chunk = max(1, int(read_chunk_bytes))
        self._crc = str(crc)
        self._poll_sleep_s = float(poll_sleep_s)

        self._stats = TransportStats()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._framer = LineFramer()
        self._started_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def stats(self) -> TransportStats:
        return self._stats

    @property
    def port(self) -> Port:
        return self._port

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the background reader thread.

        Calling ``start()`` while the source is already running is a
        no-op. Calling it after ``stop()`` re-creates the thread.
        """
        with self._started_lock:
            if self._started and self.is_running():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="SerialSource", daemon=True
            )
            self._thread.start()
            self._started = True

    def stop(self, timeout_s: float = 1.0) -> None:
        """Signal the reader thread to stop and wait for it to exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
        with self._started_lock:
            self._started = False
            self._thread = None

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._drain_once()
            except Exception as exc:  # noqa: BLE001
                # Treat anything as a transient read error.
                if not self._reconnect_on_error:
                    log.error("serial transport fatal error: %s", exc)
                    return
                self._stats.reconnects += 1
                log.warning(
                    "serial transport error, reconnecting in %.1fs: %s",
                    self._reconnect_backoff_s,
                    exc,
                )
                if self._stop_event.wait(self._reconnect_backoff_s):
                    return
        # Final drain — best-effort, do not retry on error.
        try:
            self._port.close()
        except Exception:  # noqa: BLE001
            pass

    def _drain_once(self) -> None:
        """Read one chunk, frame it, decode and push."""
        if not self._port.is_open:
            # Some ports (LoopbackPort) start open; SerialPort only
            # opens on first read. We attempt the read; if it
            # raises, the outer loop reconnects.
            pass
        chunk = self._port.read(self._read_chunk, timeout_s=self._poll_sleep_s)
        if not chunk:
            return
        self._stats.bytes_read += len(chunk)
        for line in self._framer.feed(chunk):
            self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        try:
            if self._tracker is not None:
                with self._tracker.stage("transport"):
                    frame = FrameCodec.decode(
                        line,
                        crc=self._crc,
                        allowed_channels=self._allowed,
                    )
            else:
                frame = FrameCodec.decode(
                    line,
                    crc=self._crc,
                    allowed_channels=self._allowed,
                )
        except FrameCrcError as exc:
            self._stats.crc_errors += 1
            log.debug("crc error: %s", exc)
            return
        except FrameFormatError as exc:
            self._stats.format_errors += 1
            log.debug("format error: %s", exc)
            return
        except FrameValueError as exc:
            self._stats.value_errors += 1
            log.debug("value error: %s", exc)
            return
        # Stamp the wall clock at push time (after decode) so the
        # latency tracker measures the full decode + push.
        frame.produced_wall_time = _time.perf_counter()
        if self._queue.push(frame):
            self._stats.frames_pushed += 1
        # If the queue is full and not dropping, push() returns
        # False; we silently drop the counter for that case. The
        # queue's own drop policy is the source of truth.

    # ------------------------------------------------------------------
    # Test-friendly single-shot
    # ------------------------------------------------------------------
    def pump_once(self, timeout_s: float = 0.0) -> int:
        """Drain one chunk synchronously.

        Useful in tests that want to drive the transport's logic
        without spawning a thread. Returns the number of frames
        pushed.
        """
        before = self._stats.frames_pushed
        try:
            chunk = self._port.read(
                self._read_chunk, timeout_s=timeout_s
            )
        except Exception:  # noqa: BLE001
            return 0
        if not chunk:
            return 0
        self._stats.bytes_read += len(chunk)
        for line in self._framer.feed(chunk):
            self._handle_line(line)
        return self._stats.frames_pushed - before


__all__ = ["SerialSource", "TransportStats"]
