"""
WireReplayer — streams the bytes of a recorded ``wire.bin`` into a
:class:`~backend.hardware.ports.Port` so a downstream
:class:`~backend.hardware.transport.SerialSource` sees the same bytes
it would see from a real ``/dev/tty*`` device.

The replay is the centrepiece of the PHASE 15 HIL validation flow:

* :class:`WireRecorder` (this package) writes a ``wire.bin`` of AERO
  frames during a live or synthetic run.
* :class:`WireReplayer` reads that file back in chunks, writes each
  chunk to a :class:`LoopbackPort`, and the paired SerialSource
  decodes the bytes into :class:`TelemetryFrame` objects on the same
  :class:`TelemetryQueue` the synthetic ``StreamSource`` uses.
* The downstream pipeline (``Preprocessor`` → digital twin → …)
  runs unchanged — proving that the wire protocol is a true
  round-trip for live hardware.

By default the replayer spawns a background daemon thread (matches
``SerialSource.start()``) so callers can run a full pipeline pass
without blocking the calling thread. Tests can call
:meth:`replay_sync` to drive the replay in-line.
"""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path
from typing import Optional

from backend.hardware.ports import Port


# ---------------------------------------------------------------------
# WireReplayer
# ---------------------------------------------------------------------
class WireReplayer:
    """Replays a recorded ``wire.bin`` into a :class:`Port`.

    Parameters
    ----------
    path:
        Path to a binary file produced by
        :class:`~backend.hil.recorder.WireRecorder`. The file is read
        sequentially and its bytes are pushed verbatim into ``port``.
    port:
        The destination port. A :class:`LoopbackPort` is the natural
        choice (paired with a :class:`SerialSource`); a real
        :class:`SerialPort` is the production path.
    chunk_bytes:
        Max bytes pushed per ``port.write()`` call. Defaults to 4096
        — matches :class:`SerialSource`'s default read chunk size so
        a single record round-trips in 1–2 write/read pairs per
        ~6 KB frame.
    """

    def __init__(
        self,
        path: Path,
        port: Port,
        *,
        chunk_bytes: int = 4096,
    ) -> None:
        self._path = Path(path)
        self._port = port
        self._chunk = max(1, int(chunk_bytes))
        self._bytes_replayed = 0
        self._lines_replayed = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def bytes_replayed(self) -> int:
        with self._lock:
            return self._bytes_replayed

    @property
    def lines_replayed(self) -> int:
        with self._lock:
            return self._lines_replayed

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Optional[BaseException]:
        with self._lock:
            return self._error

    # ------------------------------------------------------------------
    # Sync replay
    # ------------------------------------------------------------------
    def replay_sync(
        self,
        *,
        rate_hz: Optional[float] = None,
    ) -> int:
        """Replay the file synchronously, return bytes written.

        Parameters
        ----------
        rate_hz:
            If set, sleep ``1.0 / rate_hz`` between chunks to
            throttle the replay. ``None`` writes as fast as the
            :class:`Port` will accept (test / offline mode).

        The method counts ``\\n`` bytes in each written chunk into
        :attr:`lines_replayed` so a caller can verify that the right
        number of AERO frames were pushed.
        """
        if not self._path.is_file():
            raise FileNotFoundError(f"wire.bin not found: {self._path}")
        sleep_s = (1.0 / float(rate_hz)) if rate_hz and rate_hz > 0 else 0.0
        bytes_written = 0
        with self._path.open("rb") as fh:
            while True:
                if self._stop_event.is_set():
                    break
                chunk = fh.read(self._chunk)
                if not chunk:
                    break
                n = self._port.write(chunk)
                bytes_written += n
                self._bump(n, chunk.count(b"\n"))
                if sleep_s > 0:
                    if self._stop_event.wait(sleep_s):
                        break
        return bytes_written

    # ------------------------------------------------------------------
    # Async replay (daemon thread)
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn a daemon thread that calls :meth:`replay_sync`."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._done_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="WireReplayer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        """Signal the thread to stop and wait for it to exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._thread = None

    def wait(self, timeout_s: Optional[float] = None) -> bool:
        """Block until the replay thread finishes.

        Returns ``True`` if the thread finished within ``timeout_s``,
        ``False`` otherwise.
        """
        return self._done_event.wait(timeout=timeout_s)

    def _run(self) -> None:
        try:
            self.replay_sync()
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                self._error = exc
        finally:
            self._done_event.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _bump(self, n: int, lines: int) -> None:
        with self._lock:
            self._bytes_replayed += int(n)
            self._lines_replayed += int(lines)


__all__ = ["WireReplayer"]
