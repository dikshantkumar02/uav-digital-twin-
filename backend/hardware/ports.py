"""
Transport ports — abstraction + Serial (real hardware) + Loopback (test).

The PHASE 14 transport does not know whether it is talking to a
real ``/dev/ttyUSB0`` device or a paired in-process loopback — it
just calls ``read`` / ``write`` on a :class:`Port` and pushes the
resulting bytes through the line framer + codec.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------
# Port protocol
# ---------------------------------------------------------------------
@runtime_checkable
class Port(Protocol):
    """Minimal read/write interface used by :class:`SerialSource`."""

    def read(self, n: int, timeout_s: float) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...


# ---------------------------------------------------------------------
# NullPort
# ---------------------------------------------------------------------
class NullPort:
    """A port that returns ``b""`` on read and discards on write.

    Useful for tests that exercise the transport's threading /
    decoding logic without feeding any real bytes. ``is_open`` is
    always True so the transport does not try to reconnect.
    """

    def read(self, n: int, timeout_s: float) -> bytes:
        return b""

    def write(self, data: bytes) -> int:
        return 0

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return True


# ---------------------------------------------------------------------
# LoopbackPort + LoopbackPair
# ---------------------------------------------------------------------
class LoopbackPort:
    """One side of an in-process :class:`LoopbackPair`.

    ``write()`` on this port enqueues bytes that ``read()`` on the
    *other* port will return. Thread-safe: the underlying queues are
    bounded and the writer blocks (or drops on overflow) per the
    pair's policy.
    """

    def __init__(
        self,
        pair: "LoopbackPair",
        *,
        side: str,
        max_queue_bytes: int = 1_048_576,
    ) -> None:
        if side not in {"a", "b"}:
            raise ValueError("side must be 'a' or 'b'")
        self._pair = pair
        self._side = side
        # The port's "input" queue is the pair's _a_in / _b_in keyed
        # by *this* side. A's writes go to _b_in (B's input). B's
        # read consumes from B's input queue.
        self._closed = False

    def read(self, n: int, timeout_s: float) -> bytes:
        if self._closed or not self._pair.is_open:
            return b""
        in_q = self._pair._in_q_for(self._side)
        try:
            data = in_q.get(timeout=timeout_s)
        except queue.Empty:
            return b""
        if n > 0 and len(data) > n:
            # Truncate to requested length; push the remainder back.
            head, tail = data[:n], data[n:]
            try:
                in_q.put_nowait(tail)
            except queue.Full:
                # The other side is not draining. Drop the tail.
                pass
            return head
        return data

    def write(self, data: bytes) -> int:
        if self._closed or not self._pair.is_open or not data:
            return 0
        # Send to the *other* side's input queue.
        other_side = "b" if self._side == "a" else "a"
        target = self._pair._in_q_for(other_side)
        try:
            target.put_nowait(bytes(data))
        except queue.Full:
            # Other side is not draining — silently drop.
            return 0
        return len(data)

    def close(self) -> None:
        self._closed = True

    @property
    def is_open(self) -> bool:
        return self._pair.is_open and not self._closed


class LoopbackPair:
    """A pair of connected :class:`LoopbackPort`s.

    A "writes" to B; B "writes" to A. Used by the PHASE 14 test
    suite to drive :class:`SerialSource` without a real /dev/tty
    device.
    """

    def __init__(self, max_queue_bytes: int = 1_048_576) -> None:
        self._max_q = int(max_queue_bytes)
        self._a_in: queue.Queue[bytes] = queue.Queue(maxsize=self._max_q)
        self._b_in: queue.Queue[bytes] = queue.Queue(maxsize=self._max_q)
        self._open = True
        self._lock = threading.Lock()
        self._a = LoopbackPort(self, side="a", max_queue_bytes=self._max_q)
        self._b = LoopbackPort(self, side="b", max_queue_bytes=self._max_q)

    def port_a(self) -> LoopbackPort:
        return self._a

    def port_b(self) -> LoopbackPort:
        return self._b

    def _in_q_for(self, side: str) -> "queue.Queue[bytes]":
        """Return the input queue that ``side`` reads from."""
        if side == "a":
            return self._a_in
        if side == "b":
            return self._b_in
        raise ValueError(f"unknown side: {side!r}")

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def close(self) -> None:
        with self._lock:
            self._open = False


# ---------------------------------------------------------------------
# SerialPort — real hardware
# ---------------------------------------------------------------------
class SerialPort:
    """A real serial port backed by :mod:`pyserial`.

    The ``serial`` module is imported lazily so the test environment
    does not pay the import cost and the loopback tests can run
    without pyserial installed. If you instantiate :class:`SerialPort`
    and pyserial is missing, you get a clear ``ImportError``.
    """

    def __init__(
        self,
        name: str,
        *,
        baudrate: int = 115200,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: float = 1,
        read_timeout_s: float = 0.5,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        self._name = str(name)
        self._baudrate = int(baudrate)
        self._bytesize = int(bytesize)
        self._parity = str(parity).upper()
        self._stopbits = float(stopbits)
        self._read_timeout_s = float(read_timeout_s)
        self._serial = None  # lazy

    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyserial is required to open a real serial port. "
                "Install it with: pip install pyserial"
            ) from exc
        parity_const = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
            "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }[self._parity]
        stopbits_const = (
            serial.STOPBITS_ONE if self._stopbits == 1
            else serial.STOPBITS_ONE_POINT_FIVE if self._stopbits == 1.5
            else serial.STOPBITS_TWO
        )
        bytesize_const = {
            5: serial.FIVEBITS, 6: serial.SIXBITS,
            7: serial.SEVENBITS, 8: serial.EIGHTBITS,
        }[self._bytesize]
        self._serial = serial.Serial(
            port=self._name,
            baudrate=self._baudrate,
            bytesize=bytesize_const,
            parity=parity_const,
            stopbits=stopbits_const,
            timeout=self._read_timeout_s,
        )

    # ------------------------------------------------------------------
    def read(self, n: int, timeout_s: float) -> bytes:
        self._ensure_open()
        # Honour the per-call timeout if it differs from the constructor's.
        prev = self._serial.timeout  # type: ignore[union-attr]
        self._serial.timeout = float(timeout_s)  # type: ignore[union-attr]
        try:
            return self._serial.read(n)  # type: ignore[union-attr]
        finally:
            self._serial.timeout = prev  # type: ignore[union-attr]

    def write(self, data: bytes) -> int:
        self._ensure_open()
        return int(self._serial.write(data))  # type: ignore[union-attr,arg-type]

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._serial = None

    @property
    def is_open(self) -> bool:
        if self._serial is None:
            return False
        try:
            return bool(self._serial.is_open)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return False


__all__ = [
    "Port",
    "SerialPort",
    "LoopbackPort",
    "LoopbackPair",
    "NullPort",
]
