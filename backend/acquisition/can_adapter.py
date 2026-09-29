"""
CAN interface abstraction (PHASE 18 — embedded telemetry acquisition).

CAN is the second physical transport the user's spec calls out.
Unlike the serial protocol, the CAN frame format is laid out
in the spec itself: each arbitration ID carries a fixed pair
(or triple) of float32 channels, the timestamp, and the
sensor-health bitfield.

The :class:`CanPort` protocol mirrors :class:`backend.hardware.ports.Port`
— it's a minimal hardware-independent interface that can be
backed by:

* a real ``SocketCAN`` device (``can0`` on Linux) — left for
  production deployment;
* a :class:`LoopbackCanBus` pair for tests;
* a :class:`NullCanBus` no-op for "send to nowhere" (so the
  test environment doesn't touch ``/dev/can*``).

No third-party ``python-can`` dependency — the abstraction
defines the interface, not the implementation. A future
production deployment can drop in a ``SocketCanPort`` without
touching anything else in this package.
"""

from __future__ import annotations

import queue
import struct
import threading
import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable


# ---------------------------------------------------------------------
# CAN frame-ID mapping
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CanFrameSpec:
    """The mapping from one CAN arbitration ID to its fields."""

    arbitration_id: int
    name: str
    fields: Tuple[str, ...]                # channel names on the wire
    fmt: str                                # struct format string


CAN_FRAME_MAP: Dict[int, CanFrameSpec] = {
    0x100: CanFrameSpec(0x100, "power",    ("rpm", "throttle"),                     "<ff"),
    0x101: CanFrameSpec(0x101, "thermal",  ("egt", "cht"),                          "<ff"),
    0x102: CanFrameSpec(0x102, "oil",      ("oil_pressure", "oil_temperature"),     "<ff"),
    0x103: CanFrameSpec(0x103, "fuel_vib", ("fuel_flow", "vibration"),              "<ff"),
    0x104: CanFrameSpec(0x104, "imu",      ("accel_x", "accel_y", "accel_z"),       "<fff"),
    0x105: CanFrameSpec(0x105, "air",      ("altitude", "airspeed"),                "<ff"),
    0x106: CanFrameSpec(0x106, "ambient",  ("ambient_temperature", "ambient_pressure"), "<ff"),
    0x200: CanFrameSpec(0x200, "health",   ("sensor_health",),                       "<B"),
    0x300: CanFrameSpec(0x300, "time",     ("timestamp_ms",),                        "<Q"),
}

CAN_FIELDS_BY_ID: Dict[int, Tuple[str, ...]] = {
    spec.arbitration_id: spec.fields for spec in CAN_FRAME_MAP.values()
}

# Convenience: which CAN ID carries which field. Used by the
# decoder to look up the spec from the field name.
CAN_ID_FOR_FIELD: Dict[str, int] = {}
for _spec in CAN_FRAME_MAP.values():
    for _f in _spec.fields:
        CAN_ID_FOR_FIELD[_f] = _spec.arbitration_id


# ---------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CanMessage:
    """A single CAN frame on the bus."""

    arbitration_id: int
    data: bytes
    timestamp: float                          # wall time at receive

    def decode(self) -> Dict[str, float]:
        """Decode ``self`` into a per-field dict using the spec.

        Returns an empty dict if the arbitration ID is unknown.
        """
        spec = CAN_FRAME_MAP.get(int(self.arbitration_id))
        if spec is None:
            return {}
        if len(self.data) != struct.calcsize(spec.fmt):
            return {}
        values = struct.unpack(spec.fmt, self.data)
        return dict(zip(spec.fields, values))


# ---------------------------------------------------------------------
# Port protocol
# ---------------------------------------------------------------------
@runtime_checkable
class CanPort(Protocol):
    """Minimal CAN port interface.

    Mirrors :class:`backend.hardware.ports.Port`. Send / recv are
    blocking-with-timeout; ``close`` is idempotent.
    """

    def send(self, frame_id: int, data: bytes) -> None: ...
    def recv(self, timeout_s: float) -> Optional[CanMessage]: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...


# ---------------------------------------------------------------------
# NullCanBus
# ---------------------------------------------------------------------
class NullCanBus:
    """A no-op CAN bus.

    Every ``send`` is a silent drop. ``recv`` returns ``None``
    immediately. Used by tests and "send to nowhere" deployments
    so the test environment never touches ``/dev/can*``.
    """

    def send(self, frame_id: int, data: bytes) -> None:
        return None

    def recv(self, timeout_s: float) -> Optional[CanMessage]:
        return None

    def close(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return True


# ---------------------------------------------------------------------
# LoopbackCanBus
# ---------------------------------------------------------------------
class LoopbackCanBus:
    """A pair of in-process CAN buses for tests.

    :meth:`port_a` and :meth:`port_b` return two :class:`_LoopbackSide`
    instances — ``port_a``'s ``send`` lands in ``port_b``'s
    ``recv`` queue, and vice versa. The pair is thread-safe
    (backed by ``queue.Queue``) and bounded.
    """

    def __init__(self, max_queue_messages: int = 1024) -> None:
        self._max_q = int(max_queue_messages)
        self._a_in: "queue.Queue[CanMessage]" = queue.Queue(maxsize=self._max_q)
        self._b_in: "queue.Queue[CanMessage]" = queue.Queue(maxsize=self._max_q)
        self._open = True
        self._lock = threading.Lock()
        self._a = _LoopbackSide(self, "a")
        self._b = _LoopbackSide(self, "b")

    def port_a(self) -> "_LoopbackSide":
        return self._a

    def port_b(self) -> "_LoopbackSide":
        return self._b

    def _recv_q_for(self, side: str) -> "queue.Queue[CanMessage]":
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


class _LoopbackSide:
    """One side of a :class:`LoopbackCanBus` pair."""

    def __init__(self, pair: LoopbackCanBus, side: str) -> None:
        if side not in {"a", "b"}:
            raise ValueError("side must be 'a' or 'b'")
        self._pair = pair
        self._side = side

    def send(self, frame_id: int, data: bytes) -> None:
        if not self._pair.is_open or not data:
            return None
        target_side = "b" if self._side == "a" else "a"
        target = self._pair._recv_q_for(target_side)
        try:
            target.put_nowait(
                CanMessage(
                    arbitration_id=int(frame_id),
                    data=bytes(data),
                    timestamp=_time.perf_counter(),
                )
            )
        except queue.Full:
            return None
        return None

    def recv(self, timeout_s: float) -> Optional[CanMessage]:
        if not self._pair.is_open:
            return None
        q = self._pair._recv_q_for(self._side)
        try:
            return q.get(timeout=float(timeout_s))
        except queue.Empty:
            return None

    def close(self) -> None:
        # The pair is shared; closing is a pair-level operation.
        self._pair.close()
        return None

    @property
    def is_open(self) -> bool:
        return self._pair.is_open


# ---------------------------------------------------------------------
# Encoder / decoder helpers
# ---------------------------------------------------------------------
def encode_sample_to_can_messages(sample) -> List[CanMessage]:
    """Encode a :class:`TelemetrySample` as a list of CAN messages.

    One message per arbitration ID — the spec's frame layout.
    The order is the spec's own ordering so the receiver can
    process them in any order without losing fields.
    """
    if not hasattr(sample, "to_dict"):
        raise TypeError("sample must have a to_dict() method")
    d = sample.to_dict()
    out: List[CanMessage] = []
    now = _time.perf_counter()
    for spec in CAN_FRAME_MAP.values():
        if spec.name == "health":
            out.append(CanMessage(
                arbitration_id=spec.arbitration_id,
                data=struct.pack(spec.fmt, int(d["sensor_health"]) & 0xFF),
                timestamp=now,
            ))
        elif spec.name == "time":
            out.append(CanMessage(
                arbitration_id=spec.arbitration_id,
                data=struct.pack(spec.fmt, int(d["timestamp"]) & 0xFFFFFFFFFFFFFFFF),
                timestamp=now,
            ))
        else:
            values = [float(d[f]) for f in spec.fields]
            out.append(CanMessage(
                arbitration_id=spec.arbitration_id,
                data=struct.pack(spec.fmt, *values),
                timestamp=now,
            ))
    return out


def decode_can_messages_to_dict(messages: List[CanMessage]) -> Dict[str, float]:
    """Decode a list of CAN messages into a per-field dict.

    The :class:`TelemetrySample` constructor reads its fields
    by name; this helper is the bus-side counterpart. Stale or
    unknown IDs are silently ignored.
    """
    out: Dict[str, float] = {}
    for m in messages:
        out.update(m.decode())
    return out


__all__ = [
    "CAN_FRAME_MAP",
    "CAN_FIELDS_BY_ID",
    "CAN_ID_FOR_FIELD",
    "CanFrameSpec",
    "CanMessage",
    "CanPort",
    "LoopbackCanBus",
    "NullCanBus",
    "encode_sample_to_can_messages",
    "decode_can_messages_to_dict",
]
