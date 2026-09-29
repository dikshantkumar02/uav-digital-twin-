"""
Wire-protocol codec — TelemetryFrame ↔ bytes.

Format (one ASCII line, ``\\n`` terminated)::

    AERO,1,<seq>,<time_s>,<status>,<ch1>=<v1>:<m1>,<ch2>=<v2>:<m2>,...,*<crc16>\\n

* ``AERO,1,``     magic + version.
* ``<seq>``       uint32 sequence number.
* ``<time_s>``    float seconds since mission start.
* ``<status>``    one of OK / STALE / INVALID / DROPPED.
* ``<ch>=<v>:<m>``  repeated for each channel. ``<v>`` is a float; an
                  empty value (e.g. ``rpm=``) means dropout (``None``).
                  ``<m>`` is a single character (see
                  :data:`backend.hardware.wire.NOISE_MODE_CODE`).
* ``*<crc16>``    4 hex digits, CRC-16-CCITT/XMODEM over the body
                  (everything between ``AERO,1,`` and ``*``).

Decoding is strict: malformed lines, bad magic, unknown channels,
missing fields, or CRC mismatches all raise one of the typed
exceptions below. The transport catches them, increments a counter,
and continues with the next line — a single bad frame never stalls
the queue.
"""

from __future__ import annotations

import time as _time
from typing import Dict, Optional, Set

from backend.sensors import NoiseMode, SensorReading, SensorSample
from backend.telemetry import FrameStatus, TelemetryFrame

from .crc import crc16_hex
from .wire import NOISE_MODE_CODE, NOISE_MODE_FROM_CODE, SENSOR_NAME_SET


MAGIC = b"AERO"
VERSION = 1


# ---------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------
class FrameFormatError(ValueError):
    """Bad magic, bad version, missing fields, or unparsable structure."""


class FrameCrcError(ValueError):
    """CRC mismatch — the frame was corrupted in transit."""


class FrameValueError(ValueError):
    """Unknown channel name, unparsable value, or unrecognised mode."""


# ---------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------
class FrameCodec:
    """Encoder + decoder for the AERO wire protocol."""

    MAGIC = MAGIC
    VERSION = VERSION
    HEADER = b"AERO,1,"  # body prefix; CRC covers everything after this
    TRAILER_SEP = b"*"   # separates body from CRC
    LINE_END = b"\n"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @staticmethod
    def encode(
        frame: TelemetryFrame,
        *,
        crc: str = "ccitt",
    ) -> bytes:
        """Encode a :class:`TelemetryFrame` as a single line of bytes.

        Parameters
        ----------
        frame:
            Source frame. ``frame.sequence``, ``frame.time_s``, and
            ``frame.status`` go into the header; every reading in
            ``frame.sample.readings`` becomes a ``<ch>=<v>:<m>`` token.
        crc:
            ``"ccitt"`` (default) appends a 4-hex-digit CRC trailer.
            ``"none"`` skips CRC — useful for unit tests.
        """
        body = FrameCodec._encode_body(frame)
        if crc == "none":
            return body + FrameCodec.LINE_END
        if crc != "ccitt":
            raise ValueError(f"unknown crc policy: {crc!r}")
        return body + b"*" + crc16_hex(body).encode() + FrameCodec.LINE_END

    @staticmethod
    def _encode_body(frame: TelemetryFrame) -> bytes:
        parts = [
            FrameCodec.HEADER,
            f"{int(frame.sequence)},".encode(),
            f"{float(frame.time_s):.6f},".encode(),
            f"{frame.status.value},".encode(),
        ]
        # Channels in a deterministic order (sorted) so two encoders
        # running on the same input produce byte-identical lines.
        for name in sorted(frame.sample.readings.keys()):
            reading = frame.sample.readings[name]
            if reading.value is None:
                value_str = ""
            else:
                value_str = f"{float(reading.value):.6f}"
            mode_code = NOISE_MODE_CODE[reading.mode]
            parts.append(f"{name}={value_str}:{mode_code},".encode())
        # Strip the trailing comma.
        out = b"".join(parts)
        if out.endswith(b","):
            out = out[:-1]
        return out

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    @staticmethod
    def decode(
        line: bytes,
        *,
        crc: str = "ccitt",
        allowed_channels: Optional[Set[str]] = None,
        now_wall_time: Optional[float] = None,
    ) -> TelemetryFrame:
        """Decode one line of wire bytes into a :class:`TelemetryFrame`.

        Parameters
        ----------
        line:
            The full line (no trailing newline). The decoder will
            strip a trailing CR for CRLF inputs.
        crc:
            ``"ccitt"`` (default) verifies the 4-hex-digit CRC.
            ``"none"`` skips verification.
        allowed_channels:
            Optional channel-name allowlist. ``None`` accepts any
            channel name in :data:`SENSOR_NAME_SET`. An empty set
            rejects every channel.
        now_wall_time:
            Override for the produced_wall_time field. Defaults to
            ``time.perf_counter()``.

        Raises
        ------
        FrameFormatError
            Bad magic, version, structure, or empty line.
        FrameCrcError
            CRC mismatch.
        FrameValueError
            Unknown channel, unparsable value, or unknown mode code.
        """
        if isinstance(line, (bytearray, memoryview)):
            line = bytes(line)
        if not isinstance(line, bytes):
            raise FrameFormatError("line must be bytes")
        if not line:
            raise FrameFormatError("empty line")
        # Strip a trailing CR.
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            raise FrameFormatError("empty line after CR strip")

        # ---- CRC trailer (split body off, then verify after structural checks) ----
        if crc == "ccitt":
            star = line.rfind(b"*")
            if star <= 0 or star + 5 != len(line):
                raise FrameFormatError("missing or malformed CRC trailer")
            crc_field = line[star + 1:].decode("ascii", errors="replace")
            raw_body = line[:star]
            if not crc_field or len(crc_field) != 4:
                raise FrameFormatError("CRC must be 4 hex digits")
            try:
                expected = int(crc_field, 16)
            except ValueError as exc:
                raise FrameFormatError(f"invalid CRC hex: {crc_field!r}") from exc
        elif crc == "none":
            raw_body = line
            expected = None
        else:
            raise ValueError(f"unknown crc policy: {crc!r}")

        # ---- Body parsing (structural checks first) ----
        body = raw_body
        if not body.startswith(FrameCodec.HEADER):
            raise FrameFormatError(
                f"bad magic/version: expected {FrameCodec.HEADER!r} prefix"
            )
        body = body[len(FrameCodec.HEADER):]
        if not body:
            raise FrameFormatError("empty body after magic")
        fields = body.split(b",")
        if len(fields) < 3:
            raise FrameFormatError(
                f"need at least 3 header fields (seq, time_s, status), got {len(fields)}"
            )

        try:
            sequence = int(fields[0])
        except ValueError as exc:
            raise FrameFormatError(f"non-integer sequence: {fields[0]!r}") from exc
        try:
            time_s = float(fields[1])
        except ValueError as exc:
            raise FrameFormatError(f"non-numeric time_s: {fields[1]!r}") from exc
        try:
            status = FrameStatus(fields[2].decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise FrameFormatError(f"unknown status: {fields[2]!r}") from exc

        readings: Dict[str, SensorReading] = {}
        allow: Optional[Set[str]] = (
            SENSOR_NAME_SET if allowed_channels is None else set(allowed_channels)
        )
        for token in fields[3:]:
            if not token:
                continue
            try:
                name_b, rest = token.split(b"=", 1)
            except ValueError as exc:
                raise FrameFormatError(f"missing '=' in channel token {token!r}") from exc
            name = name_b.decode("ascii")
            if name not in allow:
                raise FrameValueError(f"unknown channel: {name!r}")
            try:
                value_b, mode_b = rest.split(b":", 1)
            except ValueError as exc:
                raise FrameFormatError(
                    f"missing ':' in channel token {token!r}"
                ) from exc
            value_str = value_b.decode("ascii")
            if value_str == "":
                value: Optional[float] = None
            else:
                try:
                    value = float(value_str)
                except ValueError as exc:
                    raise FrameValueError(
                        f"unparsable value for channel {name!r}: {value_str!r}"
                    ) from exc
            mode_code = mode_b.decode("ascii", errors="replace")
            if mode_code not in NOISE_MODE_FROM_CODE:
                raise FrameValueError(f"unknown mode code: {mode_code!r}")
            mode = NOISE_MODE_FROM_CODE[mode_code]
            # If value is None we mark the mode as DROPPED regardless
            # of what the wire said — a "stuck" with value None is
            # contradictory.
            if value is None and mode is not NoiseMode.DROPPED:
                mode = NoiseMode.DROPPED
            readings[name] = SensorReading(value=value, mode=mode)

        # ---- CRC verification last (so structural problems surface first) ----
        if expected is not None:
            actual = int(crc16_hex(raw_body), 16)
            if actual != expected:
                raise FrameCrcError(
                    f"CRC mismatch: expected {expected:04x}, got {actual:04x}"
                )

        sample = SensorSample(time_s=time_s, readings=readings)
        return TelemetryFrame(
            sequence=sequence,
            time_s=time_s,
            produced_wall_time=(
                now_wall_time if now_wall_time is not None else _time.perf_counter()
            ),
            sample=sample,
            status=status,
        )


__all__ = [
    "FrameCodec",
    "FrameFormatError",
    "FrameCrcError",
    "FrameValueError",
]
