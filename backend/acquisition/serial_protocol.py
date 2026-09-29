"""
MCU serial protocol (PHASE 18 — embedded telemetry acquisition).

The serial protocol is what the user's firmware speaks over the
UART to the edge computer. The layout is binary — bandwidth on
an MAVLink-style wire is precious and a per-line CR/LF framing
would dominate at 115200 baud — so the frame is fixed-format::

    [STX=0xAA] [LEN] [CMD] [SEQ_LO] [SEQ_HI] [PAYLOAD...] [CRC_LO] [CRC_HI] [ETX=0x55]

* ``LEN`` is the payload length in bytes (0–255).
* ``CMD`` is one of the keys in :data:`COMMANDS`.
* ``SEQ`` is a 16-bit little-endian sequence number.
* ``PAYLOAD`` is the command-specific bytes.
* ``CRC16`` is CRC-16-CCITT/XMODEM (re-using the PHASE 14
  primitive) over the bytes from ``LEN`` through the last
  payload byte.
* ``STX = 0xAA``, ``ETX = 0x55``.

This module is a pure encoder/decoder — no I/O. The transport
that reads bytes off the wire is in :mod:`backend.hardware`.
The :class:`SensorAbstraction` is responsible for converting
:class:`TelemetrySample` objects into :class:`McuFrame`
payloads (for ``SAMPLE``) and back.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict

from backend.hardware.crc import crc16_ccitt


# ---------------------------------------------------------------------
# Command set
# ---------------------------------------------------------------------
COMMANDS: Dict[int, str] = {
    0x01: "HELLO",        # MCU → host, on boot
    0x02: "HEARTBEAT",    # MCU → host, every ~1 s
    0x03: "SAMPLE",       # MCU → host, periodic
    0x04: "ERROR",        # MCU → host, on fault
    0x10: "CALIBRATE",    # host → MCU
    0x11: "SET_RATE",     # host → MCU
    0x12: "RESET",        # host → MCU
}
COMMAND_NAMES: Dict[str, int] = {v: k for k, v in COMMANDS.items()}


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
STX = 0xAA
ETX = 0x55
MAX_PAYLOAD = 255
HEADER_LEN = 5       # LEN + CMD + SEQ_LO + SEQ_HI
TRAILER_LEN = 3      # CRC_LO + CRC_HI + ETX


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------
class McuFormatError(ValueError):
    """Bad STX/ETX, length, or structure."""


class McuCrcError(ValueError):
    """CRC mismatch — the frame was corrupted in transit."""


# ---------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class McuFrame:
    """A binary MCU frame.

    ``payload`` is a ``bytes`` object up to 255 bytes. ``command``
    is the symbolic name looked up from :data:`COMMANDS`; the
    underlying byte is preserved in :attr:`command_id` so the
    decoder can round-trip unknown commands (forward
    compatibility).
    """

    command_id: int
    sequence: int
    payload: bytes

    def __post_init__(self) -> None:
        if not (0 <= int(self.command_id) <= 0xFF):
            raise ValueError(
                f"command_id must be a single byte, got {self.command_id}"
            )
        if not (0 <= int(self.sequence) <= 0xFFFF):
            raise ValueError(
                f"sequence must fit in 16 bits, got {self.sequence}"
            )
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError(
                f"payload too long: {len(self.payload)} > {MAX_PAYLOAD}"
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def command(self) -> str:
        """Symbolic command name, or ``"UNKNOWN"`` if not in the table."""
        return COMMANDS.get(int(self.command_id), "UNKNOWN")

    @property
    def length(self) -> int:
        return len(self.payload)

    # ------------------------------------------------------------------
    # CRC
    # ------------------------------------------------------------------
    def _crc_bytes(self) -> bytes:
        """Bytes the CRC is computed over: LEN..last payload byte."""
        return (
            bytes([self.length, int(self.command_id) & 0xFF])
            + struct.pack("<H", int(self.sequence) & 0xFFFF)
            + bytes(self.payload)
        )

    @property
    def crc_ok(self) -> bool:
        """True if the round-tripped frame's CRC matches the embedded CRC."""
        # Re-encode and check; the property is convenience for
        # round-trip verification by tests.
        raw = self.encode()
        expected = raw[len(raw) - 3 : len(raw) - 1]
        actual = self.crc_bytes()
        return int.from_bytes(expected, "little") == crc16_ccitt(actual)

    def crc_value(self) -> int:
        return crc16_ccitt(self._crc_bytes())

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self) -> bytes:
        """Serialize to a ``bytes`` object ready to write to the UART."""
        crc = self.crc_value()
        return (
            bytes([STX, self.length, int(self.command_id) & 0xFF])
            + struct.pack("<H", int(self.sequence) & 0xFFFF)
            + bytes(self.payload)
            + struct.pack("<H", crc)
            + bytes([ETX])
        )

    @classmethod
    def decode(cls, raw: bytes) -> "McuFrame":
        """Parse ``raw`` into a :class:`McuFrame`.

        Raises
        ------
        McuFormatError
            Bad STX, ETX, length, or structure.
        McuCrcError
            CRC mismatch (frame corrupted in transit).
        """
        if not isinstance(raw, (bytes, bytearray)):
            raise McuFormatError("raw must be bytes")
        raw = bytes(raw)
        if len(raw) < HEADER_LEN + TRAILER_LEN:
            raise McuFormatError(
                f"frame too short: {len(raw)} bytes, "
                f"need at least {HEADER_LEN + TRAILER_LEN}"
            )
        if raw[0] != STX:
            raise McuFormatError(
                f"bad STX: expected 0x{STX:02x}, got 0x{raw[0]:02x}"
            )
        if raw[-1] != ETX:
            raise McuFormatError(
                f"bad ETX: expected 0x{ETX:02x}, got 0x{raw[-1]:02x}"
            )
        length = raw[1]
        if len(raw) != HEADER_LEN + length + TRAILER_LEN:
            raise McuFormatError(
                f"length mismatch: declared {length} payload, "
                f"frame is {len(raw)} bytes"
            )
        if length > MAX_PAYLOAD:
            raise McuFormatError(
                f"declared payload too long: {length} > {MAX_PAYLOAD}"
            )
        command_id = raw[2]
        (sequence,) = struct.unpack("<H", raw[3:5])
        payload = bytes(raw[5 : 5 + length])
        crc_lo, crc_hi = raw[5 + length], raw[5 + length + 1]
        expected = (crc_hi << 8) | crc_lo

        body = bytes(raw[1 : 5 + length])  # LEN..last payload byte
        actual = crc16_ccitt(body)
        if actual != expected:
            raise McuCrcError(
                f"CRC mismatch: expected 0x{expected:04x}, got 0x{actual:04x}"
            )
        return cls(command_id=int(command_id), sequence=int(sequence), payload=payload)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------
def make_hello_frame(sequence: int = 0, firmware_version: str = "1.0.0") -> McuFrame:
    """Build a HELLO frame announcing the firmware version."""
    payload = firmware_version.encode("ascii")[:MAX_PAYLOAD]
    return McuFrame(command_id=COMMAND_NAMES["HELLO"], sequence=sequence, payload=payload)


def make_heartbeat_frame(sequence: int, uptime_ms: int) -> McuFrame:
    """Build a HEARTBEAT frame carrying uptime in ms (uint32 LE)."""
    payload = struct.pack("<I", int(uptime_ms) & 0xFFFFFFFF)
    return McuFrame(command_id=COMMAND_NAMES["HEARTBEAT"], sequence=sequence, payload=payload)


def make_error_frame(sequence: int, code: int, message: str = "") -> McuFrame:
    """Build an ERROR frame with a uint8 code and optional ASCII message."""
    msg_bytes = message.encode("ascii")[: MAX_PAYLOAD - 1]
    payload = bytes([int(code) & 0xFF]) + msg_bytes
    return McuFrame(command_id=COMMAND_NAMES["ERROR"], sequence=sequence, payload=payload)


__all__ = [
    "COMMANDS",
    "COMMAND_NAMES",
    "STX",
    "ETX",
    "McuFormatError",
    "McuCrcError",
    "McuFrame",
    "make_hello_frame",
    "make_heartbeat_frame",
    "make_error_frame",
]
