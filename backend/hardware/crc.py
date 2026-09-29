"""
CRC-16-CCITT (XMODEM variant).

Used by the PHASE 14 wire protocol to validate each frame before
the transport pushes it onto the telemetry queue. Polynomial 0x1021,
init 0xFFFF, no reflection, no xorout — the same parameters used by
NMEA 0183 and MAVLink v1.

The implementation is pure Python (no external deps) and is fast
enough for the line-rate we care about (~1 µs per KB).
"""

from __future__ import annotations

# CRC-16-CCITT/XMODEM parameters.
_POLYNOMIAL = 0x1021
_INIT = 0xFFFF


def crc16_ccitt(data: bytes, init: int = _INIT) -> int:
    """Compute CRC-16-CCITT/XMODEM over ``data``.

    Parameters
    ----------
    data:
        Bytes to checksum.
    init:
        Initial CRC value. Default 0xFFFF (XMODEM). The MALE UAV
        hardware spec does not pin a different init, so 0xFFFF is
        used everywhere.

    Returns
    -------
    int
        16-bit CRC value, in the range 0..0xFFFF.
    """
    crc = int(init) & 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _POLYNOMIAL) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc16_hex(data: bytes, init: int = _INIT) -> str:
    """Compute the CRC and return it as a 4-character lowercase hex string.

    Suitable for the ``*XXXX`` trailer on each wire frame.
    """
    return f"{crc16_ccitt(data, init=init):04x}"


__all__ = ["crc16_ccitt", "crc16_hex"]
