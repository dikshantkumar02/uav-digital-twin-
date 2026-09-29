"""
Hardware interface package (PHASE 14).

A serial / UART transport that ingests real engine telemetry frames,
decodes them with the :class:`FrameCodec`, and pushes the resulting
:class:`~backend.telemetry.TelemetryFrame` objects onto a
:class:`~backend.telemetry.TelemetryQueue`. The downstream
pipeline (``Preprocessor`` → digital twin → …) is unchanged.

Public surface::

    from backend.hardware import (
        # Ports
        Port, SerialPort, LoopbackPort, LoopbackPair, NullPort,
        # Codec
        FrameCodec, FrameFormatError, FrameCrcError, FrameValueError,
        # Transport
        SerialSource, TransportStats,
        # Wire helpers
        NOISE_MODE_CODE, NOISE_MODE_FROM_CODE,
    )
    from backend.hardware.config import HardwareConfig
"""

from .crc import crc16_ccitt, crc16_hex
from .framing import LineFramer
from .ports import LoopbackPair, LoopbackPort, NullPort, Port, SerialPort
from .protocol import (
    FrameCodec,
    FrameCrcError,
    FrameFormatError,
    FrameValueError,
)
from .transport import SerialSource, TransportStats
from .wire import NOISE_MODE_CODE, NOISE_MODE_FROM_CODE, SENSOR_NAME_SET


__all__ = [
    # Ports
    "Port",
    "SerialPort",
    "LoopbackPort",
    "LoopbackPair",
    "NullPort",
    # Codec
    "FrameCodec",
    "FrameFormatError",
    "FrameCrcError",
    "FrameValueError",
    "MAGIC",
    "VERSION",
    # Transport
    "SerialSource",
    "TransportStats",
    # Framing
    "LineFramer",
    # CRC
    "crc16_ccitt",
    "crc16_hex",
    # Wire helpers
    "NOISE_MODE_CODE",
    "NOISE_MODE_FROM_CODE",
    "SENSOR_NAME_SET",
]
