"""PHASE 14 tests — hardware interface (serial / UART transport)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from backend.config import (
    HardwareConfig,
    load_config,
    load_hardware_config,
)
from backend.hardware import (
    FrameCodec,
    FrameCrcError,
    FrameFormatError,
    FrameValueError,
    LineFramer,
    LoopbackPair,
    LoopbackPort,
    NullPort,
    SerialSource,
    TransportStats,
    crc16_ccitt,
    crc16_hex,
)
from backend.hardware.config import HardwareConfig as HardwareConfigLocal
from backend.sensors import NoiseMode, SensorReading, SensorSample
from backend.telemetry import (
    FrameStatus,
    LatencyTracker,
    TelemetryFrame,
    TelemetryQueue,
)
from backend.telemetry import StreamSource

pytestmark = pytest.mark.phase14

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_frame(
    sequence: int,
    time_s: float,
    *,
    readings: dict[str, SensorReading] | None = None,
    status: FrameStatus = FrameStatus.OK,
) -> TelemetryFrame:
    """Build a TelemetryFrame with a real sample and clock."""
    if readings is None:
        readings = {
            "rpm": SensorReading(value=2350.0 + sequence, mode=NoiseMode.NORMAL),
            "egt": SensorReading(value=690.0, mode=NoiseMode.NORMAL),
            "cht": SensorReading(value=198.0, mode=NoiseMode.NORMAL),
            "oil_pressure": SensorReading(value=58.0, mode=NoiseMode.NORMAL),
            "oil_temperature": SensorReading(value=85.0, mode=NoiseMode.NORMAL),
            "fuel_flow": SensorReading(value=28.0, mode=NoiseMode.NORMAL),
            "vibration": SensorReading(value=0.4, mode=NoiseMode.NORMAL),
            "imu_accel": SensorReading(value=0.1, mode=NoiseMode.NORMAL),
            "altitude": SensorReading(value=1500.0, mode=NoiseMode.NORMAL),
            "airspeed": SensorReading(value=60.0, mode=NoiseMode.NORMAL),
            "ambient_temperature": SensorReading(value=15.0, mode=NoiseMode.NORMAL),
            "ambient_pressure": SensorReading(value=101325.0, mode=NoiseMode.NORMAL),
        }
    sample = SensorSample(time_s=time_s, readings=readings)
    return TelemetryFrame(
        sequence=sequence,
        time_s=time_s,
        produced_wall_time=time.time(),
        sample=sample,
        status=status,
    )


# ---------------------------------------------------------------------
# TestCrc
# ---------------------------------------------------------------------
class TestCrc:
    def test_crc16_ccitt_empty(self) -> None:
        # Init value is the empty-data CRC by convention.
        assert crc16_ccitt(b"") == 0xFFFF

    def test_crc16_ccitt_known_vector(self) -> None:
        # Well-known XMODEM-CRC test vector: "123456789" -> 0x29B1
        # (init=0xFFFF, poly=0x1021, no reflect, no xorout).
        assert crc16_ccitt(b"123456789") == 0x29B1
        assert crc16_hex(b"123456789") == "29b1"

    def test_crc16_ccitt_detects_single_bit_flip(self) -> None:
        payload = b"this is a test payload of moderate length"
        good = crc16_ccitt(payload)
        # Flip one bit in the middle.
        bad_bytes = bytearray(payload)
        bad_bytes[10] ^= 0x01
        assert crc16_ccitt(bytes(bad_bytes)) != good


# ---------------------------------------------------------------------
# TestLineFramer
# ---------------------------------------------------------------------
class TestLineFramer:
    def test_split_single_line(self) -> None:
        fr = LineFramer()
        out = fr.feed(b"hello\n")
        assert out == [b"hello"]
        assert fr.buffered_bytes == 0

    def test_split_partial_then_complete(self) -> None:
        fr = LineFramer()
        assert fr.feed(b"hel") == []
        assert fr.buffered_bytes == 3
        assert fr.feed(b"lo\n") == [b"hello"]
        assert fr.buffered_bytes == 0

    def test_split_handles_crlf_and_overflow(self) -> None:
        fr = LineFramer(max_line_bytes=128)
        # CRLF line
        assert fr.feed(b"a\r\nb") == [b"a"]
        # A line that exceeds the limit gets counted as overflow.
        huge = b"x" * 200 + b"\n"
        assert fr.feed(huge) == []
        assert fr.overflow_lines == 1


# ---------------------------------------------------------------------
# TestFrameCodec
# ---------------------------------------------------------------------
class TestFrameCodec:
    def test_encode_decode_round_trip(self) -> None:
        frame = _make_frame(sequence=42, time_s=4.2)
        encoded = FrameCodec.encode(frame)
        assert encoded.endswith(b"\n")
        # Strip CRC trailer for sanity
        assert b"*29b1" in encoded or b"*" in encoded
        decoded = FrameCodec.decode(encoded.rstrip(b"\n"))
        assert decoded.sequence == frame.sequence
        assert decoded.time_s == pytest.approx(frame.time_s)
        assert decoded.status == frame.status
        # Read keys in stable order.
        for name, reading in frame.sample.readings.items():
            d_reading = decoded.sample.readings[name]
            assert d_reading.value == pytest.approx(reading.value)
            assert d_reading.mode == reading.mode

    def test_encode_dropout_channel(self) -> None:
        readings = {
            "rpm": SensorReading(value=None, mode=NoiseMode.DROPPED),
            "egt": SensorReading(value=690.0, mode=NoiseMode.NORMAL),
        }
        frame = _make_frame(sequence=7, time_s=0.7, readings=readings)
        encoded = FrameCodec.encode(frame)
        decoded = FrameCodec.decode(encoded.rstrip(b"\n"))
        assert decoded.sample.readings["rpm"].value is None
        # Decoder normalises "value=None + non-DROPPED mode" to DROPPED.
        assert decoded.sample.readings["rpm"].mode == NoiseMode.DROPPED
        assert decoded.sample.readings["egt"].value == pytest.approx(690.0)

    def test_decode_bad_magic_raises_format_error(self) -> None:
        with pytest.raises(FrameFormatError):
            FrameCodec.decode(b"FOO,1,0,0.0,OK,*0000")

    def test_decode_bad_crc_raises_crc_error(self) -> None:
        frame = _make_frame(sequence=1, time_s=0.1)
        encoded = FrameCodec.encode(frame).rstrip(b"\n")
        # Flip a digit inside one of the channel values (a non-structural
        # byte that, if mutated, still produces a parseable number).
        # `rpm` is the last channel in the sorted output, so its value
        # sits near the end of the body, well clear of the time_s field
        # and channel separators.
        body = encoded[: encoded.rfind(b"*")]
        rpm_pos = body.find(b"rpm=")
        assert rpm_pos > 0
        # Flip the first digit of the rpm value.
        target = rpm_pos + len(b"rpm=") + 1  # '2' in '2351.0...'
        mutated = (
            body[:target]
            + bytes([body[target] ^ 0x01])
            + body[target + 1 :]
            + encoded[encoded.rfind(b"*") :]
        )
        with pytest.raises(FrameCrcError):
            FrameCodec.decode(mutated)

    def test_decode_unknown_channel_raises_value_error(self) -> None:
        line = b"AERO,1,0,0.0,OK,nosuch=1.0:N,*0000"
        with pytest.raises(FrameValueError):
            FrameCodec.decode(line)

    def test_encode_with_crc_none_skips_trailer(self) -> None:
        frame = _make_frame(sequence=1, time_s=0.1)
        encoded = FrameCodec.encode(frame, crc="none")
        assert encoded.endswith(b"\n")
        assert b"*" not in encoded
        decoded = FrameCodec.decode(encoded.rstrip(b"\n"), crc="none")
        assert decoded.sequence == 1


# ---------------------------------------------------------------------
# TestSerialSource
# ---------------------------------------------------------------------
class TestSerialSource:
    def test_pushes_decoded_frames_to_queue(self) -> None:
        pair = LoopbackPair()
        q = TelemetryQueue(max_size=64)
        src = SerialSource(pair.port_a(), q)
        for i in range(10):
            pair.port_b().write(FrameCodec.encode(_make_frame(i, i * 0.1)))
        # Drive the reader manually via pump_once until everything drains.
        for _ in range(20):
            if len(q) >= 10:
                break
            src.pump_once(timeout_s=0.01)
        assert len(q) == 10
        assert src.stats.frames_pushed == 10
        assert src.stats.format_errors == 0
        assert src.stats.crc_errors == 0

    def test_handles_malformed_lines_gracefully(self) -> None:
        pair = LoopbackPair()
        q = TelemetryQueue(max_size=64)
        src = SerialSource(pair.port_a(), q)
        # 5 valid + 1 garbage + 1 unknown channel + 1 short line.
        for i in range(5):
            pair.port_b().write(FrameCodec.encode(_make_frame(i, i * 0.1)))
        pair.port_b().write(b"junk\n")
        pair.port_b().write(b"AERO,1,99,1.0,OK,nosuch=1.0:N,*0000\n")
        pair.port_b().write(b"\n")  # empty line after CR strip
        # Pump until both the queue is full AND the error counters
        # show the bad lines were processed.
        for _ in range(40):
            if len(q) >= 5 and (
                src.stats.format_errors + src.stats.value_errors >= 2
            ):
                break
            src.pump_once(timeout_s=0.01)
        assert len(q) == 5
        assert src.stats.frames_pushed == 5
        # At least one format error (junk / empty), at least one value error.
        assert src.stats.format_errors + src.stats.value_errors >= 2

    def test_handles_partial_lines_until_complete(self) -> None:
        pair = LoopbackPair()
        q = TelemetryQueue(max_size=8)
        src = SerialSource(pair.port_a(), q)
        frame = _make_frame(1, 0.1)
        encoded = FrameCodec.encode(frame)
        # Split the encoded line in the middle of the body.
        cut = len(encoded) // 2
        pair.port_b().write(encoded[:cut])
        for _ in range(5):
            if src.stats.frames_pushed >= 1:
                break
            assert src.pump_once(timeout_s=0.01) == 0
        # Send the rest — reader must reassemble.
        pair.port_b().write(encoded[cut:])
        for _ in range(5):
            if src.stats.frames_pushed >= 1:
                break
            src.pump_once(timeout_s=0.01)
        assert src.stats.frames_pushed == 1
        assert len(q) == 1

    def test_stop_drains_thread(self) -> None:
        pair = LoopbackPair()
        q = TelemetryQueue(max_size=16)
        src = SerialSource(pair.port_a(), q, poll_sleep_s=0.001)
        src.start()
        assert src.is_running()
        for i in range(3):
            pair.port_b().write(FrameCodec.encode(_make_frame(i, i * 0.1)))
        time.sleep(0.05)
        src.stop(timeout_s=2.0)
        assert not src.is_running()
        # Calling stop again is a no-op.
        src.stop(timeout_s=0.5)

    def test_latency_tracker_records_transport_stage(self) -> None:
        pair = LoopbackPair()
        q = TelemetryQueue(max_size=16)
        tracker = LatencyTracker()
        src = SerialSource(pair.port_b(), q, latency_tracker=tracker)  # read on B side
        for i in range(10):
            pair.port_a().write(FrameCodec.encode(_make_frame(i, i * 0.1)))
        for _ in range(20):
            if len(q) >= 10:
                break
            src.pump_once(timeout_s=0.01)
        assert src.stats.frames_pushed == 10
        assert "transport" in tracker.stages
        assert tracker.stages["transport"].samples >= 10


# ---------------------------------------------------------------------
# TestPorts
# ---------------------------------------------------------------------
class TestPorts:
    def test_loopback_pair_round_trip(self) -> None:
        pair = LoopbackPair()
        pair.port_a().write(b"hello")
        # Read from the *other* side.
        out = pair.port_b().read(n=5, timeout_s=0.1)
        assert out == b"hello"

    def test_null_port_is_no_op(self) -> None:
        p = NullPort()
        assert p.read(10, 0.0) == b""
        assert p.write(b"x") == 0
        assert p.is_open is True


# ---------------------------------------------------------------------
# TestHardwareConfig
# ---------------------------------------------------------------------
class TestHardwareConfig:
    def test_defaults(self) -> None:
        cfg = HardwareConfig.defaults()
        assert cfg.port == "loopback"
        assert cfg.baudrate == 115200
        assert cfg.bytesize == 8
        assert cfg.parity == "N"
        assert cfg.stopbits == 1.0
        assert cfg.read_timeout_s == 0.5
        assert cfg.read_chunk_bytes == 4096
        assert cfg.crc == "ccitt"
        assert cfg.reconnect_on_error is True
        assert cfg.reconnect_backoff_s == 1.0
        assert cfg.expected_channels is None

    def test_load_hardware_config_missing_file_returns_defaults(self, tmp_path) -> None:
        cfg = load_hardware_config(tmp_path)
        assert cfg == HardwareConfig.defaults()

    def test_load_hardware_config_validates_yaml(self, tmp_path) -> None:
        path = tmp_path / "hardware.yaml"
        path.write_text(
            "hardware:\n  port: /dev/ttyUSB0\n  baudrate: 9600\n  crc: none\n",
            encoding="utf-8",
        )
        cfg = load_hardware_config(tmp_path)
        assert cfg.port == "/dev/ttyUSB0"
        assert cfg.baudrate == 9600
        assert cfg.crc == "none"

    def test_parity_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            HardwareConfig(parity="X")

    def test_crc_must_be_valid(self) -> None:
        with pytest.raises(ValueError):
            HardwareConfig(crc="md5")

    def test_hardware_config_in_loaded_config(self) -> None:
        cfg = load_config()
        assert isinstance(cfg.hardware, HardwareConfig)
        # Default port is "loopback" (matches default).
        assert cfg.hardware.port == "loopback"


# ---------------------------------------------------------------------
# TestStreamSourceParity
# ---------------------------------------------------------------------
class TestStreamSourceParity:
    """The PHASE 5 simulated source and the PHASE 14 serial source must
    both feed the same TelemetryQueue. This test confirms the two
    paths remain compatible: the queue accepts frames from each."""

    def test_synthetic_and_serial_share_queue_shape(self) -> None:
        from backend.sensors import SensorBundle
        from backend.simulation import EngineSimulator

        cfg = load_config()
        simulator = EngineSimulator(cfg.engine, cfg.environment, dt_s=0.1)
        bundle = SensorBundle(cfg.sensors, simulator, master_seed=1)

        q = TelemetryQueue(max_size=64)
        synth = StreamSource(bundle, q)
        synth.run(max_steps=5)
        synthetic_frames = list(q.drain())
        assert len(synthetic_frames) == 5
        # All synthetic frames are OK.
        assert all(f.status == FrameStatus.OK for f in synthetic_frames)

        # Now feed a serial frame through the same queue.
        pair = LoopbackPair()
        src = SerialSource(pair.port_a(), q)
        for i in range(3):
            pair.port_b().write(FrameCodec.encode(_make_frame(100 + i, i * 0.1)))
        for _ in range(10):
            if len(q) >= 3:
                break
            src.pump_once(timeout_s=0.01)
        assert len(q) == 3
        serial_frames = list(q.drain())
        assert all(f.sequence >= 100 for f in serial_frames)
