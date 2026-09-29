"""
WireRecorder — captures a stream of :class:`DashboardSnapshot` as
AERO wire bytes and a parallel JSONL file.

The recorder owns:

* a :class:`~backend.hardware.FrameCodec` (PHASE 14) that turns a
  synthetic :class:`~backend.telemetry.TelemetryFrame` (one per
  snapshot) into wire bytes;
* a :class:`~backend.hil.capture.SnapshotCapture` that turns the
  same snapshot into a JSONL line;
* a :class:`Manifest` (filled in on close).

Replay (see :class:`~backend.hil.replayer.WireReplayer`) consumes
the resulting ``wire.bin`` through a :class:`LoopbackPort` and feeds
the same bytes back into the :class:`SerialSource` transport path.
That makes the recorded run a byte-perfect substitute for live
hardware.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Iterable

from backend.dashboard.snapshot import DashboardSnapshot
from backend.environment import EnvironmentState
from backend.hardware import FrameCodec
from backend.sensors import (
    NoiseMode,
    SENSOR_CHANNELS,
    SensorReading,
    SensorSample,
)
from backend.simulation import EngineState
from backend.telemetry import TelemetryFrame

from .capture import SnapshotCapture
from .manifest import Manifest


# ---------------------------------------------------------------------
# Channel -> (engine attr, env attr) extraction
# ---------------------------------------------------------------------
# Maps the 12 SENSOR_CHANNELS to the field on EngineState /
# EnvironmentState where the corresponding "true" value lives. Kept
# in sync with backend.sensors.channels._true_value.
def _channel_value(name: str, engine: EngineState, env: EnvironmentState):
    if name == "rpm":
        return engine.rpm
    if name == "egt":
        return engine.egt_c
    if name == "cht":
        return engine.cht_c
    if name == "oil_pressure":
        return engine.oil_pressure_psi
    if name == "oil_temperature":
        return engine.oil_temperature_c
    if name == "fuel_flow":
        return engine.fuel_flow_lph
    if name == "vibration":
        return engine.vibration_rms_g
    if name == "imu_accel":
        return env.vertical_accel_mps2
    if name == "altitude":
        return env.altitude_m
    if name == "airspeed":
        return env.airspeed_mps
    if name == "ambient_temperature":
        return env.atmosphere.temperature_c
    if name == "ambient_pressure":
        return env.atmosphere.pressure_pa
    raise KeyError(f"unknown sensor channel: {name}")


def _snapshot_to_telemetry_frame(snapshot: DashboardSnapshot) -> TelemetryFrame:
    """Build a :class:`TelemetryFrame` from a :class:`DashboardSnapshot`.

    Each channel gets a ``SensorReading(value=..., mode=NORMAL)``.
    The frame's sequence / time / status mirror the snapshot.
    """
    readings = {
        name: SensorReading(
            value=_channel_value(name, snapshot.engine_state, snapshot.environment),
            mode=NoiseMode.NORMAL,
        )
        for name in SENSOR_CHANNELS
    }
    sample = SensorSample(
        time_s=snapshot.time_s,
        readings=readings,
        engine=snapshot.engine_state,
        env=snapshot.environment,
    )
    # The PHASE 5 TelemetryFrame accepts a FrameStatus string. We
    # import locally to keep the recorder's surface small.
    from backend.telemetry import FrameStatus

    try:
        status = FrameStatus(snapshot.frame_status)
    except ValueError:
        status = FrameStatus.OK
    return TelemetryFrame(
        sequence=int(snapshot.tick_index),
        time_s=float(snapshot.time_s),
        produced_wall_time=_time.perf_counter(),
        sample=sample,
        status=status,
    )


# ---------------------------------------------------------------------
# WireRecorder
# ---------------------------------------------------------------------
class WireRecorder:
    """Records a sequence of :class:`DashboardSnapshot` as AERO wire bytes."""

    def __init__(
        self,
        out_dir: Path,
        manifest: Manifest,
        *,
        crc: str = "ccitt",
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = manifest
        self._crc = str(crc)
        self._bin_path = self._out_dir / "wire.bin"
        self._jsonl_path = self._out_dir / "snapshots.jsonl"
        self._bin_fh = self._bin_path.open("wb")
        self._capture = SnapshotCapture(self._jsonl_path)
        self._n = 0

    # ------------------------------------------------------------------
    @property
    def bin_path(self) -> Path:
        return self._bin_path

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def frames_recorded(self) -> int:
        return self._n

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # ------------------------------------------------------------------
    def record(self, snapshot: DashboardSnapshot) -> int:
        """Append one snapshot's wire bytes and JSONL line. Returns sequence."""
        frame = _snapshot_to_telemetry_frame(snapshot)
        encoded = FrameCodec.encode(frame, crc=self._crc)
        self._bin_fh.write(encoded)
        self._bin_fh.flush()
        self._capture.capture(snapshot)
        self._n += 1
        return frame.sequence

    def record_iter(self, snapshots: Iterable[DashboardSnapshot]) -> int:
        """Record an iterable of snapshots. Returns the count recorded."""
        n = 0
        for snap in snapshots:
            self.record(snap)
            n += 1
        return n

    # ------------------------------------------------------------------
    def close(self) -> Manifest:
        """Flush + close, return the final manifest (with n_frames filled)."""
        self._capture.close()
        if not self._bin_fh.closed:
            self._bin_fh.flush()
            self._bin_fh.close()
        final = self._manifest.with_frames(self._n)
        Manifest.write_yaml(self._out_dir / "manifest.yaml", final)
        return final


__all__ = ["WireRecorder", "_snapshot_to_telemetry_frame"]
