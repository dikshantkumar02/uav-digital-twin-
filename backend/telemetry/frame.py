"""
Telemetry frame — a single time-stamped packet on the bus.

A frame is created by the stream source (one per sensor tick) and is
the unit of work the rest of the system consumes.

PHASE 17 (streaming pipeline) extends the frame with the five fields
the user spec mandates for every bus message:

* ``mission_id`` — which mission / scenario this frame belongs to
* ``vehicle_id`` — which UAV (MALE UAV) the engine is mounted on
* ``engine_id`` — which engine on the vehicle
* ``data_quality`` — numeric 0..1 score, set by
  :class:`~backend.telemetry.adapter.TelemetryAdapter`
* ``model_version`` — semver string identifying the running model
  stack ("phase17-streaming-1.0.0" by default)

All five default so existing PHASE 1–16 callers (incl. the HIL
WireRecorder / WireReplayer) keep working unmodified.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from backend.sensors import SensorSample


class FrameStatus(str, Enum):
    """Validation status of a telemetry frame after preprocessing."""

    OK = "OK"                 # every channel produced a recent value
    STALE = "STALE"           # some channels have a recent-enough fallback value
    INVALID = "INVALID"       # too many channels missing — frame must not drive AI
    DROPPED = "DROPPED"       # produced but dropped at the queue (overflow)


@dataclass
class TelemetryFrame:
    """One packet on the telemetry bus."""

    sequence: int
    time_s: float
    produced_wall_time: float        # wall-clock at frame creation
    sample: SensorSample
    status: FrameStatus = FrameStatus.OK
    preprocessed_wall_time: Optional[float] = None
    notes: Dict[str, str] = field(default_factory=dict)
    # --- PHASE 17 (streaming pipeline) ---
    mission_id: str = "default-mission"
    vehicle_id: str = "default-vehicle"
    engine_id: str = "default-engine"
    data_quality: float = 1.0
    model_version: str = "phase17-streaming-1.0.0"

    def to_dict(self) -> dict:
        d: dict = {
            "sequence": self.sequence,
            "time_s": self.time_s,
            "produced_wall_time": self.produced_wall_time,
            "preprocessed_wall_time": self.preprocessed_wall_time,
            "status": self.status.value,
            "mission_id": self.mission_id,
            "vehicle_id": self.vehicle_id,
            "engine_id": self.engine_id,
            "data_quality": float(self.data_quality),
            "model_version": self.model_version,
        }
        for k, v in self.notes.items():
            d[f"note.{k}"] = v
        if self.sample is not None:
            d.update(self.sample.to_dict())
        return d
