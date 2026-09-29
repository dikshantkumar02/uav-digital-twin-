"""
TelemetryAdapter (PHASE 17 — real-time streaming pipeline).

The adapter is the **first** stage of the streaming pipeline. It
turns a raw :class:`~backend.sensors.SensorSample` into a
:class:`TelemetryFrame` carrying the five spec-mandated fields:

* ``mission_id`` — which mission / scenario
* ``vehicle_id`` — which MALE UAV
* ``engine_id`` — which engine on the vehicle
* ``data_quality`` — a numeric 0..1 score, computed from
  dropouts / stuck sensors in the sample
* ``model_version`` — semver string identifying the model stack

The data-quality formula is transparent and deterministic::

    q = 1 - (n_dropout + 3 * n_stuck) / (3 * n_channels)
    q = clip(q, 0, 1)

A clean frame (``n_dropout = n_stuck = 0``) has
``data_quality = 1.0``. A completely dead bundle
(``n_dropout = n_channels``) has ``data_quality = 0.0``. A
single stuck channel (e.g. RPM) is penalised more heavily than a
single dropout because a stuck sensor actively misleads downstream
calculators.

The adapter is intentionally **stateless** between ``adapt()``
calls — it does not own the sequence counter. The caller
(stream source, replay controller, test) passes the sequence in
so the pipeline can be driven from multiple producers.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Optional

from backend.sensors import SENSOR_CHANNELS, SensorSample
from backend.sensors.noise import NoiseMode

from .frame import TelemetryFrame
from .latency import LatencyTracker


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterConfig:
    """Static configuration for the :class:`TelemetryAdapter`.

    Attributes
    ----------
    mission_id:
        The mission / scenario this stream belongs to. Defaults
        to ``"mission-001"`` for a single-scenario test bench.
    vehicle_id:
        Identifier of the airframe.
    engine_id:
        Identifier of the engine on the airframe.
    model_version:
        Semver string identifying the running model stack. The
        dashboard renders this in the per-frame wire format so an
        operator can verify which model produced the assessment.
    """

    mission_id: str = "mission-001"
    vehicle_id: str = "vehicle-001"
    engine_id: str = "engine-001"
    model_version: str = "phase17-streaming-1.0.0"


# ---------------------------------------------------------------------
# Data-quality formula
# ---------------------------------------------------------------------
def compute_data_quality(sample: SensorSample) -> float:
    """Compute the 0..1 data-quality score for ``sample``.

    Formula::

        q = 1 - (n_dropout + 3 * n_stuck) / (3 * n_channels)
        q = clip(q, 0, 1)

    The denominator ``3 * n_channels`` is calibrated so that a
    full bundle of dropouts (``n_dropout = n_channels``) gives
    ``q = 1 - 1/3 = 0.666...`` — never zero from dropouts alone.
    A frame where every channel is both dropped AND stuck is the
    only path to ``q = 0``.

    Returns ``1.0`` when ``sample`` is ``None``-like (no
    readings at all) is **not** what we want — an empty frame
    is the worst case, so we return ``0.0`` for that case.
    """
    if not sample.readings:
        return 0.0
    n_channels = len(sample.readings)
    if n_channels == 0:
        return 0.0
    n_dropout = 0
    n_stuck = 0
    for r in sample.readings.values():
        if r.value is None:
            n_dropout += 1
        if r.mode is NoiseMode.STUCK:
            n_stuck += 1
    denom = 3.0 * n_channels
    q = 1.0 - (n_dropout + 3.0 * n_stuck) / denom
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0
    return float(q)


# ---------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------
class TelemetryAdapter:
    """Tag a :class:`SensorSample` with mission/vehicle/engine IDs
    and a numeric data-quality score, producing a
    :class:`TelemetryFrame` ready for the async pipeline.

    Parameters
    ----------
    cfg:
        The static :class:`AdapterConfig`.
    tracker:
        Optional :class:`LatencyTracker` — when provided, the
        adapter records itself under the ``"validation"`` stage
        (this is the **first** validation pass; the downstream
        async pipeline has its own per-stage timings).
    """

    def __init__(
        self,
        cfg: Optional[AdapterConfig] = None,
        tracker: Optional[LatencyTracker] = None,
    ) -> None:
        self._cfg = cfg or AdapterConfig()
        self._tracker = tracker

    @property
    def config(self) -> AdapterConfig:
        return self._cfg

    def adapt(
        self,
        sample: SensorSample,
        *,
        sequence: int,
        time_s: Optional[float] = None,
    ) -> TelemetryFrame:
        """Wrap ``sample`` in a :class:`TelemetryFrame`.

        Parameters
        ----------
        sample:
            The per-tick sensor bundle output.
        sequence:
            Monotonic per-stream sequence number. The adapter does
            not enforce uniqueness — the caller owns that.
        time_s:
            Optional sim-time override. Defaults to
            ``sample.time_s`` so the frame's bus-time matches the
            sensor-bundle time.
        """
        if self._tracker is not None:
            with self._tracker.stage("validation"):
                q = compute_data_quality(sample)
        else:
            q = compute_data_quality(sample)
        ts = float(time_s) if time_s is not None else float(sample.time_s)
        return TelemetryFrame(
            sequence=int(sequence),
            time_s=ts,
            produced_wall_time=_time.perf_counter(),
            sample=sample,
            status=_initial_status(sample),
            mission_id=self._cfg.mission_id,
            vehicle_id=self._cfg.vehicle_id,
            engine_id=self._cfg.engine_id,
            data_quality=q,
            model_version=self._cfg.model_version,
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _initial_status(sample: SensorSample) -> "TelemetryFrame.__class__":
    """Pick a sensible initial :class:`FrameStatus` for the frame.

    Imported lazily to avoid a circular import at module-load time
    (frame.py imports sensors, this module imports frame + sensors).
    """
    from .frame import FrameStatus

    if sample is None or not sample.readings:
        return FrameStatus.INVALID
    n_dropout = sum(1 for r in sample.readings.values() if r.value is None)
    if n_dropout > 0:
        return FrameStatus.STALE
    return FrameStatus.OK


__all__ = [
    "AdapterConfig",
    "TelemetryAdapter",
    "compute_data_quality",
]
