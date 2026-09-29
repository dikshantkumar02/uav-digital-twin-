"""
Dashboard snapshot — the wire format (PHASE 13 + PHASE 17 + PHASE 19).

A :class:`DashboardSnapshot` is the single per-tick payload the
dashboard publishes. It bundles every per-tick dataclass produced
by PHASES 2-12 (engine state, environment, anomaly, fault
classification, health, RUL, risk) into a frozen record whose
:meth:`to_dict` is the JSON-serialisable wire format consumed by
the FastAPI endpoints and the WebSocket stream.

The wire format is the dashboard's public data contract — the
frontend is built against the keys produced by :meth:`to_dict`.

PHASE 17 (real-time streaming pipeline) adds a ``latency`` block
so the dashboard shows the **measured** per-stage latency, not
a hard-coded constant.

PHASE 19 (control-room dashboard) extends the wire format with:

* ``residual`` — the per-channel Digital Twin observed/predicted
  residual, used by the "DT expected vs observed" chart.
* ``twin_expected`` — a flat ``{channel: predicted}`` dict, the
  EGT / RPM / fuel-flow expected-value series.
* ``alerts`` — the deduplicated active alert list (FAULT TYPE /
  CONFIDENCE / EVIDENCE / ENV / SENSOR HEALTH / RECOMMENDED
  INVESTIGATION), derived by :mod:`backend.dashboard.alerts`.
* ``events`` — recent scenario events (fault onsets, sensor
  faults, mission-phase transitions).
* ``model_confidence`` — the aggregated model confidence, a
  single 0..1 number for the bottom strip.

All PHASE 19 fields are optional and default to ``None`` /
empty so the existing PHASE 13 / 17 / 18 callers keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.environment import EnvironmentState
from backend.simulation import EngineState
from backend.diagnostics import AnomalyAssessment
from backend.health import HealthIndex
from backend.rul import RulEstimate
from backend.risk import RiskAssessment
from backend.ml import FaultClassification


# ---------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------
WIRE_KEYS = (
    "time_s",
    "tick_index",
    "scenario_name",
    "frame_status",
    "engine_state",
    "environment",
    "anomaly",
    "fault_classification",
    "health",
    "rul",
    "risk",
    "latency",
    "metadata",
    # PHASE 19 — control-room dashboard additions
    "residual",
    "twin_expected",
    "alerts",
    "events",
    "model_confidence",
)

# ``metadata`` is a nested dict with these keys; listed here so
# the dashboard front-end and tests can iterate over the contract
# without re-typing them.
METADATA_KEYS = (
    "mission_id",
    "vehicle_id",
    "engine_id",
    "data_quality",
    "model_version",
)


@dataclass(frozen=True)
class DashboardSnapshot:
    """A frozen per-tick bundle of all PHASE 2-12 outputs.

    Attributes
    ----------
    time_s:
        Simulator time at this tick (seconds).
    tick_index:
        Monotonically increasing index from 0.
    scenario_name:
        Name of the active scenario (from :data:`SCENARIO_REGISTRY`).
    frame_status:
        PHASE 5 telemetry-frame status — ``"OK"`` (default),
        ``"STALE"``, or ``"INVALID"``. Surfaces honestly to the UI.
    engine_state:
        The truth layer (PHASE 3).
    environment:
        Flight state + atmosphere + wind (PHASE 2).
    health:
        Per-subsystem + overall health index (PHASE 10).
    rul:
        Remaining useful life + uncertainty (PHASE 11).
    risk:
        Mission risk go/no-go + drivers (PHASE 12).
    anomaly:
        Per-tick anomaly assessment (PHASE 8). ``None`` when not
        available.
    fault_classification:
        Per-window fault classification (PHASE 9). ``None`` when
        the classifier is disabled or has not yet produced a
        calibrated output.
    latency:
        PHASE 17 — measured per-stage latency (seconds) and
        end-to-end latency. ``None`` when the snapshot was not
        produced by the async streaming pipeline.
    mission_id / vehicle_id / engine_id / data_quality / model_version:
        PHASE 17 — bus-message metadata. Surfaced through
        :meth:`to_dict` under the ``"metadata"`` key so the top
        level matches :data:`WIRE_KEYS`.
    residual:
        PHASE 19 — per-channel digital-twin residual (observed
        vs. predicted). ``None`` when the snapshot was not
        produced by the digital twin. Each entry is a
        :class:`ChannelResidual` dict with ``observed``,
        ``predicted``, ``residual``, ``z_score``, ``confidence``,
        ``in_bounds``.
    twin_expected:
        PHASE 19 — a flat ``{channel: predicted_value}`` dict,
        the per-channel expected-value series used by the
        "DT expected vs observed" chart.
    alerts:
        PHASE 19 — deduplicated active alerts. Each alert is a
        dict with ``fault_type``, ``confidence``,
        ``severity``, ``evidence``, ``env_context``,
        ``sensor_health``, ``recommendation``, ``time_s``.
        Empty tuple when no condition is active.
    events:
        PHASE 19 — recent scenario events (fault onsets, sensor
        faults, mission-phase transitions). Each event is a
        dict with ``time_s``, ``kind``, ``description``.
    model_confidence:
        PHASE 19 — single 0..1 number for the bottom strip.
        Aggregates ``risk.confidence``, ``health.confidence``,
        ``anomaly.confidence``. ``0.0`` when nothing is
        calibrated.
    """

    time_s: float
    tick_index: int
    scenario_name: str
    frame_status: str
    engine_state: EngineState
    environment: EnvironmentState
    health: HealthIndex
    rul: RulEstimate
    risk: RiskAssessment
    anomaly: Optional[AnomalyAssessment] = None
    fault_classification: Optional[FaultClassification] = None
    latency: Optional[Dict[str, float]] = None
    mission_id: str = "mission-001"
    vehicle_id: str = "vehicle-001"
    engine_id: str = "engine-001"
    data_quality: float = 1.0
    model_version: str = "phase17-streaming-1.0.0"
    # --- PHASE 19 ---
    residual: Optional[Dict[str, Any]] = None
    twin_expected: Optional[Dict[str, float]] = None
    alerts: Tuple[Dict[str, Any], ...] = ()
    events: Tuple[Dict[str, Any], ...] = ()
    model_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict matching :data:`WIRE_KEYS`."""
        d: Dict[str, Any] = {
            "time_s": float(self.time_s),
            "tick_index": int(self.tick_index),
            "scenario_name": self.scenario_name,
            "frame_status": self.frame_status,
            "engine_state": self.engine_state.to_dict(),
            "environment": self.environment.to_dict(),
            "anomaly": self.anomaly.to_dict() if self.anomaly is not None else None,
            "fault_classification": (
                self.fault_classification.to_dict()
                if self.fault_classification is not None
                else None
            ),
            "health": self.health.to_dict(),
            "rul": self.rul.to_dict(),
            "risk": self.risk.to_dict(),
        }
        if self.latency is not None:
            d["latency"] = dict(self.latency)
        else:
            d["latency"] = None
        d["metadata"] = {
            "mission_id": self.mission_id,
            "vehicle_id": self.vehicle_id,
            "engine_id": self.engine_id,
            "data_quality": float(self.data_quality),
            "model_version": self.model_version,
        }
        # PHASE 19 — control-room dashboard blocks.
        d["residual"] = self.residual
        d["twin_expected"] = (
            dict(self.twin_expected) if self.twin_expected is not None else None
        )
        d["alerts"] = list(self.alerts)
        d["events"] = list(self.events)
        d["model_confidence"] = float(self.model_confidence)
        return d


__all__ = ["DashboardSnapshot", "WIRE_KEYS", "METADATA_KEYS"]
