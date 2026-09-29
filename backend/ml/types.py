"""
Public types for the fault-classification layer (PHASE 9).

The :class:`FaultClassification` is the per-window output of the
:class:`~backend.ml.classifier.FaultClassifier`. It always carries
a :class:`CalibrationStatus` so consumers can tell a *confident*
classification apart from a *default "I don't know"* response
(when the model has not been trained on this dataset yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from backend.faults import FaultClass


class CalibrationStatus(str, Enum):
    """Calibration state of the underlying ML model.

    * ``MODEL_NOT_CALIBRATED`` — no model has been loaded; the
      classifier returns a default ``HEALTHY`` classification with
      zero confidence and a note. This is the project-spec'd
      "no fake outputs" behaviour.
    * ``CALIBRATED`` — a trained model is in use; the classification
      is meaningful.
    * ``MODEL_DEGRADED`` — the model is loaded but its self-reported
      accuracy on a held-out set is below the configured floor. The
      classification is still emitted but flagged.
    """

    MODEL_NOT_CALIBRATED = "MODEL_NOT_CALIBRATED"
    CALIBRATED = "CALIBRATED"
    MODEL_DEGRADED = "MODEL_DEGRADED"


# The set of feature names the extractor produces. Kept in one
# place so the trainer and the inference path can agree on the
# column order without relying on string-typing at every call site.
FEATURE_NAMES: List[str] = [
    # Per-channel residual stats (11 channels × 4 stats).
    *[f"res.{ch}.mean_abs_z" for ch in (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "altitude", "airspeed",
        "ambient_temperature", "ambient_pressure",
    )],
    *[f"res.{ch}.max_abs_z" for ch in (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "altitude", "airspeed",
        "ambient_temperature", "ambient_pressure",
    )],
    *[f"res.{ch}.std_z" for ch in (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "altitude", "airspeed",
        "ambient_temperature", "ambient_pressure",
    )],
    *[f"res.{ch}.outlier_count" for ch in (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "altitude", "airspeed",
        "ambient_temperature", "ambient_pressure",
    )],
    # Per-window sensor-health aggregates.
    "sensors.dropped_count",
    "sensors.stuck_count",
    "sensors.drift_count",
    "sensors.fault_count",
    "sensors.spike_count",
    # Overall residual stats.
    "residual.overall_mean_abs_z",
    "residual.overall_max_abs_z",
    "residual.overall_outlier_count",
    # Flight state (mean over window).
    "flight.mean_rpm",
    "flight.mean_map_inhg",
    "flight.mean_altitude_m",
    "flight.mean_airspeed_mps",
    "flight.mean_throttle",
    # Environment (mean over window).
    "env.mean_pressure_pa",
    "env.mean_temperature_c",
    "env.mean_wind_mps",
]

FEATURE_DIM: int = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FaultClassification:
    """Per-window output of the classifier."""

    time_s: float
    fault_class: FaultClass
    confidence: float                          # max class probability
    probabilities: Dict[FaultClass, float] = field(default_factory=dict)
    status: CalibrationStatus = CalibrationStatus.MODEL_NOT_CALIBRATED
    features_used: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def is_calibrated(self) -> bool:
        return self.status is not CalibrationStatus.MODEL_NOT_CALIBRATED

    def to_dict(self) -> dict:
        out: dict = {
            "time_s": self.time_s,
            "fault_class": self.fault_class.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "features_used": self.features_used,
            "notes": list(self.notes),
        }
        for fc, p in self.probabilities.items():
            out[f"prob.{fc.value}"] = float(p)
        return out


__all__ = [
    "CalibrationStatus",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FaultClassification",
]
