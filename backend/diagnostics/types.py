"""
Public types for the anomaly-detection layer (PHASE 8).

Three dataclasses are the **stable surface** the rest of the system
consumes:

* :class:`AnomalyLabel` — coarse-grained status of a channel or a
  whole assessment.
* :class:`ChannelAnomaly` — the per-channel score, label, confidence
  and the contributing signals (for explainability).
* :class:`AnomalyAssessment` — the per-tick output of the detector.

The thresholds that turn scores into labels are kept in
:class:`AnomalyThresholds` so callers (and PHASE 9's ML layer) can
adjust them without touching the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AnomalyLabel(str, Enum):
    """Coarse-grained anomaly status.

    * ``NORMAL`` — score below the warn threshold.
    * ``WARN`` — score between warn and anomaly thresholds.
    * ``ANOMALY`` — score at or above the anomaly threshold.
    * ``INSUFFICIENT_DATA`` — not enough evidence to decide; the
      score and confidence are not meaningful in this state. This
      mirrors the spec's ``INSUFFICIENT_DATA`` / ``SENSOR_DATA_INVALID``
      pattern: a missing output is reported as such, never faked.
    """

    NORMAL = "NORMAL"
    WARN = "WARN"
    ANOMALY = "ANOMALY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class AnomalyThresholds:
    """Score-to-label thresholds. All in [0, 1]."""

    warn_score: float = 0.45
    anomaly_score: float = 0.70
    min_confidence: float = 0.30


def _label_for(score: float, thr: AnomalyThresholds) -> AnomalyLabel:
    if score >= thr.anomaly_score:
        return AnomalyLabel.ANOMALY
    if score >= thr.warn_score:
        return AnomalyLabel.WARN
    return AnomalyLabel.NORMAL


@dataclass(frozen=True)
class ChannelAnomaly:
    """Per-channel anomaly output."""

    channel: str
    score: float                          # 0..1, higher = more anomalous
    label: AnomalyLabel
    confidence: float                     # 0..1
    contributors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "score": self.score,
            "label": self.label.value,
            "confidence": self.confidence,
            "contributors": list(self.contributors),
        }


@dataclass(frozen=True)
class AnomalyAssessment:
    """Per-tick anomaly output of the detector."""

    time_s: float
    overall_score: float                  # 0..1
    overall_label: AnomalyLabel
    confidence: float                     # 0..1
    channels: Dict[str, ChannelAnomaly] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    frame_status: str = "OK"              # mirror of the input TelemetryFrame.status

    @property
    def is_anomaly(self) -> bool:
        return self.overall_label is AnomalyLabel.ANOMALY

    @property
    def is_warn(self) -> bool:
        return self.overall_label is AnomalyLabel.WARN

    @property
    def is_insufficient(self) -> bool:
        return self.overall_label is AnomalyLabel.INSUFFICIENT_DATA

    @property
    def contributing_channels(self) -> List[str]:
        return [
            name for name, ca in self.channels.items()
            if ca.label in (AnomalyLabel.ANOMALY, AnomalyLabel.WARN)
        ]

    def to_dict(self) -> dict:
        out: dict = {
            "time_s": self.time_s,
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.value,
            "confidence": self.confidence,
            "frame_status": self.frame_status,
            "notes": list(self.notes),
            "contributing_channels": self.contributing_channels,
        }
        for name, ca in self.channels.items():
            for k, v in ca.to_dict().items():
                out[f"channel.{name}.{k}"] = v
        return out


__all__ = [
    "AnomalyAssessment",
    "AnomalyLabel",
    "AnomalyThresholds",
    "ChannelAnomaly",
]
