"""
Public types for the health-index layer (PHASE 10).

Three dataclasses are the **stable surface** the rest of the system
consumes:

* :class:`SubsystemHealth` — per-subsystem score in [0, 1], with the
  channels that drove the score below 1.0 and a per-subsystem
  confidence.
* :class:`HealthIndex` — the per-tick overall health assessment,
  with overall score, label, confidence, trend, and the contributing
  faults (from the PHASE 9 classifier).
* :class:`HealthLabel` — coarse-grained status (HEALTHY / DEGRADED
  / CRITICAL / INSUFFICIENT_DATA). Mirrors the spec's
  "no fake outputs" pattern: missing evidence is reported as such,
  never faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Subsystem(str, Enum):
    """The five engine subsystems tracked by the health index.

    Each subsystem has a dedicated :mod:`subsystems` scorer that maps
    its channel readings + (when available) anomaly scores into a
    score in [0, 1].
    """

    THERMAL = "THERMAL"
    LUBRICATION = "LUBRICATION"
    PERFORMANCE = "PERFORMANCE"
    MECHANICAL = "MECHANICAL"
    SENSORS = "SENSORS"


class HealthLabel(str, Enum):
    """Coarse-grained health status.

    * ``HEALTHY`` — overall score >= 0.80.
    * ``DEGRADED`` — overall score in [0.55, 0.80).
    * ``CRITICAL`` — overall score in (0, 0.55).
    * ``INSUFFICIENT_DATA`` — no evidence (e.g. all subsystems
      reported INSUFFICIENT_DATA, or not enough ticks for a trend).
    """

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class HealthTrend(str, Enum):
    """Trend of the overall health score over recent ticks.

    * ``IMPROVING`` — score_now > prev + 0.02.
    * ``STABLE`` — |score_now - prev| <= 0.02.
    * ``DEGRADING`` — score_now < prev - 0.02.
    * ``INSUFFICIENT_DATA`` — not enough history to compare.
    """

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# Default subsystem weights. Tuned so a single sensor anomaly
# cannot drop the overall index below 0.7 on its own, but a hard
# lubrication failure can. Weights are normalised before use.
DEFAULT_WEIGHTS: Dict[Subsystem, float] = {
    Subsystem.THERMAL: 0.20,
    Subsystem.LUBRICATION: 0.25,
    Subsystem.PERFORMANCE: 0.20,
    Subsystem.MECHANICAL: 0.20,
    Subsystem.SENSORS: 0.15,
}


def label_for(score: float) -> HealthLabel:
    """Map an overall score in [0, 1] to a :class:`HealthLabel`."""
    if score >= 0.80:
        return HealthLabel.HEALTHY
    if score >= 0.55:
        return HealthLabel.DEGRADED
    if score > 0.0:
        return HealthLabel.CRITICAL
    return HealthLabel.INSUFFICIENT_DATA


@dataclass(frozen=True)
class SubsystemHealth:
    """Per-subsystem health output."""

    subsystem: Subsystem
    score: float                                # 0..1, 1 = healthy
    confidence: float                           # 0..1
    contributors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_insufficient(self) -> bool:
        return self.score <= 0.0 and self.confidence <= 0.0

    def to_dict(self) -> dict:
        return {
            "subsystem": self.subsystem.value,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "contributors": list(self.contributors),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class HealthIndex:
    """Per-tick health output of the calculator."""

    time_s: float
    overall_score: float                        # 0..1, 1 = healthy
    overall_label: HealthLabel
    confidence: float                           # 0..1
    subsystems: Dict[Subsystem, SubsystemHealth] = field(default_factory=dict)
    trend: HealthTrend = HealthTrend.INSUFFICIENT_DATA
    wear: Optional[float] = None                # from engine truth model
    contributing_faults: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.overall_label is HealthLabel.HEALTHY

    @property
    def is_degraded(self) -> bool:
        return self.overall_label is HealthLabel.DEGRADED

    @property
    def is_critical(self) -> bool:
        return self.overall_label is HealthLabel.CRITICAL

    @property
    def is_insufficient(self) -> bool:
        return self.overall_label is HealthLabel.INSUFFICIENT_DATA

    @property
    def contributing_channels(self) -> List[str]:
        out: List[str] = []
        for sub in self.subsystems.values():
            out.extend(sub.contributors)
        return out

    def subsystem_score(self, name: Subsystem) -> Optional[float]:
        sub = self.subsystems.get(name)
        if sub is None:
            return None
        return float(sub.score)

    def to_dict(self) -> dict:
        out: dict = {
            "time_s": self.time_s,
            "overall_score": float(self.overall_score),
            "overall_label": self.overall_label.value,
            "confidence": float(self.confidence),
            "trend": self.trend.value,
            "wear": self.wear,
            "contributing_faults": dict(self.contributing_faults),
            "contributing_channels": self.contributing_channels,
            "notes": list(self.notes),
        }
        for sub_name, sub in self.subsystems.items():
            for k, v in sub.to_dict().items():
                out[f"sub.{sub_name.value}.{k}"] = v
        return out


__all__ = [
    "DEFAULT_WEIGHTS",
    "HealthIndex",
    "HealthLabel",
    "HealthTrend",
    "Subsystem",
    "SubsystemHealth",
    "label_for",
]
