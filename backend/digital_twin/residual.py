"""
Residual model — observed minus predicted, with confidence.

The Digital Twin emits a per-channel residual each tick. The diagnostic
engine (PHASE 8) consumes these residuals; the RUL engine (PHASE 11)
consumes the long-running statistics.

Residuals are stored as a flat dataclass plus a per-channel dict so the
rest of the system can read either view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# Map: sensor channel -> twin state field name.
# These pairs are what the residual is computed against.
CHANNEL_TO_STATE: Dict[str, str] = {
    "rpm": "rpm",
    "egt": "egt_c",
    "cht": "cht_c",
    "oil_pressure": "oil_pressure_psi",
    "oil_temperature": "oil_temperature_c",
    "fuel_flow": "fuel_flow_lph",
    "vibration": "vibration_rms_g",
    "altitude": "altitude_m",
    "airspeed": "airspeed_mps",
    "ambient_temperature": "ambient_temperature_c",
    "ambient_pressure": "ambient_pressure_pa",
}


@dataclass
class ChannelResidual:
    """One channel's observed vs. predicted residual."""

    channel: str
    observed: Optional[float]
    predicted: float
    residual: Optional[float]      # observed - predicted
    z_score: Optional[float]       # residual / sigma_expected
    confidence: float              # [0, 1] — 1 = very confident
    in_bounds: bool                # |z| <= 3 (typical statistical bound)

    @property
    def is_outlier(self) -> bool:
        return self.z_score is not None and abs(self.z_score) > 3.0

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "observed": self.observed,
            "predicted": self.predicted,
            "residual": self.residual,
            "z_score": self.z_score,
            "confidence": self.confidence,
            "in_bounds": self.in_bounds,
        }


@dataclass
class ResidualFrame:
    """Per-tick residual bundle."""

    time_s: float
    residuals: Dict[str, ChannelResidual] = field(default_factory=dict)
    overall_confidence: float = 1.0

    def __getitem__(self, key: str) -> ChannelResidual:
        return self.residuals[key]

    def to_dict(self) -> dict:
        out: dict = {"time_s": self.time_s, "overall_confidence": self.overall_confidence}
        for name, r in self.residuals.items():
            for k, v in r.to_dict().items():
                out[f"residual.{name}.{k}"] = v
        return out

    def outlier_channels(self) -> list[str]:
        return [name for name, r in self.residuals.items() if r.is_outlier]


__all__ = ["CHANNEL_TO_STATE", "ChannelResidual", "ResidualFrame"]
