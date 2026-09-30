"""
Pydantic data models for simulation requests, responses, and health endpoints.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


def _check_finite(v: Optional[float], field_name: str) -> Optional[float]:
    if v is not None and not math.isfinite(v):
        raise ValueError(f"{field_name} must be a finite numeric value")
    return v


class SimulationRequest(BaseModel):
    """Input payload for POST /simulate."""

    rpm: float = Field(
        ...,
        ge=0.0,
        description="Crankshaft rotational speed in RPM (must be >= 0)",
        examples=[5000.0],
    )
    throttle: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Throttle lever position between 0.0 (idle) and 1.0 (full open)",
        examples=[0.85],
    )
    manifold_pressure_inhg: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional explicit manifold absolute pressure (inHg). If omitted, derived from throttle.",
        examples=[26.5],
    )
    atmospheric_pressure_inhg: float = Field(
        29.92,
        gt=0.0,
        description="Ambient atmospheric pressure in inHg (standard day is 29.92 inHg / 101.325 kPa)",
        examples=[29.92],
    )
    ambient_temperature_c: float = Field(
        15.0,
        description="Outside air ambient temperature in degrees Celsius",
        examples=[15.0],
    )
    altitude_m: float = Field(
        0.0,
        ge=-500.0,
        le=15000.0,
        description="Geometric altitude above mean sea level in meters",
        examples=[1000.0],
    )

    @field_validator(
        "rpm",
        "throttle",
        "manifold_pressure_inhg",
        "atmospheric_pressure_inhg",
        "ambient_temperature_c",
        "altitude_m",
        mode="after",
    )
    @classmethod
    def validate_finite(cls, v: Optional[float], info: Any) -> Optional[float]:
        return _check_finite(v, info.field_name)

    @model_validator(mode="after")
    def validate_pressures_and_rpm(self) -> SimulationRequest:
        MAX_SIM_RPM = 6500.0
        if self.rpm > MAX_SIM_RPM:
            raise ValueError(
                f"RPM {self.rpm} exceeds simulation maximum permissible limit ({MAX_SIM_RPM} RPM)."
            )

        if self.manifold_pressure_inhg is not None:
            if self.manifold_pressure_inhg > (self.atmospheric_pressure_inhg + 0.05):
                raise ValueError(
                    f"Naturally aspirated manifold pressure ({self.manifold_pressure_inhg:.2f} inHg) "
                    f"cannot exceed ambient atmospheric pressure ({self.atmospheric_pressure_inhg:.2f} inHg)."
                )

        return self


class ModelMetadata(BaseModel):
    """Engine model metadata and certification status."""

    model_id: str = "ROTAX-912S-SIM"
    selected_variant: str = "Rotax 912 S/ULS (Carbureted, 100 hp)"
    model_status: str = "UNOFFICIAL_SIMULATION"
    calibration_status: str = "SYNTHETIC_RESEARCH_CALIBRATION"
    manufacturer_validation: bool = False
    certification_status: str = "UNNOTIFIED_NON_CERTIFIED"
    reduction_gear_ratio: float = 2.4286


class SimulationResponse(BaseModel):
    """Output payload from POST /simulate."""

    crankshaft_rpm: float
    propeller_rpm: float
    estimated_power_kw: float
    estimated_power_hp: float
    manifold_pressure_inhg: float
    fuel_flow_lph: float
    bsfc_g_per_kwh: float
    egt_c: float
    cht_c: float
    oil_pressure_psi: float
    oil_temperature_c: float
    warnings: List[str] = Field(default_factory=list)
    model_metadata: ModelMetadata
    safety_disclaimer: str


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = "healthy"
    app_env: str = "development"
    version: str = "1.0.0"
    timestamp: str


class ReadyResponse(BaseModel):
    """Response payload for GET /ready."""

    status: str = "ready"
    model_ready: bool = True
    model_id: str = "ROTAX-912S-SIM"
    timestamp: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class RootResponse(BaseModel):
    """Response payload for GET /."""

    service: str = "Unofficial Rotax 912 Simulation API"
    version: str = "1.0.0"
    status: str = "running"
    docs_enabled: bool = True
    safety_disclaimer: str
    endpoints: Dict[str, str] = Field(
        default_factory=lambda: {
            "root": "/",
            "health": "/health",
            "ready": "/ready",
            "simulate": "/simulate",
            "docs": "/docs",
        }
    )
