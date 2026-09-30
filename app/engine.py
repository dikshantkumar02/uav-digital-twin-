"""
Engine simulation logic and 2D performance map evaluation for Rotax 912 simulation.

SAFETY DISCLAIMER:
1. This is an unofficial simulation-only project.
2. It is not an official BRP-Rotax product or certified engine model.
3. It must not be used for real aircraft operation, flight-critical control,
   aircraft certification, maintenance release, or real engine limit determination.
4. Never claim that synthetic engine maps are manufacturer-provided data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import yaml

from app.models import ModelMetadata, SimulationRequest, SimulationResponse

log = logging.getLogger(__name__)

DISCLAIMER_TEXT = (
    "SAFETY DISCLAIMER: Unofficial simulation-only model. Not an official BRP-Rotax product. "
    "Not for real flight operations, aircraft certification, or real limit determination. "
    "Performance maps are synthetic approximations."
)


def _bilinear_interp(
    x_axis: List[float], y_axis: List[float], values: List[List[float]]
) -> Callable[[float, float], float]:
    """Compile a fast 2D bilinear interpolator with edge clamping."""
    nx, ny = len(x_axis), len(y_axis)
    if nx < 2 or ny < 2:
        raise ValueError("Interpolation axes must have at least 2 points each.")
    if len(values) != nx:
        raise ValueError(
            f"Map values row count ({len(values)}) does not match x-axis length ({nx})."
        )
    for i, row in enumerate(values):
        if len(row) != ny:
            raise ValueError(
                f"Map values row {i} length ({len(row)}) does not match y-axis length ({ny})."
            )

    def evaluate(x: float, y: float) -> float:
        if x <= x_axis[0]:
            i = 0
            x_norm = 0.0
        elif x >= x_axis[-1]:
            i = nx - 2
            x_norm = 1.0
        else:
            i = 0
            while i < nx - 2 and x_axis[i + 1] < x:
                i += 1
            x_norm = (x - x_axis[i]) / (x_axis[i + 1] - x_axis[i])

        if y <= y_axis[0]:
            j = 0
            y_norm = 0.0
        elif y >= y_axis[-1]:
            j = ny - 2
            y_norm = 1.0
        else:
            j = 0
            while j < ny - 2 and y_axis[j + 1] < y:
                j += 1
            y_norm = (y - y_axis[j]) / (y_axis[j + 1] - y_axis[j])

        q11 = values[i][j]
        q12 = values[i][j + 1]
        q21 = values[i + 1][j]
        q22 = values[i + 1][j + 1]

        val = (
            (1.0 - x_norm) * (1.0 - y_norm) * q11
            + (1.0 - x_norm) * y_norm * q12
            + x_norm * (1.0 - y_norm) * q21
            + x_norm * y_norm * q22
        )
        return float(val)

    return evaluate


class Rotax912Simulator:
    """Read-only physical simulation engine for Rotax 912 S/ULS."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self._loaded = False
        self._load_config()

    def _load_config(self) -> None:
        path = Path(self.config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Simulation config not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "engine" not in raw:
            raise ValueError(f"Invalid engine configuration in: {path}")

        eng = raw["engine"]
        self.metadata = ModelMetadata(
            model_id=eng.get("model_id", "ROTAX-912S-SIM"),
            selected_variant=eng.get(
                "variant", "Rotax 912 S/ULS (Carbureted, 100 hp)"
            ),
            model_status=eng.get("model_status", "UNOFFICIAL_SIMULATION"),
            calibration_status=eng.get(
                "calibration_status", "SYNTHETIC_RESEARCH_CALIBRATION"
            ),
            manufacturer_validation=bool(eng.get("manufacturer_validation", False)),
            certification_status=eng.get(
                "certification_status", "UNNOTIFIED_NON_CERTIFIED"
            ),
            reduction_gear_ratio=float(eng.get("reduction_gear_ratio", 2.4286)),
        )

        geom = eng.get("geometry", {})
        self.idle_rpm = float(geom.get("idle_rpm", 1400.0))
        self.rated_rpm = float(geom.get("rated_rpm", 5500.0))
        self.redline_rpm = float(geom.get("redline_rpm", 5800.0))
        self.rated_power_kw = float(geom.get("rated_power_kw", 69.0))
        self.max_power_kw = 73.5

        tm = eng.get("throttle_to_map", {})
        self.idle_map = float(tm.get("idle_map_inhg", 11.81))
        self.max_map = float(tm.get("max_map_inhg", 29.92))

        limits = eng.get("limits", {})
        self.egt_max_c = float(limits.get("egt_max_c", 850.0))
        self.cht_max_c = float(limits.get("cht_max_c", 135.0))
        self.oil_temp_max_c = float(limits.get("oil_temp_max_c", 130.0))
        self.oil_temp_min_c = float(limits.get("oil_temp_min_c", 50.0))

        pm = eng.get("performance_maps", {})
        self._bsfc_eval = _bilinear_interp(
            pm["bsfc_g_per_kwh"]["rpm_axis"],
            pm["bsfc_g_per_kwh"]["map_axis"],
            pm["bsfc_g_per_kwh"]["values"],
        )
        self._egt_eval = _bilinear_interp(
            pm["egt_map_c"]["rpm_axis"],
            pm["egt_map_c"]["map_axis"],
            pm["egt_map_c"]["values"],
        )
        self._cht_eval = _bilinear_interp(
            pm["cht_map_c"]["rpm_axis"],
            pm["cht_map_c"]["map_axis"],
            pm["cht_map_c"]["values"],
        )

        self._loaded = True
        log.info("Rotax912Simulator initialized successfully from %s", path)

    def is_ready(self) -> bool:
        return self._loaded

    def simulate(self, req: SimulationRequest) -> SimulationResponse:
        warnings: List[str] = [
            "NOTICE: Synthetic engine performance map used. Not certified manufacturer data."
        ]

        p_amb = req.atmospheric_pressure_inhg
        if req.manifold_pressure_inhg is not None:
            map_inhg = min(req.manifold_pressure_inhg, p_amb)
        else:
            throttle = max(0.0, min(1.0, req.throttle))
            map_inhg = self.idle_map + throttle * (p_amb - self.idle_map)

        crankshaft_rpm = req.rpm
        gear_ratio = self.metadata.reduction_gear_ratio
        propeller_rpm = crankshaft_rpm / gear_ratio if gear_ratio > 0 else 0.0

        if crankshaft_rpm > self.redline_rpm:
            warnings.append(
                f"EXCEEDANCE: Crankshaft RPM ({crankshaft_rpm:.0f}) exceeds Take-off Redline ({self.redline_rpm:.0f} RPM)!"
            )
        elif crankshaft_rpm > self.rated_rpm:
            warnings.append(
                f"CAUTION: Crankshaft RPM ({crankshaft_rpm:.0f}) exceeds Maximum Continuous Rating ({self.rated_rpm:.0f} RPM)."
            )
        elif crankshaft_rpm < self.idle_rpm and crankshaft_rpm > 0.0:
            warnings.append(
                f"SUB-IDLE: Engine RPM ({crankshaft_rpm:.0f}) below nominal idle ({self.idle_rpm:.0f} RPM)."
            )

        if req.manifold_pressure_inhg is not None and req.manifold_pressure_inhg > p_amb:
            warnings.append(
                f"AERODYNAMIC LIMIT: Commanded MAP ({req.manifold_pressure_inhg:.2f} inHg) clamped to ambient ({p_amb:.2f} inHg)."
            )

        rpm_ratio = max(0.0, crankshaft_rpm / self.rated_rpm)
        map_ratio = max(0.0, map_inhg / self.max_map)
        power_factor = rpm_ratio * map_ratio

        if crankshaft_rpm >= self.rated_rpm:
            max_avail_kw = self.rated_power_kw + (self.max_power_kw - self.rated_power_kw) * (
                (crankshaft_rpm - self.rated_rpm) / (self.redline_rpm - self.rated_rpm + 1e-6)
            )
        else:
            max_avail_kw = self.rated_power_kw

        power_kw = max(0.0, max_avail_kw * power_factor)
        power_hp = power_kw * 1.34102

        bsfc = self._bsfc_eval(crankshaft_rpm, map_inhg)
        egt = self._egt_eval(crankshaft_rpm, map_inhg)
        cht = self._cht_eval(crankshaft_rpm, map_inhg)

        temp_delta = req.ambient_temperature_c - 15.0
        egt += temp_delta * 0.25
        cht += temp_delta * 0.40

        if egt > self.egt_max_c:
            warnings.append(
                f"THERMAL EXCEEDANCE: Exhaust Gas Temperature ({egt:.1f}°C) exceeds limit ({self.egt_max_c:.1f}°C)!"
            )
        if cht > self.cht_max_c:
            warnings.append(
                f"THERMAL EXCEEDANCE: Cylinder Head Temperature ({cht:.1f}°C) exceeds limit ({self.cht_max_c:.1f}°C)!"
            )

        fuel_density_g_per_l = 720.0
        fuel_flow_lph = (power_kw * bsfc) / fuel_density_g_per_l

        oil_p = 15.0 + 35.0 * min(1.0, crankshaft_rpm / 3500.0)
        oil_temp = 50.0 + (cht - 70.0) * 0.5 + temp_delta * 0.3

        if oil_temp > self.oil_temp_max_c:
            warnings.append(
                f"LUBRICATION WARNING: Oil Temperature ({oil_temp:.1f}°C) exceeds max ({self.oil_temp_max_c:.1f}°C)!"
            )

        return SimulationResponse(
            crankshaft_rpm=round(crankshaft_rpm, 1),
            propeller_rpm=round(propeller_rpm, 1),
            estimated_power_kw=round(power_kw, 2),
            estimated_power_hp=round(power_hp, 2),
            manifold_pressure_inhg=round(map_inhg, 2),
            fuel_flow_lph=round(fuel_flow_lph, 2),
            bsfc_g_per_kwh=round(bsfc, 1),
            egt_c=round(egt, 1),
            cht_c=round(cht, 1),
            oil_pressure_psi=round(oil_p, 1),
            oil_temperature_c=round(oil_temp, 1),
            warnings=warnings,
            model_metadata=self.metadata,
            safety_disclaimer=DISCLAIMER_TEXT,
        )
