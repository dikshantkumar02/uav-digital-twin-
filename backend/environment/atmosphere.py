"""
ISA1976 standard atmosphere.

Pure-physics implementation:
* Troposphere (0 <= h <= 11 km) — linear temperature lapse.
* Tropopause + lower stratosphere (11 km < h <= 20 km) — isothermal.

The model returns pressure, temperature, density and speed of sound
as a function of geometric altitude. Inputs are validated; outputs are
floats with explicit SI units.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import Atmosphere as AtmosphereCfg, SeaLevel

# Physical constants (SI).
_GAMMA = 1.4                 # ratio of specific heats for dry air
_R_AIR = 287.05287           # specific gas constant for dry air, J/(kg*K)
_G = 9.80665                 # standard gravity, m/s^2


@dataclass(frozen=True)
class AtmosphericState:
    """Snapshot of atmospheric properties at one altitude."""

    altitude_m: float
    pressure_pa: float
    temperature_k: float
    temperature_c: float
    density_kg_per_m3: float
    speed_of_sound_mps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "altitude_m": self.altitude_m,
            "pressure_pa": self.pressure_pa,
            "temperature_k": self.temperature_k,
            "temperature_c": self.temperature_c,
            "density_kg_per_m3": self.density_kg_per_m3,
            "speed_of_sound_mps": self.speed_of_sound_mps,
        }


class Atmosphere:
    """ISA1976 atmosphere with a configurable sea-level reference."""

    def __init__(self, cfg: AtmosphereCfg) -> None:
        self._cfg = cfg
        self._sl = cfg.sea_level
        self._lapse = cfg.lapse_rate_k_per_m
        self._tropopause_m = cfg.tropopause_m
        if self._lapse <= 0:
            raise ValueError("lapse_rate_k_per_m must be positive (ISA uses 0.0065)")
        if self._tropopause_m <= 0:
            raise ValueError("tropopause_m must be positive")
        if self._sl.pressure_pa <= 0 or self._sl.temperature_k <= 0 or self._sl.density_kg_per_m3 <= 0:
            raise ValueError("sea-level values must be strictly positive")

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------
    def state(self, altitude_m: float) -> AtmosphericState:
        """Return atmospheric state at geometric altitude (m).

        Negative altitudes are clamped to 0 (sea level). The tropopause
        altitude is taken from configuration so the model is tunable
        for non-standard conditions while still using the ISA gradient.
        """
        h = max(0.0, float(altitude_m))
        sl = self._sl

        if h <= self._tropopause_m:
            # Troposphere: T = T0 - L * h ; P = P0 * (T/T0)^(g/(L*R))
            t_k = sl.temperature_k - self._lapse * h
            t_k = max(t_k, 1.0)  # hard floor — shouldn't trigger in normal use
            exponent = (_G / (_R_AIR * self._lapse))
            p_pa = sl.pressure_pa * (t_k / sl.temperature_k) ** exponent
        else:
            # Lower stratosphere: T = const ; P = P_trop * exp(-g*(h-h_t)/(R*T))
            t_trop_k = sl.temperature_k - self._lapse * self._tropopause_m
            exponent = (_G / (_R_AIR * self._lapse))
            p_trop = sl.pressure_pa * (t_trop_k / sl.temperature_k) ** exponent
            t_k = t_trop_k
            p_pa = p_trop * (_G * (h - self._tropopause_m) / (_R_AIR * t_k))
            p_pa = p_trop * pow(2.718281828459045, -(_G * (h - self._tropopause_m) / (_R_AIR * t_k)))

        # Density from ideal gas: rho = P / (R * T)
        rho = p_pa / (_R_AIR * t_k)

        # Speed of sound: a = sqrt(gamma * R * T)
        a = (_GAMMA * _R_AIR * t_k) ** 0.5

        return AtmosphericState(
            altitude_m=h,
            pressure_pa=p_pa,
            temperature_k=t_k,
            temperature_c=t_k - 273.15,
            density_kg_per_m3=rho,
            speed_of_sound_mps=a,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def pressure_pa(self, altitude_m: float) -> float:
        return self.state(altitude_m).pressure_pa

    def temperature_c(self, altitude_m: float) -> float:
        return self.state(altitude_m).temperature_c

    def density(self, altitude_m: float) -> float:
        return self.state(altitude_m).density_kg_per_m3

    def sea_level(self) -> AtmosphericState:
        return self.state(0.0)

    @staticmethod
    def from_dict(cfg_dict: dict) -> "Atmosphere":
        """Convenience factory for ad-hoc construction (e.g. in tests)."""
        return Atmosphere(AtmosphereCfg.model_validate(cfg_dict))


__all__ = ["Atmosphere", "AtmosphericState"]
