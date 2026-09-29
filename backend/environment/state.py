"""
Unified wind state — what the rest of the system reads each tick.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindState:
    """Vertical wind / disturbance state at one time step."""

    time_s: float
    turbulence_w_mps: float        # band-limited turbulence vertical component
    gust_w_mps: float              # discrete-gust contribution
    total_w_mps: float             # turbulence + gust
    is_gust_active: bool           # True if a discrete gust is currently in progress
    # PHASE 20 — per-phase steady wind bias (m/s). Default 0.0 keeps
    # the existing wind dictionary shape unchanged when no mission
    # profile is in use. The mission-profile generator threads the
    # active phase's ``wind_mps`` through here so the steady bias
    # appears in the per-tick wind state and in
    # :meth:`EnvironmentState.to_dict()`.
    steady_wind_mps: float = 0.0

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "time_s": self.time_s,
            "turbulence_w_mps": self.turbulence_w_mps,
            "gust_w_mps": self.gust_w_mps,
            "total_w_mps": self.total_w_mps,
            "is_gust_active": self.is_gust_active,
            "steady_wind_mps": self.steady_wind_mps,
        }


__all__ = ["WindState"]
