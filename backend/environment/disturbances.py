"""
Environmental disturbances.

* :class:`TurbulenceModel` — band-limited Gaussian process approximating
  vertical Dryden turbulence. First-order Gauss-Markov with adjustable
  correlation time.
* :class:`GustModel`       — discrete Poisson-arrival vertical gusts of
  bounded amplitude and short finite duration.

Both models are deterministic given a fixed random generator, so the
mission runner is fully reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.config import GustsCfg, TurbulenceCfg

from .state import WindState


# ---------------------------------------------------------------------
# Turbulence
# ---------------------------------------------------------------------
class TurbulenceModel:
    """First-order Gauss-Markov vertical turbulence.

    The continuous-time process is
        dw/dt = -w / tau + sqrt(2 * sigma^2 / tau) * xi(t)
    discretised with a forward Euler step of ``dt`` seconds. ``sigma`` is
    the steady-state standard deviation, ``tau`` the correlation time.

    The discrete form (re-derived for clarity) is

        w_{k+1} = (1 - dt/tau) * w_k + sqrt(2 * sigma^2 * dt / tau) * N(0,1)

    which preserves the steady-state variance of the continuous process
    to O(dt) accuracy. ``1 - dt/tau`` is clamped to [0, 1] for stability.
    """

    def __init__(self, cfg: TurbulenceCfg, intensity_to_sigma: float = 3.0) -> None:
        if not cfg.enabled:
            self._enabled = False
            self._w = 0.0
            return
        self._enabled = True
        # `intensity` is a unitless envelope in [0, 2]; map it to a
        # physically plausible vertical wind sigma. 3 m/s per "unit"
        # is a representative moderate-turbulence calibration.
        self._sigma = float(cfg.intensity) * float(intensity_to_sigma)
        self._tau = float(cfg.correlation_time_s)
        if self._tau <= 0:
            raise ValueError("turbulence correlation_time_s must be positive")
        self._rng = np.random.default_rng(cfg.seed)
        self._w = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_intensity(self, value: float, intensity_to_sigma: float = 3.0) -> None:
        """Update the turbulence envelope at runtime (clamped to [0, 2]).

        Used by the fault injector (PHASE 7) for the
        ``ENVIRONMENTAL_DISTURBANCE`` class. The underlying RNG is
        unchanged so determinism is preserved.
        """
        v = max(0.0, min(2.0, float(value)))
        self._sigma = v * float(intensity_to_sigma)

    def reset(self) -> None:
        self._w = 0.0

    def step(self, dt: float) -> float:
        """Advance the turbulence process by ``dt`` seconds and return the
        current vertical wind speed (m/s).
        """
        if not self._enabled or dt <= 0.0:
            return self._w
        # Forward-Euler update with variance preservation.
        alpha = max(0.0, min(1.0, 1.0 - dt / self._tau))
        noise = self._rng.normal(0.0, 1.0)
        var_step = max(0.0, 1.0 - alpha * alpha) * (self._sigma ** 2)
        increment = math.sqrt(var_step) * noise
        self._w = alpha * self._w + increment
        return self._w


# ---------------------------------------------------------------------
# Gusts
# ---------------------------------------------------------------------
@dataclass
class GustEvent:
    """Active discrete gust."""

    start_s: float
    duration_s: float
    amplitude_mps: float       # signed peak amplitude (m/s)


class GustModel:
    """Discrete vertical gusts with Poisson arrival and finite duration.

    Each gust is a raised-cosine pulse of length ``duration_s`` and peak
    ``amplitude_mps``. Arrival times are drawn from a Poisson process
    with rate ``rate_per_hour``.
    """

    def __init__(self, cfg: GustsCfg, duration_s: float = 8.0) -> None:
        self._enabled = cfg.enabled
        self._rate_per_s = cfg.rate_per_hour / 3600.0
        self._max_amplitude = cfg.amplitude_mps
        self._duration_s = duration_s
        self._rng = np.random.default_rng(cfg.seed)
        self._next_arrival_s: float = self._sample_interarrival()
        self._active: Optional[GustEvent] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_amplitude(self, value: float) -> None:
        """Update the gust peak amplitude at runtime (must be > 0).

        Used by the fault injector (PHASE 7) for the
        ``ENVIRONMENTAL_DISTURBANCE`` class. The RNG is unchanged so
        determinism is preserved across calls.
        """
        if value > 0.0:
            self._max_amplitude = float(value)

    def reset(self) -> None:
        self._next_arrival_s = self._sample_interarrival()
        self._active = None

    def _sample_interarrival(self) -> float:
        if not self._enabled or self._rate_per_s <= 0.0:
            return float("inf")
        u = max(self._rng.uniform(1e-9, 1.0), 1e-9)
        return -math.log(u) / self._rate_per_s

    def step(self, t_s: float, dt: float) -> float:
        """Advance the gust process to time ``t_s``; return current gust (m/s)."""
        if not self._enabled:
            return 0.0

        # Schedule new gusts that should have started by t_s.
        # A new arrival that fires while a gust is in progress replaces
        # the current one — this preserves the configured arrival rate
        # instead of dropping overlapping events.
        while t_s >= self._next_arrival_s:
            amp = float(self._rng.uniform(-self._max_amplitude, self._max_amplitude))
            self._active = GustEvent(
                start_s=self._next_arrival_s,
                duration_s=self._duration_s,
                amplitude_mps=amp,
            )
            self._next_arrival_s += self._sample_interarrival()

        if self._active is None:
            return 0.0

        elapsed = t_s - self._active.start_s
        if elapsed < 0.0 or elapsed > self._active.duration_s:
            self._active = None
            return 0.0

        # Raised-cosine envelope: 0.5 * (1 - cos(pi * t / duration))
        # Gives smooth start/end with peak at midpoint.
        x = elapsed / self._active.duration_s
        env = 0.5 * (1.0 - math.cos(math.pi * x))
        return self._active.amplitude_mps * env

    @property
    def active_event(self) -> Optional[GustEvent]:
        return self._active


# ---------------------------------------------------------------------
# Composite wind (turbulence + gust)
# ---------------------------------------------------------------------
def compute_wind(
    t_s: float,
    dt: float,
    turb: TurbulenceModel,
    gust: GustModel,
    steady_wind_mps: float = 0.0,
) -> WindState:
    """Compose turbulence and gust contributions for the current step.

    The optional ``steady_wind_mps`` is a per-phase steady bias
    layered on top of the stochastic turbulence + gust. It is
    additive with respect to ``total_w_mps`` and exposed as a
    separate field so the per-tick output preserves the
    decomposition. The default of ``0.0`` keeps the existing wire
    format unchanged for callers that do not use mission profiles.
    """
    w_t = turb.step(dt)
    w_g = gust.step(t_s, dt)
    return WindState(
        time_s=t_s,
        turbulence_w_mps=w_t,
        gust_w_mps=w_g,
        total_w_mps=w_t + w_g,
        is_gust_active=gust.active_event is not None,
        steady_wind_mps=float(steady_wind_mps),
    )


__all__ = ["GustEvent", "GustModel", "TurbulenceModel", "compute_wind"]
