"""
PHASE 20 — MALE-UAV mission phase definitions.

A :class:`MissionPhaseSpec` is the *declarative* per-phase
metadata the MALE-UAV mission-profile generator needs:

* :class:`PhaseKind` — the 10 named phases the spec calls for
  (startup, taxi/ground, takeoff, climb, cruise, loiter,
  high-load manoeuvre, descent, landing, shutdown).
* :class:`MissionPhaseSpec` — per-phase fields the spec requires:
  duration, altitude, airspeed, throttle, engine load, ambient
  temperature, pressure, wind, turbulence level.

A phase is *what the operator wants the engine / UAV to be doing
during this window*. The :class:`ProfileGenerator` compiles a
list of phases into a continuous trajectory — the *transitions
between* phases are handled by the linear interpolation of the
underlying :class:`~backend.environment.mission.MissionProfile`
plus the engine's first-order lags.

The spec is intentionally small and human-readable. The MALE-UAV
envelope (cruise 5–8 km, 50–80 m/s, throttle 0.5–0.9) is in
:mod:`backend.environment.templates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# PhaseKind — the 10 named phases
# ---------------------------------------------------------------------
class PhaseKind(str, Enum):
    """The 10 mission phases the MALE-UAV spec calls for.

    Order is significant: templates are expected to list their
    phases in this canonical order (startup → shutdown) so
    transition windows line up. The dashboard sorts a phase
    list by ``PhaseKind``'s position, so reordering the enum
    is a deliberate operation.
    """

    STARTUP = "STARTUP"                       # engines spool up
    TAXI = "TAXI"                             # ground roll
    TAKEOFF = "TAKEOFF"                       # rotation + initial climb
    CLIMB = "CLIMB"                           # cruise climb to altitude
    CRUISE = "CRUISE"                         # steady cruise
    LOITER = "LOITER"                         # hold pattern / station keeping
    HIGH_LOAD_MANOEUVRE = "HIGH_LOAD_MANOEUVRE"   # dynamic pull-up / bank
    DESCENT = "DESCENT"                       # power reduction + descent
    LANDING = "LANDING"                       # approach + touchdown
    SHUTDOWN = "SHUTDOWN"                     # engines spool down

    @classmethod
    def canonical_order(cls) -> tuple["PhaseKind", ...]:
        """The 10 phases in the spec's canonical order."""
        return (
            cls.STARTUP,
            cls.TAXI,
            cls.TAKEOFF,
            cls.CLIMB,
            cls.CRUISE,
            cls.LOITER,
            cls.HIGH_LOAD_MANOEUVRE,
            cls.DESCENT,
            cls.LANDING,
            cls.SHUTDOWN,
        )


# ---------------------------------------------------------------------
# MissionPhaseSpec — per-phase metadata
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MissionPhaseSpec:
    """Declarative per-phase metadata.

    All fields are required except the two ``Optional`` ones
    (``ambient_temperature_c`` and ``pressure_pa``): leaving them
    ``None`` lets the generator fall back to ISA at the phase
    altitude, which is the right default for most templates. The
    HOT_WEATHER template overrides them.

    Attributes
    ----------
    kind:
        Which of the 10 phases this is.
    start_t_s:
        When the phase begins (sim time, seconds). Resolved at
        compile time by :class:`ProfileGenerator` so the user can
        describe a template by ``duration_s`` only; the start
        times are derived from the canonical phase list.
    duration_s:
        Length of the phase. The *outgoing* transition into the
        next phase spans the last ``transition_s`` of this phase
        (the linear phase interpolation handles the smoothness).
    target_altitude_m:
        Commanded altitude at the end of the phase (metres AGL).
    target_airspeed_mps:
        Commanded airspeed at the end of the phase (m/s).
    target_throttle:
        Commanded throttle at the end of the phase, in [0, 1].
    engine_load:
        Commanded accessory / generator / external-store load in
        [0, 1]. Propagated through :class:`EngineInputs.engine_load`
        to bias vibration / BSFC / fuel flow.
    ambient_temperature_c:
        Optional override of the ambient temperature (``°C``).
        ``None`` means "ISA at the phase altitude".
    pressure_pa:
        Optional override of the ambient pressure (``Pa``).
        ``None`` means "ISA at the phase altitude".
    wind_mps:
        Steady wind bias at this phase (m/s). The atmosphere
        disturbances (turbulence, gusts) are layered on top.
    turbulence_intensity:
        Turbulence envelope in [0, 2]. Maps to the same sigma
        the PHASE 2 :class:`TurbulenceModel` uses.
    transition_s:
        Length of the *outgoing* transition window into the next
        phase. Defaults to 5 s; the ``RAPID_THROTTLE`` template
        shortens this to 1 s.
    """

    kind: PhaseKind
    start_t_s: float
    duration_s: float
    target_altitude_m: float
    target_airspeed_mps: float
    target_throttle: float
    engine_load: float
    ambient_temperature_c: Optional[float]
    pressure_pa: Optional[float]
    wind_mps: float
    turbulence_intensity: float
    transition_s: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PhaseKind):
            raise TypeError(
                f"kind must be a PhaseKind, got {type(self.kind).__name__}"
            )
        if self.duration_s < 0:
            raise ValueError(
                f"duration_s must be >= 0, got {self.duration_s!r}"
            )
        if self.transition_s < 0:
            raise ValueError(
                f"transition_s must be >= 0, got {self.transition_s!r}"
            )
        if not 0.0 <= float(self.target_throttle) <= 1.0:
            raise ValueError(
                f"target_throttle must be in [0, 1], got {self.target_throttle!r}"
            )
        if not 0.0 <= float(self.engine_load) <= 1.0:
            raise ValueError(
                f"engine_load must be in [0, 1], got {self.engine_load!r}"
            )
        if not 0.0 <= float(self.turbulence_intensity) <= 2.0:
            raise ValueError(
                f"turbulence_intensity must be in [0, 2], got {self.turbulence_intensity!r}"
            )

    @property
    def end_t_s(self) -> float:
        """When the phase ends (= start_t_s + duration_s)."""
        return float(self.start_t_s) + float(self.duration_s)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict (wire format)."""
        return {
            "kind": self.kind.value,
            "start_t_s": float(self.start_t_s),
            "duration_s": float(self.duration_s),
            "target_altitude_m": float(self.target_altitude_m),
            "target_airspeed_mps": float(self.target_airspeed_mps),
            "target_throttle": float(self.target_throttle),
            "engine_load": float(self.engine_load),
            "ambient_temperature_c": (
                None if self.ambient_temperature_c is None
                else float(self.ambient_temperature_c)
            ),
            "pressure_pa": (
                None if self.pressure_pa is None
                else float(self.pressure_pa)
            ),
            "wind_mps": float(self.wind_mps),
            "turbulence_intensity": float(self.turbulence_intensity),
            "transition_s": float(self.transition_s),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissionPhaseSpec":
        """Build a :class:`MissionPhaseSpec` from its ``to_dict()`` form."""
        return cls(
            kind=PhaseKind(d["kind"]),
            start_t_s=float(d["start_t_s"]),
            duration_s=float(d["duration_s"]),
            target_altitude_m=float(d["target_altitude_m"]),
            target_airspeed_mps=float(d["target_airspeed_mps"]),
            target_throttle=float(d["target_throttle"]),
            engine_load=float(d["engine_load"]),
            ambient_temperature_c=(
                None if d.get("ambient_temperature_c") is None
                else float(d["ambient_temperature_c"])
            ),
            pressure_pa=(
                None if d.get("pressure_pa") is None
                else float(d["pressure_pa"])
            ),
            wind_mps=float(d["wind_mps"]),
            turbulence_intensity=float(d["turbulence_intensity"]),
            transition_s=float(d.get("transition_s", 5.0)),
        )


__all__ = ["MissionPhaseSpec", "PhaseKind"]
