"""
PHASE 20 — MALE-UAV mission templates.

Each template is a list of 10 :class:`MissionPhaseSpec` in
:attr:`PhaseKind.canonical_order` order. The values are tuned for
a Medium-Altitude Long-Endurance UAV envelope:

* Cruise 5–8 km, 50–80 m/s, throttle 0.5–0.9
* Loiter 10–30 min
* HOT_WEATHER = ISA + ~25 °C
* HIGH_ALTITUDE = cruise 6500 m
* LONG_ENDURANCE = 30 min loiter
* RAPID_THROTTLE = 1 s transitions, throttle oscillates ±0.15
* TURBULENT = turbulence 0.5, gust amplitude 12 m/s
* COMBINED = hot + high-alt + turbulent

The seven templates:
    1.  NORMAL          — typical training / demo mission
    2.  HIGH_ALTITUDE   — high-cruise variant
    3.  HOT_WEATHER     — ISA + 25 °C at sea level, drops with altitude
    4.  LONG_ENDURANCE  — 30 min loiter, otherwise like NORMAL
    5.  RAPID_THROTTLE  — fast throttle transients (1 s)
    6.  TURBULENT       — turbulence 0.5, gust 12 m/s
    7.  COMBINED_STRESS — hot + high-alt + turbulent

All templates share the same canonical phase structure
(startup → taxi → takeoff → climb → cruise → loiter →
high-load manoeuvre → descent → landing → shutdown) — only the
per-phase fields differ. That structure is the MALE-UAV spec's
contract: changing it would be a deliberate operation.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from .phases import MissionPhaseSpec, PhaseKind


# ---------------------------------------------------------------------
# MissionTemplate — the 7 named templates
# ---------------------------------------------------------------------
class MissionTemplate(str, Enum):
    """The 7 built-in MALE-UAV mission templates."""

    NORMAL = "NORMAL"
    HIGH_ALTITUDE = "HIGH_ALTITUDE"
    HOT_WEATHER = "HOT_WEATHER"
    LONG_ENDURANCE = "LONG_ENDURANCE"
    RAPID_THROTTLE = "RAPID_THROTTLE"
    TURBULENT = "TURBULENT"
    COMBINED_STRESS = "COMBINED_STRESS"


# ---------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------
# Each builder returns a 10-element list of MissionPhaseSpec with
# start_t_s=0.0 (ProfileGenerator resolves the actual start times
# at compile time using each phase's duration_s).

_DEFAULT_TRANSITION_S = 5.0
_RAPID_TRANSITION_S = 1.0


def _build(
    *,
    durations_s: List[float],
    altitudes_m: List[float],
    airspeeds_mps: List[float],
    throttles: List[float],
    engine_loads: List[float],
    ambient_temps_c: List[Optional[float]],
    pressures_pa: List[Optional[float]],
    winds_mps: List[float],
    turbulence: List[float],
    transition_s: float = _DEFAULT_TRANSITION_S,
) -> List[MissionPhaseSpec]:
    """Build a 10-phase template from parallel field lists.

    All lists must be length 10 and aligned with
    :attr:`PhaseKind.canonical_order`. The resulting specs have
    ``start_t_s=0.0``; the generator resolves the real start
    times by walking the duration list.
    """
    if len({len(x) for x in (
        durations_s, altitudes_m, airspeeds_mps, throttles, engine_loads,
        ambient_temps_c, pressures_pa, winds_mps, turbulence,
    )} - {10}) > 0:
        raise ValueError("all phase field lists must have length 10")
    kinds = PhaseKind.canonical_order()
    return [
        MissionPhaseSpec(
            kind=kinds[i],
            start_t_s=0.0,                       # resolved at compile time
            duration_s=float(durations_s[i]),
            target_altitude_m=float(altitudes_m[i]),
            target_airspeed_mps=float(airspeeds_mps[i]),
            target_throttle=float(throttles[i]),
            engine_load=float(engine_loads[i]),
            ambient_temperature_c=(
                None if ambient_temps_c[i] is None
                else float(ambient_temps_c[i])
            ),
            pressure_pa=(
                None if pressures_pa[i] is None
                else float(pressures_pa[i])
            ),
            wind_mps=float(winds_mps[i]),
            turbulence_intensity=float(turbulence[i]),
            transition_s=float(transition_s),
        )
        for i in range(10)
    ]


# ---------------------------------------------------------------------
# The 7 templates
# ---------------------------------------------------------------------
def _normal() -> List[MissionPhaseSpec]:
    """NORMAL — typical training / demo mission.

    * Cruise: 5000 m, 65 m/s, throttle 0.75, load 0.3
    * Loiter: 10 min, 5000 m, 50 m/s, throttle 0.55
    * High-load manoeuvre: 30 s at throttle 0.9, +5 m/s wind bias
    * All phases use ISA defaults (ambient / pressure = None)
    * Turbulence: 0.1 (light)

    Durations are tuned so the climb rate stays at or below
    20 m/s (the spec's continuity bound) and the throttle
    transitions stay at or below 0.05 per second.
    """
    return _build(
        # STARTUP=30, TAXI=60, TAKEOFF=20, CLIMB=300, CRUISE=300,
        # LOITER=600, HLM=30, DESCENT=300, LANDING=60, SHUTDOWN=30
        durations_s=[30.0, 60.0, 20.0, 300.0, 300.0, 600.0, 30.0, 300.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 5000.0, 5000.0, 5000.0, 5000.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 40.0, 60.0, 65.0, 50.0, 70.0, 55.0, 40.0, 0.0],
        throttles=[0.10, 0.15, 0.85, 0.80, 0.75, 0.55, 0.90, 0.40, 0.20, 0.05],
        engine_loads=[0.05, 0.05, 0.40, 0.30, 0.30, 0.25, 0.85, 0.20, 0.10, 0.02],
        ambient_temps_c=[None] * 10,
        pressures_pa=[None] * 10,
        winds_mps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        turbulence=[0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
    )


def _high_altitude() -> List[MissionPhaseSpec]:
    """HIGH_ALTITUDE — cruise at 6500 m instead of 5000 m.

    * Cruise: 6500 m, 75 m/s (faster — thinner air)
    * Loiter: 5 min at 6500 m, 60 m/s, throttle 0.7
    * Throttle bias: slightly higher to maintain power
    * Turbulence: light (0.1)
    """
    return _build(
        durations_s=[30.0, 60.0, 25.0, 400.0, 300.0, 300.0, 30.0, 400.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 6500.0, 6500.0, 6500.0, 6500.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 45.0, 70.0, 75.0, 60.0, 80.0, 60.0, 45.0, 0.0],
        throttles=[0.10, 0.15, 0.90, 0.85, 0.80, 0.70, 0.95, 0.45, 0.20, 0.05],
        engine_loads=[0.05, 0.05, 0.45, 0.35, 0.35, 0.30, 0.90, 0.20, 0.10, 0.02],
        ambient_temps_c=[None] * 10,
        pressures_pa=[None] * 10,
        winds_mps=[0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 5.0, 0.0, 0.0, 0.0],
        turbulence=[0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
    )


def _hot_weather() -> List[MissionPhaseSpec]:
    """HOT_WEATHER — ISA + ~25 °C at the surface, dropping with altitude.

    Sets the ambient temperature for every phase explicitly
    (ISA + 25 °C for surface phases, ISA + 25 °C − lapse × altitude
    for altitude phases — the values below were pre-computed for
    5000 m cruise using the ISA lapse of 0.0065 K/m).

    Pressure stays at ISA (so the test for the OVERHEATING fault
    can still trigger from a hot ambient).
    """
    return _build(
        durations_s=[30.0, 60.0, 20.0, 300.0, 300.0, 600.0, 30.0, 300.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 5000.0, 5000.0, 5000.0, 5000.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 40.0, 60.0, 65.0, 50.0, 70.0, 55.0, 40.0, 0.0],
        throttles=[0.15, 0.20, 0.90, 0.85, 0.80, 0.60, 0.95, 0.45, 0.25, 0.05],
        engine_loads=[0.10, 0.10, 0.50, 0.40, 0.40, 0.30, 0.95, 0.25, 0.15, 0.02],
        ambient_temps_c=[38.85, 38.85, 38.2, 7.5, 7.5, 7.5, 7.5, 35.8, 38.85, 38.85],
        pressures_pa=[None] * 10,
        winds_mps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        turbulence=[0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
    )


def _long_endurance() -> List[MissionPhaseSpec]:
    """LONG_ENDURANCE — 30 min loiter, otherwise like NORMAL.

    The total mission length is dominated by the loiter phase.
    Cruise and loiter throttle biased slightly higher because
    long-endurance UAVs are usually fuel-limited and operate
    at conservative power.
    """
    return _build(
        durations_s=[30.0, 60.0, 20.0, 300.0, 300.0, 1800.0, 30.0, 300.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 5000.0, 5000.0, 5000.0, 5000.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 40.0, 60.0, 60.0, 50.0, 70.0, 55.0, 40.0, 0.0],
        throttles=[0.10, 0.15, 0.85, 0.80, 0.70, 0.55, 0.90, 0.40, 0.20, 0.05],
        engine_loads=[0.05, 0.05, 0.40, 0.30, 0.30, 0.25, 0.85, 0.20, 0.10, 0.02],
        ambient_temps_c=[None] * 10,
        pressures_pa=[None] * 10,
        winds_mps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        turbulence=[0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
    )


def _rapid_throttle() -> List[MissionPhaseSpec]:
    """RAPID_THROTTLE — fast throttle transients (1 s transitions).

    Same envelope as NORMAL but every ``transition_s`` is 1 s
    instead of 5 s, so the linear phase interpolation produces
    steeper throttle ramps. Useful for testing engine response
    under aggressive throttle changes.
    """
    return _build(
        durations_s=[30.0, 60.0, 20.0, 300.0, 300.0, 600.0, 30.0, 300.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 5000.0, 5000.0, 5000.0, 5000.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 40.0, 60.0, 65.0, 50.0, 70.0, 55.0, 40.0, 0.0],
        # Slightly oscillated throttles (±0.15 around the NORMAL values).
        throttles=[0.25, 0.30, 0.85, 0.95, 0.75, 0.55, 0.90, 0.40, 0.20, 0.20],
        engine_loads=[0.05, 0.05, 0.40, 0.30, 0.30, 0.25, 0.85, 0.20, 0.10, 0.02],
        ambient_temps_c=[None] * 10,
        pressures_pa=[None] * 10,
        winds_mps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        turbulence=[0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
        transition_s=_RAPID_TRANSITION_S,
    )


def _turbulent() -> List[MissionPhaseSpec]:
    """TURBULENT — turbulence 0.5 across the board, gust bias 12 m/s.

    Useful for testing the disturbance pipeline. The UAV envelope
    is the same as NORMAL but the air is rough.
    """
    return _build(
        durations_s=[30.0, 60.0, 20.0, 300.0, 300.0, 600.0, 30.0, 300.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 5000.0, 5000.0, 5000.0, 5000.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 40.0, 60.0, 65.0, 50.0, 70.0, 55.0, 40.0, 0.0],
        throttles=[0.10, 0.15, 0.85, 0.80, 0.75, 0.55, 0.90, 0.40, 0.20, 0.05],
        engine_loads=[0.05, 0.05, 0.40, 0.30, 0.30, 0.25, 0.85, 0.20, 0.10, 0.02],
        ambient_temps_c=[None] * 10,
        pressures_pa=[None] * 10,
        winds_mps=[3.0, 3.0, 5.0, 8.0, 10.0, 10.0, 12.0, 8.0, 5.0, 3.0],
        turbulence=[0.20, 0.20, 0.30, 0.40, 0.50, 0.50, 0.50, 0.40, 0.20, 0.10],
    )


def _combined_stress() -> List[MissionPhaseSpec]:
    """COMBINED_STRESS — hot + high-alt + turbulent.

    Tests the *most demanding* realistic envelope: a 6500 m
    cruise in 30 °C ambient with 0.5 turbulence and 12 m/s gusts.
    The OVERHEATING fault should be trivially reproducible in
    this template because the ambient heat is already biased
    high and the engine is running at high throttle.
    """
    return _build(
        durations_s=[30.0, 60.0, 25.0, 400.0, 300.0, 600.0, 30.0, 400.0, 60.0, 30.0],
        altitudes_m=[0.0, 0.0, 50.0, 6500.0, 6500.0, 6500.0, 6500.0, 500.0, 0.0, 0.0],
        airspeeds_mps=[0.0, 15.0, 45.0, 70.0, 75.0, 60.0, 80.0, 60.0, 45.0, 0.0],
        throttles=[0.15, 0.20, 0.90, 0.85, 0.80, 0.65, 0.95, 0.45, 0.25, 0.05],
        engine_loads=[0.10, 0.10, 0.45, 0.40, 0.40, 0.30, 0.95, 0.25, 0.15, 0.02],
        # ISA + 25 °C offset, again pre-computed for 6500 m
        # (ISA at 6500 m ≈ -24 °C, +25 °C offset ≈ +1 °C) and
        # 500 m (~ +35.8 °C).
        ambient_temps_c=[38.85, 38.85, 38.2, 1.0, 1.0, 1.0, 1.0, 35.8, 38.85, 38.85],
        pressures_pa=[None] * 10,
        winds_mps=[3.0, 3.0, 5.0, 8.0, 10.0, 10.0, 12.0, 8.0, 5.0, 3.0],
        turbulence=[0.20, 0.20, 0.30, 0.40, 0.50, 0.50, 0.50, 0.40, 0.20, 0.10],
    )


# ---------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------
MISSION_TEMPLATES: Dict[MissionTemplate, List[MissionPhaseSpec]] = {
    MissionTemplate.NORMAL: _normal(),
    MissionTemplate.HIGH_ALTITUDE: _high_altitude(),
    MissionTemplate.HOT_WEATHER: _hot_weather(),
    MissionTemplate.LONG_ENDURANCE: _long_endurance(),
    MissionTemplate.RAPID_THROTTLE: _rapid_throttle(),
    MissionTemplate.TURBULENT: _turbulent(),
    MissionTemplate.COMBINED_STRESS: _combined_stress(),
}


def get_template(name: str) -> Optional[List[MissionPhaseSpec]]:
    """Look up a template by name. Returns ``None`` if unknown.

    Accepts both the enum value (``"NORMAL"``) and the enum
    member name (``"NORMAL"``) — the two are identical strings.
    """
    try:
        tpl = MissionTemplate(name)
    except ValueError:
        return None
    return MISSION_TEMPLATES.get(tpl)


def list_templates() -> List[str]:
    """Return the names of all built-in templates."""
    return [t.value for t in MissionTemplate]


__all__ = [
    "MISSION_TEMPLATES",
    "MissionTemplate",
    "get_template",
    "list_templates",
]
