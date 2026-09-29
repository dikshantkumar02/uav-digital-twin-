"""
Mission-phase classification + remaining-time helpers (PHASE 12).

The mission profile (PHASE 2) carries no explicit phase tag. This
module derives a coarse phase heuristically from the profile and
the current time, and exposes a small per-phase modifier table
that nudges the risk score by a small signed amount.

All functions are pure and read only the profile's
``waypoints`` (a :class:`list` of :class:`Waypoint`).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from backend.config import Mission, Waypoint
from backend.environment import MissionProfile

from .types import MissionPhase


# Convenience alias for callers that pass a raw config Mission.
_ProfileLike = Union[MissionProfile, Mission]


# ---------------------------------------------------------------------
# Phase modifiers
# ---------------------------------------------------------------------
# Small signed additions to the final risk score, applied AFTER the
# per-signal contributions and BEFORE the [0, 1] clip. Magnitudes
# are bounded to 0.10 so they nudge the band only at the boundary.
PHASE_MODIFIERS: Dict[MissionPhase, float] = {
    MissionPhase.PRE_FLIGHT: 0.00,
    MissionPhase.TAKEOFF: +0.10,    # no margin to land
    MissionPhase.CLIMB: +0.05,      # limited landing options
    MissionPhase.CRUISE: 0.00,      # neutral
    MissionPhase.DESCENT: -0.02,    # landing options opening up
    MissionPhase.LANDING: -0.05,    # already landing
    MissionPhase.POST_MISSION: 0.00,
}


# ---------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------
# How long after the first waypoint we still consider the mission to
# be in TAKEOFF (sec). The mission profile's first waypoint is the
# start of the operational envelope, so we treat a short window
# immediately afterwards as the takeoff roll.
_TAKEOFF_WINDOW_S: float = 30.0
# Altitude threshold for LANDING detection (m AGL proxy).
_LANDING_ALTITUDE_M: float = 300.0
# Throttle threshold for LANDING detection.
_LANDING_THROTTLE: float = 0.30
# Altitude band for CLIMB/DESCENT vs CRUISE classification (fraction
# of the previous waypoint's altitude).
_ALTITUDE_BAND: float = 0.05


def _prev_waypoint(waypoints, t_s: float) -> Waypoint:
    """Return the waypoint whose time is the largest one <= ``t_s``."""
    chosen = waypoints[0]
    for w in waypoints:
        if w.t_s <= float(t_s):
            chosen = w
        else:
            break
    return chosen


def phase_for(time_s: float, profile: _ProfileLike) -> MissionPhase:
    """Classify the mission phase at time ``time_s`` (seconds).

    Detection rules (evaluated in order):

    1. ``t < waypoints[0].t_s``             -> ``PRE_FLIGHT``
    2. ``t > waypoints[-1].t_s``            -> ``POST_MISSION``
    3. ``t < waypoints[0].t_s + 30``        -> ``TAKEOFF``
    4. ``throttle < 0.3`` AND
       ``altitude < 300 m``                  -> ``LANDING``
    5. ``altitude > prev.altitude * 1.05``  -> ``CLIMB``
    6. ``altitude < prev.altitude * 0.95``  -> ``DESCENT``
    7. otherwise                             -> ``CRUISE``
    """
    if not isinstance(profile, MissionProfile):
        # Allow callers to pass a raw config Mission too.
        profile = MissionProfile.from_config(profile)

    waypoints = profile.waypoints
    t = float(time_s)

    if t < waypoints[0].t_s:
        return MissionPhase.PRE_FLIGHT
    if t > waypoints[-1].t_s:
        return MissionPhase.POST_MISSION

    cmd = profile.at(t)
    if t < waypoints[0].t_s + _TAKEOFF_WINDOW_S:
        return MissionPhase.TAKEOFF

    if cmd.throttle < _LANDING_THROTTLE and cmd.altitude_m < _LANDING_ALTITUDE_M:
        return MissionPhase.LANDING

    prev = _prev_waypoint(waypoints, t)
    if cmd.altitude_m > prev.altitude_m * (1.0 + _ALTITUDE_BAND):
        return MissionPhase.CLIMB
    if cmd.altitude_m < prev.altitude_m * (1.0 - _ALTITUDE_BAND):
        return MissionPhase.DESCENT
    return MissionPhase.CRUISE


def hours_to_destination(
    time_s: float, profile: _ProfileLike
) -> Optional[float]:
    """Return the hours remaining until the end of the mission.

    Returns ``None`` for ``POST_MISSION`` (no destination left) and
    for ``PRE_FLIGHT`` callers (the mission has not started).
    """
    if not isinstance(profile, MissionProfile):
        profile = MissionProfile.from_config(profile)

    t = float(time_s)
    if t > profile.duration_s:
        return None
    if t < 0.0:
        return 0.0
    return max(0.0, (profile.duration_s - t) / 3600.0)


__all__ = [
    "PHASE_MODIFIERS",
    "hours_to_destination",
    "phase_for",
]
