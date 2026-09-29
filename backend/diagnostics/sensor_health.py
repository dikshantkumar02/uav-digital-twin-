"""
Sensor-health scoring (PHASE 8).

Maps each :class:`~backend.sensors.NoiseMode` to a per-channel
anomaly score and a list of contributors. Pure, table-driven, easy
to override in PHASE 9.

The defaults are SYNTHETIC (see ``docs/PHASE_STATUS.md``). They are
intentionally coarse so a single noisy channel doesn't trip the
detector on its own, but a STUCK or DROPPED channel can't be
ignored either.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from backend.sensors import NoiseMode


# (score, [contributors]) per NoiseMode.
# NORMAL → 0.0, no contributors.
DEFAULT_SENSOR_HEALTH_TABLE: Dict[NoiseMode, Tuple[float, List[str]]] = {
    NoiseMode.NORMAL:    (0.00, []),
    NoiseMode.DRIFTING:  (0.55, ["sensor_drift"]),
    NoiseMode.STUCK:     (0.90, ["sensor_stuck"]),
    NoiseMode.SPIKE:     (0.45, ["sensor_spike"]),
    NoiseMode.DROPPED:   (0.70, ["sensor_dropout"]),
    NoiseMode.FAULT:     (0.70, ["sensor_fault"]),
}


def sensor_health_score(
    mode: NoiseMode,
    table: Dict[NoiseMode, Tuple[float, List[str]]] = None,
) -> Tuple[float, List[str]]:
    """Return ``(score, contributors)`` for the given sensor mode.

    Missing channel (caller passes ``None``) is treated as NORMAL
    with no contributors so the fusion layer can stay simple.
    """
    if mode is None:
        return 0.0, []
    tbl = table or DEFAULT_SENSOR_HEALTH_TABLE
    return tbl.get(mode, (0.0, []))


__all__ = [
    "DEFAULT_SENSOR_HEALTH_TABLE",
    "sensor_health_score",
]
