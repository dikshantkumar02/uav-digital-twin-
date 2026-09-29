"""
Health index orchestrator (PHASE 10).

:class:`HealthIndexCalculator` ties together the per-subsystem
scorers (:mod:`subsystems`), the aggregator (:mod:`aggregator`), and
the upstream PHASE 8 / PHASE 9 outputs. One call to
:meth:`HealthIndexCalculator.update` produces a
:class:`~backend.health.types.HealthIndex` for the current tick.

Design notes
------------

* The calculator is **stateful only for the trend buffer**; all
  per-tick math is in pure functions. ``reset()`` clears the trend.
* Any of the upstream inputs may be ``None``: a missing engine
  state zeros out four of the five subsystems; a missing anomaly
  assessment zeros out the ``SENSORS`` subsystem; a missing
  classification contributes no ``contributing_faults``.
* The ``SENSORS`` subsystem is always evaluated, even if the other
  four subsystems are unknown — that is the only way to know
  whether the *other* scores are trustworthy.
"""

from __future__ import annotations

import time as _time
from collections import deque
from typing import Deque, Dict, Mapping, Optional

from backend.config import EngineConfig
from backend.diagnostics import AnomalyAssessment
from backend.faults import FaultClass
from backend.ml import FaultClassification
from backend.simulation import EngineState

from .aggregator import (
    aggregate,
    new_trend_history,
    normalize_weights,
    update_trend_history,
)
from .subsystems import (
    score_lubrication,
    score_mechanical,
    score_performance,
    score_sensors,
    score_thermal,
)
from .types import (
    DEFAULT_WEIGHTS,
    HealthIndex,
    HealthLabel,
    Subsystem,
    SubsystemHealth,
)


# Probability threshold for a fault class to be reported in
# ``contributing_faults`` (PHASE 11 RUL uses this).
FAULT_CONTRIBUTION_THRESHOLD: float = 0.10


class HealthIndexCalculator:
    """Per-tick engine health index calculator.

    Parameters
    ----------
    cfg:
        Engine configuration (for operating envelopes).
    weights:
        Optional override for subsystem weights. Defaults to
        :data:`~backend.health.types.DEFAULT_WEIGHTS`.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        weights: Optional[Mapping[Subsystem, float]] = None,
    ) -> None:
        self._cfg = cfg
        self._weights = normalize_weights(weights or DEFAULT_WEIGHTS)
        self._trend: Deque[float] = new_trend_history()
        self._last: Optional[HealthIndex] = None
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------
    @property
    def weights(self) -> Dict[Subsystem, float]:
        return dict(self._weights)

    @property
    def trend_history(self) -> Deque[float]:
        return self._trend

    @property
    def last_index(self) -> Optional[HealthIndex]:
        return self._last

    @property
    def update_count(self) -> int:
        return self._update_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        state: Optional[EngineState] = None,
        anomaly: Optional[AnomalyAssessment] = None,
        classification: Optional[FaultClassification] = None,
        time_s: Optional[float] = None,
    ) -> HealthIndex:
        """Compute the health index for the current tick.

        Parameters
        ----------
        state:
            Current engine state (PHASE 3). If ``None``, four of
            the five subsystems report INSUFFICIENT_DATA.
        anomaly:
            Per-tick anomaly assessment (PHASE 8). Used by the
            ``SENSORS`` subsystem. May be ``None``.
        classification:
            Per-window fault classification (PHASE 9). Used to
            populate ``contributing_faults``. May be ``None``.
        time_s:
            Time stamp of the update. Defaults to the engine
            state's time, or to ``0.0`` if no state is supplied.
        """
        ts = float(time_s) if time_s is not None else (
            float(state.time_s) if state is not None else 0.0
        )

        subsystems: Dict[Subsystem, SubsystemHealth] = {
            Subsystem.THERMAL: score_thermal(state, self._cfg),
            Subsystem.LUBRICATION: score_lubrication(state, self._cfg),
            Subsystem.PERFORMANCE: score_performance(state, self._cfg),
            Subsystem.MECHANICAL: score_mechanical(state, self._cfg),
            Subsystem.SENSORS: score_sensors(anomaly),
        }

        contributing = self._extract_contributing_faults(classification)
        wear = float(state.wear) if state is not None else None

        idx = aggregate(
            subsystems,
            weights=self._weights,
            time_s=ts,
            wear=wear,
            contributing_faults=contributing,
            trend_history=self._trend,
        )
        # Update the trend buffer with the new overall score.
        update_trend_history(self._trend, idx.overall_score)
        self._last = idx
        self._update_count += 1
        return idx

    def reset(self) -> None:
        """Clear the trend buffer and the last index."""
        self._trend = new_trend_history()
        self._last = None
        self._update_count = 0

    def measure_latency(self, n_iter: int = 100) -> float:
        """Return average microseconds per :meth:`update` call.

        Useful for tests that need a latency budget.
        """
        t0 = _time.perf_counter()
        for _ in range(int(n_iter)):
            self.update()
        elapsed = _time.perf_counter() - t0
        return 1e6 * elapsed / float(n_iter)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_contributing_faults(
        classification: Optional[FaultClassification],
    ) -> Dict[str, float]:
        """Pull the high-probability fault classes out of a classification.

        The PHASE 9 classifier returns a ``probabilities`` dict on
        every classification, but only when a model is loaded. If
        the classifier is uncalibrated, ``probabilities`` is
        typically empty — that's fine, the contribution dict is
        just empty.
        """
        if classification is None:
            return {}
        if not classification.probabilities:
            return {}
        out: Dict[str, float] = {}
        for fc, p in classification.probabilities.items():
            v = float(p)
            if v >= FAULT_CONTRIBUTION_THRESHOLD:
                key: str
                if isinstance(fc, FaultClass):
                    key = fc.value
                else:
                    key = str(fc)
                out[key] = v
        return out


__all__ = [
    "FAULT_CONTRIBUTION_THRESHOLD",
    "HealthIndexCalculator",
]
