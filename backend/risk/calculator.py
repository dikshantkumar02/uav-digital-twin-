"""
Mission risk orchestrator (PHASE 12).

:class:`MissionRiskCalculator` ties together the per-tick
upstream outputs (PHASE 8 anomaly, PHASE 10 health, PHASE 11 RUL)
and the mission profile (PHASE 2) into a per-tick
:class:`~backend.risk.types.RiskAssessment`. The calculator owns
the trend buffer; all per-tick math is in pure functions
(``aggregate``, ``phase_for``, ``hours_to_destination``).
``reset()`` clears the trend.
"""

from __future__ import annotations

import time as _time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from backend.config import RiskConfig, load_config
from backend.diagnostics import AnomalyAssessment
from backend.environment import MissionProfile
from backend.health import HealthIndex
from backend.rul import RulEstimate

from .aggregate import aggregate
from .mission_phase import hours_to_destination, phase_for
from .types import TREND_WINDOW, RiskAssessment
from backend.rul.types import RulTrend


# Default dt for trend interpretation.
DEFAULT_DT_S: float = 0.1


class MissionRiskCalculator:
    """Per-tick mission risk calculator.

    Parameters
    ----------
    profile:
        The :class:`~backend.environment.MissionProfile` driving
        the mission-context helpers (``phase_for``,
        ``hours_to_destination``).
    config:
        Optional :class:`RiskConfig`. Defaults to
        :meth:`RiskConfig.defaults` (the schema defaults), or the
        bundle loaded from ``load_config(...).risk`` when
        ``config_from_yaml_dir`` is provided.
    """

    def __init__(
        self,
        profile: MissionProfile,
        *,
        config: Optional[RiskConfig] = None,
        config_from_yaml_dir: Optional[Path] = None,
    ) -> None:
        if config is not None and config_from_yaml_dir is not None:
            raise ValueError(
                "pass either `config` or `config_from_yaml_dir`, not both"
            )
        self._profile = profile
        if config is not None:
            self._config = config
        elif config_from_yaml_dir is not None:
            self._config = load_config(Path(config_from_yaml_dir)).risk
        else:
            self._config = RiskConfig.defaults()
        self._trend: Deque[float] = deque(maxlen=int(TREND_WINDOW))
        self._last: Optional[RiskAssessment] = None
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------
    @property
    def profile(self) -> MissionProfile:
        return self._profile

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def trend_history(self) -> Deque[float]:
        return self._trend

    @property
    def last_assessment(self) -> Optional[RiskAssessment]:
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
        health: HealthIndex,
        rul: RulEstimate,
        anomaly: Optional[AnomalyAssessment] = None,
        time_s: Optional[float] = None,
        dt_s: float = DEFAULT_DT_S,
    ) -> RiskAssessment:
        """Compute the risk assessment for the current tick.

        Parameters
        ----------
        health:
            Current :class:`HealthIndex` (PHASE 10).
        rul:
            Current :class:`RulEstimate` (PHASE 11).
        anomaly:
            Optional current :class:`AnomalyAssessment` (PHASE 8).
        time_s:
            Time stamp. Defaults to the health index's time, or
            to ``0.0``.
        dt_s:
            Mission dt in seconds (used to interpret trend rates
            and to feed the hours-to-critical extrapolation).
        """
        ts: float
        if time_s is not None:
            ts = float(time_s)
        else:
            ts = float(health.time_s)

        phase = phase_for(ts, self._profile)
        h2d = hours_to_destination(ts, self._profile)

        assessment = aggregate(
            health=health,
            rul=rul,
            anomaly=anomaly,
            phase=phase,
            hours_to_destination=h2d,
            config=self._config,
            trend_history=self._trend,
            time_s=ts,
            dt_s=float(dt_s),
        )
        self._last = assessment
        self._update_count += 1
        return assessment

    def reset(self) -> None:
        """Clear the trend buffer, last assessment, and update count."""
        self._trend = deque(maxlen=int(TREND_WINDOW))
        self._last = None
        self._update_count = 0

    def measure_latency(self, n_iter: int = 200) -> float:
        """Return average microseconds per :meth:`update` call.

        Useful for tests that need a latency budget.
        """
        # Use a minimal-but-valid health + rul so the call shape
        # matches the real path without needing a full pipeline.
        h = _MINIMAL_HEALTH
        r = _MINIMAL_RUL
        t0 = _time.perf_counter()
        for _ in range(int(n_iter)):
            self.update(health=h, rul=r, anomaly=None, time_s=0.0)
        elapsed = _time.perf_counter() - t0
        return 1e6 * elapsed / float(n_iter)


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
from backend.diagnostics import AnomalyLabel
from backend.faults import FaultClass
from backend.health import HealthLabel, HealthTrend
from backend.rul import RulModelStatus, RulStatus

# Minimal inputs for ``measure_latency``. Health is healthy,
# RUL is OK, anomaly is missing — produces a near-zero risk
# score and a GO status.
_MINIMAL_HEALTH = HealthIndex(
    time_s=0.0,
    overall_score=1.0,
    overall_label=HealthLabel.HEALTHY,
    confidence=1.0,
    trend=HealthTrend.STABLE,
)
_MINIMAL_RUL = RulEstimate(
    time_s=0.0,
    tte_hours_central=50_000.0,
    tte_hours_lower=40_000.0,
    tte_hours_upper=60_000.0,
    wear_rate_per_hour=1e-5,
    confidence=1.0,
    status=RulStatus.RUL_OK,
    trend=RulTrend.INSUFFICIENT_DATA,
    model_status=RulModelStatus.CLOSED_FORM,
)


__all__ = [
    "DEFAULT_DT_S",
    "MissionRiskCalculator",
]
