"""
RUL calculator orchestrator (PHASE 11).

:class:`RulCalculator` ties together the wear-rate model (closed
form or trained RF), the Monte-Carlo layer, the aggregator, and
the upstream :class:`~backend.health.HealthIndex` (PHASE 10). One
call to :meth:`update` produces a :class:`RulEstimate` for the
current tick.

The calculator is **stateful only for the trend buffer**; all
per-tick math is in pure functions. ``reset()`` clears the trend.

When a :class:`~backend.rul.model.TrainedWearModel` is loaded,
the rate per hour comes from the RF. Otherwise the closed-form
fallback is used and ``model_status`` reports
``RulModelStatus.CLOSED_FORM`` — the system is functional from
day one, and reports ``RUL_UNCERTAIN`` if the health signals are
insufficient.
"""

from __future__ import annotations

import time as _time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from backend.config import EngineConfig
from backend.health import HealthIndex
from backend.simulation import EngineState

from .aggregate import aggregate, new_trend_history, update_trend_history
from .model import (
    ClosedFormWearRate,
    HybridRulModel,
    TrainedWearModel,
    load_model as load_trained_wear_model,
)
from .monte_carlo import (
    DEFAULT_DT_S,
    DEFAULT_FAULT_DRIFT_PER_HOUR,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_N_SAMPLES,
    DEFAULT_NOISE_STD,
    DEFAULT_QUANTILE_HIGH,
    DEFAULT_QUANTILE_LOW,
    TteDistribution,
    simulate_tte,
)
from .types import (
    ResidualTrendInput,
    RulEstimate,
    RulModelStatus,
    RulStatus,
)


# Default confidence floor for non-uncertain status.
DEFAULT_MIN_CONFIDENCE_FOR_OK: float = 0.60


class RulCalculator:
    """Per-tick Remaining-Useful-Life calculator.

    Parameters
    ----------
    cfg:
        Engine configuration (for base wear rate, max wear).
    model:
        Optional trained :class:`TrainedWearModel`. If ``None``,
        the closed-form fallback is used.
    horizon_hours:
        MC simulation horizon (default 2000 h).
    n_samples:
        Number of MC trajectories (default 200).
    seed:
        Random seed for the MC layer (default 0 — deterministic
        for tests).
    """

    def __init__(
        self,
        cfg: EngineConfig,
        model: Optional[TrainedWearModel] = None,
        model_path: Optional[Path] = None,
        *,
        horizon_hours: float = DEFAULT_HORIZON_HOURS,
        n_samples: int = DEFAULT_N_SAMPLES,
        seed: int = 0,
    ) -> None:
        self._cfg = cfg
        self._wear_model: ClosedFormWearRate = ClosedFormWearRate.from_config(cfg)
        self._hybrid_model = HybridRulModel()
        self._trained: Optional[TrainedWearModel] = model
        if model_path is not None:
            self._trained = load_trained_wear_model(Path(model_path))
        self._horizon_hours = float(horizon_hours)
        self._n_samples = int(n_samples)
        self._seed = int(seed)
        self._trend: Deque[float] = new_trend_history()
        self._last: Optional[RulEstimate] = None
        self._update_count: int = 0
        self._hours_running: float = 0.0

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------
    @property
    def wear_model(self) -> ClosedFormWearRate:
        return self._wear_model

    @property
    def trained_model(self) -> Optional[TrainedWearModel]:
        return self._trained

    @property
    def horizon_hours(self) -> float:
        return self._horizon_hours

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def trend_history(self) -> Deque[float]:
        return self._trend

    @property
    def last_estimate(self) -> Optional[RulEstimate]:
        return self._last

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def hours_running(self) -> float:
        return self._hours_running

    @property
    def model_status(self) -> RulModelStatus:
        if self._trained is None:
            return RulModelStatus.CLOSED_FORM
        return RulModelStatus.MODEL_CALIBRATED

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(self, path: Path) -> None:
        """Load a trained wear-rate model from disk."""
        self._trained = load_trained_wear_model(Path(path))

    def set_trained_model(self, model: TrainedWearModel) -> None:
        self._trained = model

    def reset(self) -> None:
        """Clear the trend buffer, last estimate, and mission clock."""
        self._trend = new_trend_history()
        self._last = None
        self._update_count = 0
        self._hours_running = 0.0

    def update(
        self,
        *,
        health: HealthIndex,
        state: Optional[EngineState] = None,
        residual_trend: Optional[ResidualTrendInput] = None,
        time_s: Optional[float] = None,
        dt_s: float = DEFAULT_DT_S,
        rpm: Optional[float] = None,
        cht_c: Optional[float] = None,
        egt_c: Optional[float] = None,
        oil_pressure_bar: Optional[float] = None,
        oil_temperature_c: Optional[float] = None,
        fuel_flow_lph: Optional[float] = None,
        vibration_rms_g: Optional[float] = None,
        mission_phase: Optional[str] = None,
        engine_age_hours: Optional[float] = None,
        cumulative_cycles: Optional[int] = None,
    ) -> RulEstimate:
        """Compute the RUL estimate for the current tick.

        Parameters
        ----------
        health:
            Current :class:`HealthIndex` (PHASE 10).
        state:
            Optional current :class:`EngineState`.
        residual_trend:
            Optional :class:`ResidualTrendInput` carrying recent residual history.
        time_s:
            Time stamp (s).
        dt_s:
            Mission dt in seconds.
        rpm, cht_c, egt_c, oil_pressure_bar, oil_temperature_c, fuel_flow_lph, vibration_rms_g:
            Telemetry inputs used to derive physics degradation.
        mission_phase:
            Current mission phase (e.g. Takeoff, Climb, Cruise, etc.).
        engine_age_hours:
            Total accrued engine operating hours.
        cumulative_cycles:
            Total accrued thermal/mission cycles.
        """
        # Resolve the timestamp.
        ts: float
        if time_s is not None:
            ts = float(time_s)
        elif state is not None:
            ts = float(state.time_s)
        else:
            ts = float(health.time_s)

        # Resolve telemetry and wear values from EngineState if provided.
        if state is not None:
            if rpm is None:
                rpm = float(state.rpm)
            if cht_c is None:
                cht_c = float(state.cht_c)
            if egt_c is None:
                egt_c = float(state.egt_c)
            if oil_pressure_bar is None:
                oil_pressure_bar = float(state.oil_pressure_psi) * 0.06894757
            if oil_temperature_c is None:
                oil_temperature_c = float(state.oil_temperature_c)
            if fuel_flow_lph is None:
                fuel_flow_lph = float(state.fuel_flow_lph)
            if vibration_rms_g is None:
                vibration_rms_g = float(state.vibration_rms_g)
            current_wear = float(state.wear) if state.wear is not None else 0.0
        elif health.wear is not None:
            current_wear = float(health.wear)
        else:
            current_wear = 0.0

        # Nominal telemetry defaults if unmeasured
        if rpm is None:
            rpm = 4800.0
        if cht_c is None:
            cht_c = 120.0
        if egt_c is None:
            egt_c = 780.0
        if oil_pressure_bar is None:
            oil_pressure_bar = 4.0
        if oil_temperature_c is None:
            oil_temperature_c = 95.0
        if fuel_flow_lph is None:
            fuel_flow_lph = 16.0
        if vibration_rms_g is None:
            vibration_rms_g = 0.65

        # Total accrued engine hours
        if engine_age_hours is None:
            engine_age_hours = current_wear * float(self._cfg.degradation.max_wear) * 2000.0 + self._hours_running

        # Fault severity from health index
        fault_severity = sum(
            float(prob) for prob in (health.contributing_faults or {}).values()
        )

        # Advance mission clock
        self._hours_running += float(dt_s) / 3600.0

        # Evaluate authoritative Hybrid RUL model
        hybrid_res = self._hybrid_model.evaluate(
            rpm=rpm,
            cht_c=cht_c,
            egt_c=egt_c,
            oil_pressure_bar=oil_pressure_bar,
            oil_temp_c=oil_temperature_c,
            fuel_flow_lph=fuel_flow_lph,
            vibration_rms_g=vibration_rms_g,
            mission_phase=mission_phase,
            engine_age_hours=engine_age_hours,
            cumulative_cycles=cumulative_cycles,
            fault_severity=fault_severity,
            wear=current_wear,
            health_score=health.overall_score,
        )

        # Compute wear rate (prefer trained model over closed form if explicit model loaded)
        if self._trained is not None:
            wear_rate = self._trained.rate_per_hour(
                health, hours_running=self._hours_running,
                residual_trend=residual_trend,
            )
            model_status = RulModelStatus.MODEL_CALIBRATED
        else:
            wear_rate = self._wear_model.rate_per_hour(health)
            model_status = RulModelStatus.CLOSED_FORM
        wear_rate = max(0.0001, float(wear_rate))

        # Run Monte-Carlo simulation with calibrated Rotax 914 horizon
        tte_dist = simulate_tte(
            wear_model=self._wear_model if self._trained is None else _TrainedAdapter(self._trained),
            current_health=health,
            current_wear=current_wear,
            max_wear=float(self._cfg.degradation.max_wear),
            horizon_hours=self._horizon_hours,
            n_samples=self._n_samples,
            dt_s=DEFAULT_DT_S,
            quantile_low=DEFAULT_QUANTILE_LOW,
            quantile_high=DEFAULT_QUANTILE_HIGH,
            noise_std=DEFAULT_NOISE_STD,
            fault_drift_per_hour=DEFAULT_FAULT_DRIFT_PER_HOUR,
            seed=self._seed + self._update_count,
        )

        if self._trained is not None:
            rem_hours = float(tte_dist.central)
            lo_bound = float(tte_dist.lower)
            hi_bound = float(tte_dist.upper)
            conf = float(tte_dist.confidence)
        else:
            rem_hours = float(hybrid_res["remaining_hours"])
            lo_bound = float(hybrid_res["lower_hours"])
            hi_bound = float(hybrid_res["upper_hours"])
            conf = float(hybrid_res["confidence"])

        tte = TteDistribution(
            samples=tte_dist.samples,
            central=rem_hours,
            lower=lo_bound,
            upper=hi_bound,
            confidence=conf,
            mean_wear_rate=wear_rate,
        )

        est = aggregate(
            time_s=ts,
            health=health,
            tte=tte,
            wear_rate_per_hour=wear_rate,
            model_status=model_status,
            trend_history=self._trend,
            remaining_hours=rem_hours,
            remaining_cycles=hybrid_res["remaining_cycles"],
            health_index=hybrid_res["health_index"],
            uncertainty_hours=hybrid_res["uncertainty_hours"],
        )
        update_trend_history(self._trend, est.tte_hours_central)
        self._last = est
        self._update_count += 1
        return est

    def measure_latency(self, n_iter: int = 50) -> float:
        """Return average microseconds per :meth:`update` call.

        Useful for tests that need a latency budget.
        """
        t0 = _time.perf_counter()
        for _ in range(int(n_iter)):
            self.update(health=_MINIMAL_HEALTH)
        elapsed = _time.perf_counter() - t0
        return 1e6 * elapsed / float(n_iter)


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
class _TrainedAdapter:
    """Wrap a :class:`TrainedWearModel` in the WearRateModel protocol.

    The MC layer expects a ``rate_per_hour(health, hours_running)``
    method on the wear model. The trained model has the same
    signature, but the MC layer's type hint is the
    :class:`WearRateModel` protocol, so this adapter keeps mypy
    quiet without changing behaviour.
    """

    def __init__(self, model: TrainedWearModel) -> None:
        self._model = model

    def rate_per_hour(self, health: HealthIndex, *, hours_running: float = 0.0) -> float:
        return self._model.rate_per_hour(health, hours_running=hours_running)


# A minimal health index used by ``measure_latency`` so the test
# never needs to construct a real one.
from backend.health import HealthLabel, HealthTrend

_MINIMAL_HEALTH = HealthIndex(
    time_s=0.0,
    overall_score=1.0,
    overall_label=HealthLabel.HEALTHY,
    confidence=1.0,
    trend=HealthTrend.STABLE,
)


__all__ = [
    "DEFAULT_MIN_CONFIDENCE_FOR_OK",
    "RulCalculator",
]
