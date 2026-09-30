"""
Wear-rate model (PHASE 11).

The RUL estimator's job is to project the current engine wear
state forward to end-of-life. The first step is a **current
wear-rate** (hours⁻¹) — how fast is the engine wearing right now,
given the health signals.

Two wear-rate models are exposed:

* :class:`ClosedFormWearRate` — a pure function. ``rate = base
  * (1 + k1 * (1 - health) + k2 * fault_severity_sum + k3 *
  trend_penalty)``. Used when no trained model is loaded; the
  whole RUL pipeline is still functional, but the estimate is a
  transparent projection rather than a learned correction.

* :class:`RandomForestWearRate` — a small ``sklearn`` Random
  Forest regressor. Trained on synthetic fault scenarios; the
  dataset builder runs many scenarios through the simulator,
  collects (health, fault, time) → ``d(wear)/dt`` pairs, and the
  RF learns the mapping.

Both models expose a uniform :meth:`rate_per_hour` method so the
Monte Carlo layer can consume them interchangeably.

The feature vector (see :data:`FEATURE_NAMES`) was extended in
``phase11-rul-1.1.0`` to include five residual-trend columns
derived from :class:`~backend.rul.types.ResidualTrendInput`
(Digital Twin residual trends). See :data:`MODEL_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from backend.config import EngineConfig, load_config
from backend.faults import FaultClass, FaultScenario
from backend.health import HealthIndex, HealthLabel, HealthTrend, Subsystem, SubsystemHealth

from .types import ResidualTrendInput


# Model version. Bumped when the feature schema changes in a
# way that invalidates a previously-saved artefact. Persisted
# as a sidecar ``.version`` file by :func:`save_model`.
MODEL_VERSION: str = "phase11-rul-1.1.0"


# ---------------------------------------------------------------------
# Feature columns (single source of truth)
# ---------------------------------------------------------------------
# Index 0–13: the original 14 health/fault/hour features.
# Index 14–18: residual-trend features (added in 1.1.0).
FEATURE_NAMES: List[str] = [
    "health.overall_score",
    "health.trend",                      # encoded 0..3
    "health.wear",                       # optional
    "health.subsystem.thermal",
    "health.subsystem.lubrication",
    "health.subsystem.performance",
    "health.subsystem.mechanical",
    "health.subsystem.sensors",
    "fault.ENGINE_DEGRADATION",
    "fault.OVERHEATING",
    "fault.LUBRICATION_PRESSURE_ANOMALY",
    "fault.VIBRATION_ENGINE_ANOMALY",
    "fault.PERFORMANCE_LOSS",
    "hours_running",                     # mission hours so far
    # Digital Twin residual trend features (added in 1.1.0).
    "residual.max_abs_z.thermal_channels",
    "residual.max_abs_z.vibration_channel",
    "residual.max_abs_z.pressure_channels",
    "residual.max_abs_z.fuel_flow",
    "residual.abs_z_trend_per_hour",
]

FEATURE_DIM: int = len(FEATURE_NAMES)


def _encode_trend(trend: HealthTrend) -> float:
    return {
        HealthTrend.IMPROVING: 0.0,
        HealthTrend.STABLE: 1.0,
        HealthTrend.DEGRADING: 2.0,
        HealthTrend.INSUFFICIENT_DATA: 3.0,
    }.get(trend, 1.0)


# The five fault classes tracked in the feature vector. SENSOR_FAULT
# and ENVIRONMENTAL_DISTURBANCE are excluded — they don't drive
# engine wear directly.
FAULT_FEATURES: Tuple[str, ...] = (
    "ENGINE_DEGRADATION",
    "OVERHEATING",
    "LUBRICATION_PRESSURE_ANOMALY",
    "VIBRATION_ENGINE_ANOMALY",
    "PERFORMANCE_LOSS",
)


# Channel → group mapping for the residual-trend feature columns.
_RESIDUAL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "thermal": ("egt", "cht", "oil_temperature"),
    "vibration": ("vibration",),
    "pressure": ("oil_pressure", "ambient_pressure"),
    "fuel_flow": ("fuel_flow",),
}


def _residual_feature_columns(
    rt: Optional[ResidualTrendInput],
) -> Tuple[float, float, float, float, float]:
    """Compute the 5 residual-trend feature values.

    Returns zeros when ``rt`` is ``None`` — "evidence insufficient"
    is a valid input; the wear-rate model still works.
    """
    if rt is None or rt.n_ticks < 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    chan_to_max = dict(zip(rt.channel_names, rt.per_channel_max_abs_z()))
    chan_to_mean = dict(zip(rt.channel_names, rt.per_channel_mean_abs_z()))

    def _group_max(group: Tuple[str, ...]) -> float:
        vals = [chan_to_max[c] for c in group if c in chan_to_max]
        return float(max(vals)) if vals else 0.0

    def _group_mean(group: Tuple[str, ...]) -> float:
        vals = [chan_to_mean[c] for c in group if c in chan_to_mean]
        return float(np.mean(vals)) if vals else 0.0

    thermal = _group_max(_RESIDUAL_GROUPS["thermal"])
    vibration = _group_max(_RESIDUAL_GROUPS["vibration"])
    pressure = _group_max(_RESIDUAL_GROUPS["pressure"])
    fuel_flow = _group_max(_RESIDUAL_GROUPS["fuel_flow"])
    # Trend is per-hour; the slope from per_channel_trend is
    # already per-second, so multiply by 3600.
    trend_per_s = float(rt.per_channel_trend().mean())
    return thermal, vibration, pressure, fuel_flow, trend_per_s * 3600.0


def health_to_features(
    health: HealthIndex,
    *,
    hours_running: float = 0.0,
    residual_trend: Optional[ResidualTrendInput] = None,
) -> np.ndarray:
    """Convert a :class:`HealthIndex` into a 1-D feature vector.

    Missing subsystems (e.g. SENSORS not yet evaluated) default to
    0.0 — the RF will learn to treat those as low-confidence
    inputs. Missing fault classes default to 0.0. Missing residual
    trend input also defaults to 0.0.
    """
    sub = health.subsystems or {}
    faults = health.contributing_faults or {}
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[0] = float(health.overall_score)
    feat[1] = _encode_trend(health.trend)
    feat[2] = float(health.wear) if health.wear is not None else 0.0
    feat[3] = float(sub.get(_find("thermal"), _zero_sub()).score)
    feat[4] = float(sub.get(_find("lubrication"), _zero_sub()).score)
    feat[5] = float(sub.get(_find("performance"), _zero_sub()).score)
    feat[6] = float(sub.get(_find("mechanical"), _zero_sub()).score)
    feat[7] = float(sub.get(_find("sensors"), _zero_sub()).score)
    for i, name in enumerate(FAULT_FEATURES):
        feat[8 + i] = float(faults.get(name, 0.0))
    feat[13] = float(hours_running)
    # Residual-trend feature columns (indices 14..18).
    feat[14], feat[15], feat[16], feat[17], feat[18] = _residual_feature_columns(
        residual_trend
    )
    return feat


def _find(name: str):
    """Resolve a subsystem name to a ``Subsystem`` enum value."""
    return Subsystem(name.upper())


def _zero_sub():
    return SubsystemHealth(subsystem=Subsystem.THERMAL, score=0.0,
                           confidence=0.0)


# ---------------------------------------------------------------------
# Closed-form wear rate
# ---------------------------------------------------------------------
@dataclass
class ClosedFormWearRate:
    """Pure-function wear-rate model.

    Parameters
    ----------
    base_rate_per_hour:
        The wear rate of a healthy engine at rated power. Default
        is read from ``EngineConfig.degradation.base_wear_per_hour``.
    k_health:
        Multiplier on ``(1 - health.overall_score)``. Default 2.0.
    k_fault:
        Multiplier on the fault-severity sum. Default 4.0.
    k_trend:
        Multiplier on the trend penalty (0 if STABLE/IMPROVING,
        1 if DEGRADING). Default 0.5.
    max_wear:
        Engine wear cap (from ``EngineConfig.degradation.max_wear``).
    """

    base_rate_per_hour: float = 0.0005 # Calibrated for Rotax 914 2000h TBO
    k_health: float = 2.0
    k_fault: float = 4.0
    k_trend: float = 0.5
    max_wear: float = 1.0

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "ClosedFormWearRate":
        d = cfg.degradation
        return cls(
            base_rate_per_hour=float(d.base_wear_per_hour),
            max_wear=float(d.max_wear),
        )

    def rate_per_hour(
        self,
        health: HealthIndex,
        *,
        hours_running: float = 0.0,
    ) -> float:
        """Compute the current wear rate (hours⁻¹) for the given health.

        The formula is::

            rate = base
                 * (1 + k_health * (1 - overall_score))
                 * (1 + k_fault * sum(severity * prob for fault, prob
                                     in contributing_faults))
                 * (1 + k_trend * (1 if trend == DEGRADING else 0))
                 * (1 + wear)                # wear accelerates wear

        All terms are non-negative, so the rate is at least
        ``base_rate_per_hour``. The result is in wear units per
        hour (where 1.0 = end-of-life).
        """
        # Health penalty: 0 when fully healthy, k_health when score 0.
        health_term = self.k_health * (1.0 - float(health.overall_score))
        # Fault penalty: weighted sum of contributing-fault probabilities.
        fault_severity = sum(
            float(prob) for prob in (health.contributing_faults or {}).values()
        )
        fault_term = self.k_fault * fault_severity
        # Trend penalty: 0 unless trend is DEGRADING.
        trend_term = self.k_trend if health.trend is HealthTrend.DEGRADING else 0.0
        # Wear self-acceleration: the more worn, the faster it wears.
        wear_accel = float(health.wear) if health.wear is not None else 0.0
        rate = self.base_rate_per_hour * (
            (1.0 + health_term)
            * (1.0 + fault_term)
            * (1.0 + trend_term)
            * (1.0 + wear_accel)
        )
        return max(0.0, float(rate))


# ---------------------------------------------------------------------
# Rotax 914 Limits & Hybrid RUL Model (PHASE 11 / Master Prompt)
# ---------------------------------------------------------------------
def load_engine_limits(config_path: Optional[Path | str] = None) -> dict:
    """Load engine operational limits from config/engine_limits.yaml."""
    import yaml
    if config_path is None:
        p = Path(__file__).resolve().parent.parent.parent / "config" / "engine_limits.yaml"
    else:
        p = Path(config_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


@dataclass(frozen=True)
class PhysicsDegradationResult:
    """Breakdown of physical degradation components."""
    thermal_damage: float           # [0, 1] from CHT and EGT stress
    oil_degradation: float          # [0, 1] from Oil Temp and Pressure deficits
    vibration_fatigue: float        # [0, 1] from Vibration RMS
    combustion_efficiency: float    # [0, 1] combustion delivery
    mechanical_wear: float          # [0, 1] accrued age and cycle wear
    health_index: float             # [0, 1] combined condition score
    physics_rul_hours: float        # Projected physics remaining hours


class PhysicsDegradationModel:
    """Physics-based degradation model calibrated to Rotax 914 operational limits."""

    def __init__(self, limits_config: Optional[dict] = None) -> None:
        self.cfg = limits_config or load_engine_limits()
        lifecycle = self.cfg.get("engine_lifecycle", {})
        self.nominal_overhaul_hours = float(lifecycle.get("nominal_overhaul_hours", 2000.0))
        self.max_engine_life_hours = float(lifecycle.get("max_engine_life_hours", 2000.0))

        limits = self.cfg.get("operational_limits", {})
        self.cht_cont = float(limits.get("cht_c", {}).get("nominal_continuous", 120.0))
        self.egt_cont = float(limits.get("egt_c", {}).get("nominal_continuous", 780.0))
        self.oil_temp_cont = float(limits.get("oil_temp_c", {}).get("nominal_min", 95.0))
        self.oil_p_nom = float(limits.get("oil_pressure_bar", {}).get("nominal_cruise", 4.0))
        self.vib_crit = float(limits.get("vibration_rms_g", {}).get("critical", 1.8))

        phys_cfg = self.cfg.get("physics_degradation", {})
        self.w_therm = float(phys_cfg.get("thermal_damage_weight", 0.25))
        self.w_oil = float(phys_cfg.get("oil_degradation_weight", 0.25))
        self.w_vib = float(phys_cfg.get("vibration_fatigue_weight", 0.25))
        self.w_comb = float(phys_cfg.get("combustion_efficiency_weight", 0.25))
        self.w_mech = float(phys_cfg.get("mechanical_wear_weight", 0.35))

        self.cht_sens = float(phys_cfg.get("cht_damage_sensitivity", 0.007))
        self.egt_sens = float(phys_cfg.get("egt_damage_sensitivity", 0.003))
        self.oil_temp_sens = float(phys_cfg.get("oil_temp_damage_sensitivity", 0.008))
        self.oil_p_sens = float(phys_cfg.get("oil_pressure_deficit_sensitivity", 0.25))
        self.vib_sens = float(phys_cfg.get("vibration_damage_sensitivity", 0.50))

    def evaluate(
        self,
        *,
        rpm: float = 4800.0,
        cht_c: float = 120.0,
        egt_c: float = 780.0,
        oil_pressure_bar: float = 4.0,
        oil_temp_c: float = 95.0,
        fuel_flow_lph: float = 16.0,
        vibration_rms_g: float = 0.65,
        engine_age_hours: float = 0.0,
        cumulative_cycles: Optional[int] = None,
        fault_severity: float = 0.0,
        wear: Optional[float] = None,
        health_score: Optional[float] = None,
    ) -> PhysicsDegradationResult:
        # 1. Thermal damage (excess temperature above continuous envelope)
        cht_excess = max(0.0, float(cht_c) - self.cht_cont)
        egt_excess = max(0.0, float(egt_c) - self.egt_cont)
        thermal_damage = float(np.clip(
            cht_excess * self.cht_sens + egt_excess * self.egt_sens + fault_severity * 0.25,
            0.0, 0.95,
        ))

        # 2. Oil degradation (high temperature thinning + low pressure boundary lubrication)
        oil_t_excess = max(0.0, float(oil_temp_c) - self.oil_temp_cont)
        oil_p_deficit = max(0.0, self.oil_p_nom - float(oil_pressure_bar))
        oil_degradation = float(np.clip(
            oil_t_excess * self.oil_temp_sens + oil_p_deficit * self.oil_p_sens + fault_severity * 0.25,
            0.0, 0.95,
        ))

        # 3. Vibration fatigue
        vibration_fatigue = float(np.clip(
            (float(vibration_rms_g) / max(0.1, self.vib_crit)) * self.vib_sens + fault_severity * 0.30,
            0.0, 0.95,
        ))

        # 4. Combustion efficiency
        combustion_damage = float(np.clip(
            max(0.0, float(fuel_flow_lph) - 24.0) * 0.03 + fault_severity * 0.20,
            0.0, 0.90,
        ))
        combustion_efficiency = float(np.clip(1.0 - combustion_damage, 0.10, 0.99))

        # 5. Mechanical wear
        if wear is not None and wear > 0.0:
            mechanical_wear = float(np.clip(wear, 0.0, 0.99))
        else:
            mechanical_wear = float(np.clip(
                (float(engine_age_hours) / self.nominal_overhaul_hours) ** 1.35,
                0.0, 0.99,
            ))

        # Combined health index in [0.05, 0.99]
        if health_score is not None:
            # When digital twin health assessment is provided, it is the primary health indicator
            base_h = float(health_score)
            if fault_severity > 0.0:
                health_index = float(np.clip(base_h * (1.0 - 0.5 * fault_severity), 0.05, 0.99))
            elif base_h >= 0.99:
                health_index = 1.0
            else:
                health_index = float(np.clip(base_h, 0.05, 0.99))
        else:
            health_index = float(np.clip(
                1.0 - (self.w_therm * thermal_damage +
                       self.w_oil * oil_degradation +
                       self.w_vib * vibration_fatigue +
                       self.w_comb * combustion_damage +
                       self.w_mech * mechanical_wear),
                0.05, 0.99,
            ))

        # Baseline remaining overhaul life
        baseline_rem = max(0.0, self.nominal_overhaul_hours - float(engine_age_hours))
        condition_mult = (health_index ** 1.20) * (1.0 - 0.85 * fault_severity)
        physics_rul = float(np.clip(baseline_rem * condition_mult, 0.0, self.max_engine_life_hours))

        return PhysicsDegradationResult(
            thermal_damage=thermal_damage,
            oil_degradation=oil_degradation,
            vibration_fatigue=vibration_fatigue,
            combustion_efficiency=combustion_efficiency,
            mechanical_wear=mechanical_wear,
            health_index=health_index,
            physics_rul_hours=physics_rul,
        )


class AiTemporalDegradationModel:
    """AI Temporal Degradation Model derived from TITAN / AERO-TWIN telemetry dataset."""

    def __init__(self, limits_config: Optional[dict] = None) -> None:
        self.cfg = limits_config or load_engine_limits()
        # Telemetry normalisation anchors from rotax_912_dataset_summary.json
        self.mean_cht = 126.86
        self.std_cht = 14.9
        self.mean_oil_t = 105.98
        self.std_oil_t = 9.54
        self.mean_oil_p = 3.94
        self.std_oil_p = 0.44
        self.mean_vib = 0.71
        self.std_vib = 0.21

    def predict(
        self,
        *,
        cht_c: float,
        egt_c: float,
        oil_temp_c: float,
        oil_pressure_bar: float,
        vibration_rms_g: float,
        health_index: float,
        baseline_rem_hours: float,
        fault_severity: float = 0.0,
    ) -> float:
        if health_index >= 0.99 and fault_severity == 0.0:
            return float(np.clip(baseline_rem_hours, 0.0, 2000.0))

        # Multivariable telemetry deviation features
        z_cht = (float(cht_c) - self.mean_cht) / max(1.0, self.std_cht)
        z_oil_t = (float(oil_temp_c) - self.mean_oil_t) / max(1.0, self.std_oil_t)
        z_oil_p = (float(oil_pressure_bar) - self.mean_oil_p) / max(0.1, self.std_oil_p)
        z_vib = (float(vibration_rms_g) - self.mean_vib) / max(0.05, self.std_vib)

        # Dataset correlation weights: CHT (-0.175), OilT (-0.192), OilP (+0.096), Vib (-0.143)
        trend_modifier = 1.0 - 0.02 * z_cht - 0.02 * z_oil_t + 0.01 * z_oil_p - 0.02 * z_vib
        trend_modifier = float(np.clip(trend_modifier, 0.93, 1.07))

        condition_factor = (float(health_index) ** 1.20) * (1.0 - 0.85 * fault_severity)
        ai_rul = float(np.clip(baseline_rem_hours * condition_factor * trend_modifier, 0.0, 2000.0))
        return ai_rul


class HybridRulModel:
    """Authoritative Hybrid RUL Model: Remaining Life = Physics Degradation + AI Temporal Degradation.

    Calibrated to Rotax 914-class operational limits and TITAN / AERO-TWIN telemetry dataset.
    """

    def __init__(self, limits_config: Optional[dict] = None) -> None:
        self.limits_config = limits_config or load_engine_limits()
        self.physics = PhysicsDegradationModel(self.limits_config)
        self.ai = AiTemporalDegradationModel(self.limits_config)
        lifecycle = self.limits_config.get("engine_lifecycle", {})
        self.nominal_overhaul_hours = float(lifecycle.get("nominal_overhaul_hours", 2000.0))
        self.max_engine_life = float(lifecycle.get("max_engine_life_hours", 2000.0))

    def evaluate(
        self,
        *,
        rpm: float = 4800.0,
        cht_c: float = 120.0,
        egt_c: float = 780.0,
        oil_pressure_bar: float = 4.0,
        oil_temp_c: float = 95.0,
        fuel_flow_lph: float = 16.0,
        vibration_rms_g: float = 0.65,
        mission_phase: Optional[str] = None,
        engine_age_hours: float = 0.0,
        cumulative_cycles: Optional[int] = None,
        fault_severity: float = 0.0,
        wear: Optional[float] = None,
        health_score: Optional[float] = None,
    ) -> dict:
        # 1. Physics Degradation
        phys = self.physics.evaluate(
            rpm=rpm,
            cht_c=cht_c,
            egt_c=egt_c,
            oil_pressure_bar=oil_pressure_bar,
            oil_temp_c=oil_temp_c,
            fuel_flow_lph=fuel_flow_lph,
            vibration_rms_g=vibration_rms_g,
            engine_age_hours=engine_age_hours,
            cumulative_cycles=cumulative_cycles,
            fault_severity=fault_severity,
            wear=wear,
            health_score=health_score,
        )

        # 2. AI Temporal Degradation
        baseline_rem = max(0.0, self.nominal_overhaul_hours - float(engine_age_hours))
        ai_rul = self.ai.predict(
            cht_c=cht_c,
            egt_c=egt_c,
            oil_temp_c=oil_temp_c,
            oil_pressure_bar=oil_pressure_bar,
            vibration_rms_g=vibration_rms_g,
            health_index=phys.health_index,
            baseline_rem_hours=baseline_rem,
            fault_severity=fault_severity,
        )

        # 3. Hybrid Combination: Remaining Life = Physics Degradation + AI Temporal Degradation
        rem_hours = float(np.clip(
            round(0.50 * phys.physics_rul_hours + 0.50 * ai_rul, 1),
            0.0,
            self.max_engine_life,
        ))

        # 4. Remaining Cycles (calibrated to 241 cycles at 842.6h reference)
        rem_cycles = max(0, int(round(rem_hours * (241.0 / 842.6))))

        # 5. Health Index Percentage
        health_pct = round(phys.health_index * 100.0, 1)

        # 6. Confidence and Uncertainty Bounds
        model_variance = abs(phys.physics_rul_hours - ai_rul)
        base_uncertainty = 28.4 * (rem_hours / 842.6)
        uncertainty = round(float(np.clip(base_uncertainty + 0.05 * model_variance, 4.0, 75.0)), 1)
        lower_bound = round(max(0.0, rem_hours - uncertainty), 1)
        upper_bound = round(min(self.max_engine_life, rem_hours + uncertainty), 1)

        conf = round(float(np.clip(
            0.96 - 0.12 * (1.0 - phys.health_index) - 0.05 * (float(engine_age_hours) / 2000.0) - 0.20 * fault_severity,
            0.45, 0.99,
        )), 2)

        # Wear rate per hour
        total_life = max(1.0, float(engine_age_hours) + rem_hours)
        wear_rate_per_hour = max(0.0001, (1.0 - (rem_hours / self.nominal_overhaul_hours)) / total_life)

        return {
            "remaining_hours": rem_hours,
            "remaining_cycles": rem_cycles,
            "confidence": conf,
            "health_index": health_pct,
            "uncertainty_hours": uncertainty,
            "bounds": (lower_bound, upper_bound),
            "lower_hours": lower_bound,
            "upper_hours": upper_bound,
            "wear_rate_per_hour": wear_rate_per_hour,
            "physics_result": phys,
        }


# ---------------------------------------------------------------------
# Random Forest wear rate
# ---------------------------------------------------------------------
@dataclass
class TrainedWearModel:
    """A trained Random Forest wear-rate regressor."""

    model: RandomForestRegressor
    feature_importances: Dict[str, float]
    train_mae: float
    n_samples: int
    model_version: str = MODEL_VERSION
    n_train_scenarios: int = 0

    def rate_per_hour(
        self,
        health: HealthIndex,
        *,
        hours_running: float = 0.0,
        residual_trend: Optional[ResidualTrendInput] = None,
    ) -> float:
        x = health_to_features(
            health, hours_running=hours_running,
            residual_trend=residual_trend,
        ).reshape(1, -1)
        return max(0.0, float(self.model.predict(x)[0]))


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------
@dataclass
class WearRateDataset:
    """``(X, y)`` arrays for wear-rate training."""

    X: np.ndarray
    y: np.ndarray
    n_rows: int = 0

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"X must be shape (n, {FEATURE_DIM}), got {self.X.shape}"
            )
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"y length ({self.y.shape[0]}) must match X rows "
                f"({self.X.shape[0]})"
            )
        self.n_rows = int(self.X.shape[0])


def build_dataset(
    *,
    config_dir,
    seed: int = 0,
    n_ticks_per_scenario: int = 200,
) -> WearRateDataset:
    """Build a synthetic wear-rate training set.

    The builder uses the **closed-form** wear-rate model as the
    ground truth and synthesises ``HealthIndex`` samples that span
    the full feature space. Each sample's target is the
    closed-form rate plus a small Gaussian noise term. The
    trained Random Forest then learns a correction factor (or, in
    the limit of zero noise, reproduces the closed-form exactly).

    This approach gives a non-degenerate training set without
    having to run many slow simulator scenarios; the *closed-form
    model* is the same one the calculator falls back to when no
    trained model is loaded, so the RF learns its bias, not
    something unrelated.

    As of ``phase11-rul-1.1.0`` the feature vector includes five
    residual-trend columns. We synthesise a corresponding
    ``ResidualTrendInput`` per row so the RF sees non-trivial
    inputs across the whole feature range; without that, the new
    columns would always be 0 and the RF would learn to ignore
    them.
    """
    cfg = load_config(config_dir)
    cf = ClosedFormWearRate.from_config(cfg.engine)
    rng = np.random.default_rng(int(seed))
    rows: List[Tuple[np.ndarray, float]] = []

    fault_classes = (
        FaultClass.HEALTHY,
        FaultClass.ENGINE_DEGRADATION,
        FaultClass.OVERHEATING,
        FaultClass.LUBRICATION_PRESSURE_ANOMALY,
        FaultClass.VIBRATION_ENGINE_ANOMALY,
        FaultClass.PERFORMANCE_LOSS,
    )
    severities = (0.0, 0.3, 0.6)
    # Channels for the synthetic residual-trend input.
    rt_channels = (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "altitude", "airspeed",
        "ambient_temperature", "ambient_pressure",
    )
    n_rt_channels = len(rt_channels)

    def _make_synthetic_residual_trend(
        severity: float,
        n_ticks: int = 10,
    ) -> ResidualTrendInput:
        """Build a synthetic residual history correlated with severity.

        Healthy scenarios (severity 0) get z-scores near 0; fault
        scenarios get positive z-scores on the channels the fault
        is most likely to drive (oil_pressure/oil_temperature for
        lubrication, vibration for vibration, etc.). The trend
        slope scales with severity too.
        """
        base = float(rng.uniform(0.0, 0.4)) + 0.6 * float(severity)
        slope = float(rng.normal(0.0, 0.005)) + 0.02 * float(severity)
        z = np.zeros((n_rt_channels, n_ticks), dtype=np.float64)
        for i in range(n_rt_channels):
            t = np.arange(n_ticks, dtype=np.float64) * 0.1
            z[i, :] = base + slope * t / 0.1 + rng.normal(0.0, 0.1, n_ticks)
        return ResidualTrendInput(
            channel_names=rt_channels, z_history=z, dt_s=0.1,
        )

    n_scenarios = 0
    for sc_idx, sc in enumerate(
        FaultScenario(fc, severity=s, onset_time_s=0.0, duration_s=None,
                      progression="step")
        for fc in fault_classes
        for s in severities
    ):
        n_scenarios += 1
        for _ in range(int(n_ticks_per_scenario)):
            # Random health score in [0.1, 1.0].
            overall_score = float(rng.uniform(0.1, 1.0))
            wear = float(rng.uniform(0.0, 0.6))
            # Trend: mostly STABLE, sometimes DEGRADING.
            trend = (HealthTrend.DEGRADING
                     if rng.random() < 0.30
                     else HealthTrend.STABLE)
            label = (HealthLabel.HEALTHY if overall_score >= 0.80
                     else HealthLabel.DEGRADED if overall_score >= 0.55
                     else HealthLabel.CRITICAL)
            # Subsystem scores follow the overall with per-subsystem jitter.
            sub_scores = {
                Subsystem.THERMAL:     _clip(overall_score + float(rng.normal(0, 0.05))),
                Subsystem.LUBRICATION: _clip(overall_score + float(rng.normal(0, 0.05))),
                Subsystem.PERFORMANCE: _clip(overall_score + float(rng.normal(0, 0.05))),
                Subsystem.MECHANICAL:  _clip(overall_score + float(rng.normal(0, 0.05))),
                Subsystem.SENSORS:     _clip(overall_score + float(rng.normal(0, 0.03))),
            }
            subsystems = {
                name: SubsystemHealth(subsystem=name, score=v, confidence=1.0)
                for name, v in sub_scores.items()
            }
            # Fault contributions: include the scenario's fault with
            # full severity, plus some random noise from other faults.
            contribs: Dict[str, float] = {}
            if sc.fault_class is not FaultClass.HEALTHY:
                contribs[sc.fault_class.value] = float(sc.severity)
            for other in fault_classes:
                if other is FaultClass.HEALTHY or other is sc.fault_class:
                    continue
                if rng.random() < 0.10:
                    contribs[other.value] = float(rng.uniform(0.05, 0.3))
            hours_running = float(rng.uniform(0.0, 200.0))
            h = HealthIndex(
                time_s=float(sc_idx * 100),
                overall_score=overall_score,
                overall_label=label,
                confidence=1.0,
                subsystems=subsystems,
                trend=trend,
                wear=wear,
                contributing_faults=contribs,
            )
            # Synthesise a residual trend correlated with the
            # scenario's severity so the new feature columns
            # are non-degenerate.
            rt = _make_synthetic_residual_trend(float(sc.severity))
            # Target: closed-form rate + small noise.
            target = cf.rate_per_hour(h, hours_running=hours_running)
            target = max(0.0, target * float(rng.normal(1.0, 0.05)))
            feat = health_to_features(
                h, hours_running=hours_running, residual_trend=rt,
            )
            rows.append((feat, float(target)))

    if not rows:
        return WearRateDataset(X=np.zeros((0, FEATURE_DIM), dtype=np.float32),
                               y=np.zeros((0,), dtype=np.float32))
    X = np.stack([r[0] for r in rows], axis=0).astype(np.float32)
    y = np.array([r[1] for r in rows], dtype=np.float32)
    return WearRateDataset(X=X, y=y)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def train_wear_model(
    dataset: WearRateDataset,
    *,
    n_estimators: int = 30,
    seed: int = 0,
    n_train_scenarios: int = 0,
) -> TrainedWearModel:
    """Train a Random Forest regressor on a wear-rate dataset.

    Light wrapper around :class:`RandomForestRegressor` that
    records feature importances, train MAE, and the number of
    training scenarios (used by the model selector).
    """
    if dataset.n_rows == 0:
        raise ValueError("cannot train on an empty dataset")
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=1,
    )
    rf.fit(dataset.X, dataset.y)
    pred = rf.predict(dataset.X)
    mae = float(np.mean(np.abs(pred - dataset.y)))
    importances = {
        name: float(imp)
        for name, imp in zip(FEATURE_NAMES, rf.feature_importances_)
    }
    return TrainedWearModel(
        model=rf,
        feature_importances=importances,
        train_mae=mae,
        n_samples=int(dataset.n_rows),
        model_version=MODEL_VERSION,
        n_train_scenarios=int(n_train_scenarios),
    )


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def save_model(model: TrainedWearModel, path: Path) -> None:
    """Serialize a trained wear model to disk (pickle + sidecar)."""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "model": model.model,
            "feature_importances": model.feature_importances,
            "train_mae": model.train_mae,
            "n_samples": model.n_samples,
            "model_version": model.model_version,
            "n_train_scenarios": model.n_train_scenarios,
        }, f)
    # Sidecar version stamp (best-effort).
    try:
        with open(_sidecar_path(path), "w", encoding="utf-8") as f:
            f.write(str(model.model_version))
    except OSError:
        pass


def _sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".version")


def _current_model_version() -> str:
    return str(MODEL_VERSION)


class ModelVersionWarning(UserWarning):
    """Raised when a loaded model was serialised with a different
    ``model_version`` than the current code expects (or when the
    sidecar version file is missing)."""


def load_model(path: Path) -> TrainedWearModel:
    """Load a serialized wear model from disk.

    Emits :class:`ModelVersionWarning` if the sidecar version
    differs from :data:`MODEL_VERSION` or is missing.
    """
    import pickle
    from warnings import warn
    path = Path(path)
    with open(path, "rb") as f:
        d = pickle.load(f)
    model = TrainedWearModel(
        model=d["model"],
        feature_importances=d["feature_importances"],
        train_mae=float(d["train_mae"]),
        n_samples=int(d["n_samples"]),
        model_version=str(d.get("model_version", "unknown")),
        n_train_scenarios=int(d.get("n_train_scenarios", 0)),
    )
    # Version check.
    current = _current_model_version()
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        warn(
            f"model at {path} has no sidecar version file; "
            f"expected version {current}. The model will be loaded "
            f"but features/semantics may have changed.",
            ModelVersionWarning,
            stacklevel=2,
        )
    else:
        try:
            stamped = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            stamped = ""
        if stamped != current:
            warn(
                f"model at {path} was saved with version "
                f"{stamped!r} but current code expects {current!r}. "
                f"Verify feature schema is still compatible before "
                f"relying on this artefact.",
                ModelVersionWarning,
                stacklevel=2,
            )
    return model


def load_rotax_rul_dataset(
    csv_path: Optional[Path | str] = None,
    split: Optional[str] = None,
) -> WearRateDataset:
    """Load the high-fidelity Rotax 912 RUL dataset into a WearRateDataset.

    Parameters
    ----------
    csv_path:
        Optional path to CSV. If omitted, defaults to data/rotax_912_rul_dataset.csv
        or the requested split ('train', 'val', 'test').
    split:
        Optional split name ('train', 'val', 'test').
    """
    import pandas as pd

    if csv_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        if split in ("train", "val", "test"):
            csv_path = root / "data" / f"rotax_912_rul_{split}.csv"
        else:
            csv_path = root / "data" / "rotax_912_rul_dataset.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Rotax 912 RUL dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    n = len(df)
    X = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    X[:, 0] = df["health_index"].to_numpy(dtype=np.float32)
    trend_deg = ((df["wear_factor"] > 0.5) | (df["anomaly"] == 1)).to_numpy()
    X[:, 1] = np.where(trend_deg, 2.0, 1.0).astype(np.float32)
    X[:, 2] = df["wear_factor"].to_numpy(dtype=np.float32)
    X[:, 3] = df["cooling_efficiency"].to_numpy(dtype=np.float32)
    X[:, 4] = df["lubrication_health"].to_numpy(dtype=np.float32)
    X[:, 5] = df["combustion_index"].to_numpy(dtype=np.float32)
    X[:, 6] = df["vibration_health"].to_numpy(dtype=np.float32)

    sensor_drift = (df["fault_type"] == "Sensor drift").to_numpy()
    X[:, 7] = np.clip(1.0 - sensor_drift * df["fault_severity"].to_numpy(), 0.0, 1.0).astype(np.float32)

    sev = df["fault_severity"].to_numpy(dtype=np.float32)
    ftype = df["fault_type"].to_numpy()
    X[:, 8] = np.where(np.isin(ftype, ["Misfire", "Combustion instability"]), sev, 0.0)
    X[:, 9] = np.where(np.isin(ftype, ["Overheating", "Cooling degradation"]), sev, 0.0)
    X[:, 10] = np.where(ftype == "Lubrication issue", sev, 0.0)
    X[:, 11] = np.where(ftype == "Abnormal vibration", sev, 0.0)
    X[:, 12] = np.where(ftype == "Injector abnormality", sev, 0.0)

    X[:, 13] = (df["engine_age_hours"] % 100.0).to_numpy(dtype=np.float32)

    X[:, 14] = np.abs(df["cht_C"].to_numpy() - 120.0) / 15.0
    X[:, 15] = df["vibration_rms"].to_numpy() / 0.5
    X[:, 16] = np.abs(df["oil_pressure_bar"].to_numpy() - 4.0) / 0.5
    X[:, 17] = np.abs(df["fuel_flow_Lph"].to_numpy() - 16.0) / 5.0
    X[:, 18] = (sev * 0.05).astype(np.float32)

    rul = df["rul_hours"].to_numpy(dtype=np.float32)
    wear = df["wear_factor"].to_numpy(dtype=np.float32)
    y = np.maximum(0.0, (1.0 - wear) / np.maximum(rul, 1.0)).astype(np.float32)

    return WearRateDataset(X=X, y=y)


def load_rotax_telemetry_features(
    csv_path: Optional[Path | str] = None,
    split: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract raw numerical telemetry and condition features for direct RUL regression.

    Returns (X, y_rul_hours, feature_names).
    """
    import pandas as pd

    if csv_path is None:
        root = Path(__file__).resolve().parent.parent.parent
        if split in ("train", "val", "test"):
            csv_path = root / "data" / f"rotax_912_rul_{split}.csv"
        else:
            csv_path = root / "data" / "rotax_912_rul_dataset.csv"
    else:
        csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    feature_cols = [
        "ambient_temperature_C", "altitude_m", "humidity_percent", "air_density",
        "wind_speed_mps", "rpm", "manifold_pressure_kPa", "throttle_percent",
        "cht_C", "egt_C", "oil_temp_C", "oil_pressure_bar", "fuel_flow_Lph",
        "battery_voltage", "alternator_current_A", "vibration_rms",
        "injection_timing_deg", "engine_load_percent", "thermal_efficiency",
        "combustion_index", "lubrication_health", "vibration_health",
        "cooling_efficiency", "health_index", "engine_age_hours",
        "cumulative_cycles", "wear_factor", "carbon_deposit_index",
        "bearing_wear_index", "anomaly", "fault_severity",
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["rul_hours"].to_numpy(dtype=np.float32)
    return X, y, feature_cols


__all__ = [
    "AiTemporalDegradationModel",
    "ClosedFormWearRate",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "HybridRulModel",
    "MODEL_VERSION",
    "ModelVersionWarning",
    "PhysicsDegradationModel",
    "PhysicsDegradationResult",
    "TrainedWearModel",
    "WearRateDataset",
    "build_dataset",
    "health_to_features",
    "load_engine_limits",
    "load_model",
    "load_rotax_rul_dataset",
    "load_rotax_telemetry_features",
    "save_model",
    "train_wear_model",
]
