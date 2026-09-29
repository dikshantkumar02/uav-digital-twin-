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

    base_rate_per_hour: float = 1e-5
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


__all__ = [
    "ClosedFormWearRate",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "MODEL_VERSION",
    "ModelVersionWarning",
    "TrainedWearModel",
    "WearRateDataset",
    "build_dataset",
    "health_to_features",
    "load_model",
    "save_model",
    "train_wear_model",
]
