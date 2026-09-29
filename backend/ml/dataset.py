"""
Synthetic dataset builder (PHASE 9).

Runs the engine simulator + sensor bundle + anomaly detector
across a battery of fault scenarios and emits a
``(X, y)`` dataset of feature vectors and integer labels. This
is the training material for the Random Forest / Gradient
Boosting classifier.

The function is deterministic given a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from backend.config import load_config
from backend.digital_twin import DigitalTwin
from backend.diagnostics import AnomalyDetector
from backend.environment import EnvironmentState
from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.sensors import SensorBundle
from backend.simulation import EngineSimulator

from .features import FeatureExtractor
from .types import FEATURE_DIM, FaultClassification
from .window import WindowBuffer


@dataclass
class DatasetRow:
    """One row of the dataset: a feature vector and its label."""

    label: int                       # index into CLASSES
    fault_class: FaultClass
    features: np.ndarray             # shape (FEATURE_DIM,)
    scenario_id: int = 0             # 0-based index of the originating scenario


# Class set used for training. The order matters — the integer
# label is the index into this tuple.
CLASSES: Tuple[FaultClass, ...] = (
    FaultClass.HEALTHY,
    FaultClass.ENGINE_DEGRADATION,
    FaultClass.OVERHEATING,
    FaultClass.LUBRICATION_PRESSURE_ANOMALY,
    FaultClass.VIBRATION_ENGINE_ANOMALY,
    FaultClass.PERFORMANCE_LOSS,
    FaultClass.ENVIRONMENTAL_DISTURBANCE,
    FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE,
)
# SENSOR_FAULT is excluded from the truth-layer label set because it
# lives in the sensor layer; the model classifies the underlying
# truth cause, not the channel noise mode. A SENSOR_FAULT scenario
# still produces a labeled row in the dataset — we attach it to
# HEALTHY (the truth is healthy; only the sensor is bad). This
# forces the model to NOT classify the noise as a fault.
assert all(isinstance(c, FaultClass) for c in CLASSES)


def _class_index(fc: FaultClass) -> int:
    try:
        return CLASSES.index(fc)
    except ValueError:
        return CLASSES.index(FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE)


# Default scenario battery. One HEALTHY + one of each engine fault
# + a small SENSOR_FAULT mix + an ENVIRONMENTAL_DISTURBANCE.
def _default_scenarios() -> List[FaultScenario]:
    return [
        FaultScenario(FaultClass.HEALTHY, severity=0.0, onset_time_s=0.0,
                       duration_s=60.0, seed=1),
        FaultScenario(FaultClass.ENGINE_DEGRADATION, severity=0.7,
                       onset_time_s=10.0, duration_s=50.0, seed=2),
        FaultScenario(FaultClass.OVERHEATING, severity=0.6,
                       onset_time_s=10.0, duration_s=50.0, seed=3),
        FaultScenario(FaultClass.LUBRICATION_PRESSURE_ANOMALY, severity=0.5,
                       onset_time_s=10.0, duration_s=50.0, seed=4),
        FaultScenario(FaultClass.VIBRATION_ENGINE_ANOMALY, severity=0.6,
                       onset_time_s=10.0, duration_s=50.0, seed=5),
        FaultScenario(FaultClass.PERFORMANCE_LOSS, severity=0.6,
                       onset_time_s=10.0, duration_s=50.0, seed=6),
        FaultScenario(FaultClass.ENVIRONMENTAL_DISTURBANCE, severity=1.0,
                       onset_time_s=10.0, duration_s=50.0, seed=7),
    ]


@dataclass
class DatasetResult:
    """Output of :func:`build_dataset`."""

    X: np.ndarray                    # shape (n, FEATURE_DIM)
    y: np.ndarray                    # shape (n,)
    fault_classes: np.ndarray        # shape (n,) — FaultClass.value per row
    scenario_ids: np.ndarray         # shape (n,) — originating scenario index
    n_per_class: dict                # class name → count
    n_scenarios: int = 0             # how many distinct scenarios produced rows

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])


def build_dataset(
    scenarios: Optional[Iterable[FaultScenario]] = None,
    config_dir: Optional[Path] = None,
    ticks_per_window: int = 50,
    step_per_tick_s: float = 0.1,
    seed: int = 0,
) -> DatasetResult:
    """Run each scenario end-to-end and emit one row per window.

    Parameters
    ----------
    scenarios
        Iterable of :class:`FaultScenario` to run. ``None`` uses the
        default battery.
    config_dir
        Path to the YAML config dir. ``None`` uses the repo default.
    ticks_per_window
        How many ticks each window contains.
    step_per_tick_s
        Time per tick (s). The default 0.1 s matches the simulator.
    seed
        Master seed for the sensor bundle.
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "config"
    cfg = load_config(config_dir)

    if scenarios is None:
        scenarios = _default_scenarios()
    scenarios = list(scenarios)

    extractor = FeatureExtractor()
    rows: List[DatasetRow] = []

    for scenario_idx, scenario in enumerate(scenarios):
        # Build a fresh simulator + sensor bundle for each scenario
        # so RNG state doesn't leak between scenarios.
        sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=step_per_tick_s)
        bundle = SensorBundle(cfg.sensors, sim, master_seed=seed)
        twin = DigitalTwin(cfg.engine, dt_s=step_per_tick_s, closed_loop_gain=0.0)
        detector = AnomalyDetector()
        injector = FaultInjector(scenario)
        injector.attach(sim, bundle)
        window = WindowBuffer(size=ticks_per_window)

        # Run until the scenario's window is exhausted + a buffer fill.
        total = int((scenario.onset_time_s + (scenario.duration_s or 30.0) + 5.0)
                    / step_per_tick_s)
        for _ in range(total):
            tick = injector.tick(sim.runner._t)
            step = sim.step(
                degradation_severity=tick.degradation_severity,
                vibration_external=tick.vibration_external,
            )
            injector.apply_environment(sim.runner._t)
            sample = bundle.tick()
            injector.apply_sensor(sim.runner._t)
            ts_state = twin.step(step.env, observations={
                ch: r.value for ch, r in sample.readings.items() if r.value is not None
            })
            rf = ts_state._make_residual_frame() if hasattr(ts_state, "_make_residual_frame") else None  # noqa: E501
            # Build a ResidualFrame manually: observed = sensor reading,
            # predicted = twin's own predicted value. We use the twin's
            # own predict API to get predictions.
            from backend.digital_twin import ResidualFrame, ChannelResidual
            predicted = twin.predict(step.env)
            rf2 = ResidualFrame(time_s=step.env.time_s)
            for ch, r in sample.readings.items():
                pred_val = getattr(predicted, _TWIN_FIELD.get(ch, ""), None)
                obs_val = r.value
                if obs_val is None or pred_val is None:
                    continue
                # z-score: use the same default sigma the twin uses.
                sigma = _DEFAULT_SIGMA.get(ch, 1.0)
                z = (float(obs_val) - float(pred_val)) / sigma if sigma > 0 else 0.0
                conf = max(0.0, 1.0 - abs(z) / 6.0)
                rf2.residuals[ch] = ChannelResidual(
                    channel=ch, observed=obs_val, predicted=pred_val,
                    residual=obs_val - pred_val, z_score=z,
                    confidence=conf, in_bounds=abs(z) <= 3.0,
                )
            a = detector.detect(rf2, sample)
            window.append(
                time_s=step.env.time_s,
                residual=rf2,
                sample=sample,
                env=step.env,
            )
            if window.is_full():
                # Emit one row per full window.
                feats = extractor.extract(window)
                label = _class_index(scenario.fault_class)
                rows.append(DatasetRow(
                    label=label,
                    fault_class=scenario.fault_class,
                    features=feats,
                    scenario_id=scenario_idx,
                ))
                # Slide the window by half so adjacent windows overlap.
                # We do this by dropping the front half.
                for _ in range(ticks_per_window // 2):
                    if len(window) > 0:
                        window._buf.popleft()

    if not rows:
        # Caller asked for a dataset but produced no rows — surface
        # this as an empty DatasetResult rather than crashing.
        return DatasetResult(
            X=np.zeros((0, FEATURE_DIM), dtype=np.float64),
            y=np.zeros((0,), dtype=np.int64),
            fault_classes=np.array([], dtype=object),
            scenario_ids=np.zeros((0,), dtype=np.int64),
            n_per_class={},
            n_scenarios=0,
        )

    X = np.stack([r.features for r in rows], axis=0)
    y = np.array([r.label for r in rows], dtype=np.int64)
    fault_classes = np.array([r.fault_class for r in rows], dtype=object)
    scenario_ids = np.array([r.scenario_id for r in rows], dtype=np.int64)
    n_scenarios = int(scenario_ids.max()) + 1 if len(scenario_ids) else 0
    n_per_class: dict = {}
    for fc in CLASSES:
        n_per_class[fc.value] = int(np.sum(y == _class_index(fc)))
    return DatasetResult(
        X=X, y=y, fault_classes=fault_classes,
        scenario_ids=scenario_ids, n_per_class=n_per_class,
        n_scenarios=n_scenarios,
    )


# Twin state field name for each sensor channel (mirror of
# CHANNEL_TO_STATE in PHASE 6).
_TWIN_FIELD = {
    "rpm": "rpm",
    "egt": "egt_c",
    "cht": "cht_c",
    "oil_pressure": "oil_pressure_psi",
    "oil_temperature": "oil_temperature_c",
    "fuel_flow": "fuel_flow_lph",
    "vibration": "vibration_rms_g",
    "altitude": "altitude_m",
    "airspeed": "airspeed_mps",
    "ambient_temperature": "ambient_temperature_c",
    "ambient_pressure": "ambient_pressure_pa",
}

# Match DEFAULT_EXPECTED_SIGMA from PHASE 6 (and the audit fix in
# ``backend/digital_twin/model.py`` for ambient_pressure).
# ambient_pressure carries a documented ±0.5% calibration bias
# (~506 Pa) on top of nominal 10 Pa noise, so the z-score sigma
# must absorb that bias to avoid every healthy sample saturating
# z=10. 600 Pa gives ~3σ headroom.
_DEFAULT_SIGMA = {
    "rpm": 30.0,
    "egt": 25.0,
    "cht": 15.0,
    "oil_pressure": 3.0,
    "oil_temperature": 3.0,
    "fuel_flow": 5.0,
    "vibration": 0.5,
    "altitude": 10.0,
    "airspeed": 3.0,
    "ambient_temperature": 2.0,
    "ambient_pressure": 600.0,
}


__all__ = ["CLASSES", "DatasetResult", "DatasetRow", "build_dataset"]
