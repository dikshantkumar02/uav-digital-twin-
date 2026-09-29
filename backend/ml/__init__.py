"""
ML package — fault classification (PHASE 9).

Public re-exports::

    from backend.ml import (
        FaultClassifier, FaultClassification, CalibrationStatus,
        FeatureExtractor, WindowBuffer, WindowTick,
        build_dataset, train_classifier, TrainedModel,
        scenario_stratified_split, ScenarioStratifiedSplit,
        compute_metrics, EvalReport, PerClassMetrics,
        save_model, load_model, FEATURE_NAMES, FEATURE_DIM,
    )
"""

# IMPORTANT — import order matters. ``backend.diagnostics`` depends
# on :data:`CalibrationStatus` at *runtime* (not just for type
# annotations), so we must make :mod:`backend.ml.types` available
# before any module that transitively pulls in
# :mod:`backend.diagnostics` (notably :mod:`backend.ml.features`).
# Importing :mod:`.types` first keeps the public symbol ready for
# downstream consumers without triggering the
# ``diagnostics → ml → diagnostics`` cycle.
from .types import (
    FEATURE_DIM,
    FEATURE_NAMES,
    CalibrationStatus,
    FaultClassification,
)
from .classifier import FaultClassifier
from .dataset import CLASSES, DatasetResult, build_dataset
from .evaluation import EvalReport, PerClassMetrics, compute_metrics
from .features import FeatureExtractor
from .persistence import load_model, save_model
from .trainer import (
    MODEL_VERSION,
    ScenarioSplit,
    ScenarioStratifiedSplit,
    TrainedModel,
    scenario_stratified_split,
    train_classifier,
)
from .window import WindowBuffer, WindowTick

__all__ = [
    "CLASSES",
    "CalibrationStatus",
    "DatasetResult",
    "EvalReport",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "FaultClassification",
    "FaultClassifier",
    "MODEL_VERSION",
    "PerClassMetrics",
    "ScenarioSplit",
    "ScenarioStratifiedSplit",
    "TrainedModel",
    "WindowBuffer",
    "WindowTick",
    "build_dataset",
    "compute_metrics",
    "load_model",
    "save_model",
    "scenario_stratified_split",
    "train_classifier",
]

