"""
Fault classifier — inference surface (PHASE 9).

The :class:`FaultClassifier` is what the rest of the system calls
once per window. It owns a :class:`FeatureExtractor` and a
:class:`~backend.ml.window.WindowBuffer`; callers feed it new
``(residual, sample, env)`` ticks and call :meth:`classify` to
get a :class:`FaultClassification`.

Until a trained model is loaded, the classifier returns
``CalibrationStatus.MODEL_NOT_CALIBRATED`` with a default
``HEALTHY`` label and zero confidence. This is the project-spec'd
"no fake outputs" behaviour.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.faults import FaultClass

from .features import FeatureExtractor
from .persistence import load_model
from .trainer import TrainedModel
from .types import CalibrationStatus, FEATURE_NAMES, FaultClassification
from .window import WindowBuffer, WindowTick


class FaultClassifier:
    """Per-window fault classifier."""

    def __init__(
        self,
        window_size: int = 50,
        model: Optional[TrainedModel] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        self._window = WindowBuffer(size=window_size)
        self._extractor = FeatureExtractor()
        self._model: Optional[TrainedModel] = model
        if model_path is not None:
            self._model = load_model(Path(model_path))
        self._last: Optional[FaultClassification] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def window(self) -> WindowBuffer:
        return self._window

    @property
    def status(self) -> CalibrationStatus:
        if self._model is None:
            return CalibrationStatus.MODEL_NOT_CALIBRATED
        return CalibrationStatus.CALIBRATED

    @property
    def is_calibrated(self) -> bool:
        return self.status is not CalibrationStatus.MODEL_NOT_CALIBRATED

    @property
    def last_classification(self) -> Optional[FaultClassification]:
        return self._last

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        self._model = load_model(Path(path))

    def set_model(self, model: TrainedModel) -> None:
        self._model = model

    def reset(self) -> None:
        self._window.reset()
        self._last = None

    # ------------------------------------------------------------------
    # Per-window inference
    # ------------------------------------------------------------------
    def classify(self) -> FaultClassification:
        """Classify the current window and return a :class:`FaultClassification`.

        If the window is empty, returns a default with
        ``status=MODEL_NOT_CALIBRATED`` (or whatever the current
        status is) and ``fault_class=HEALTHY`` with zero
        confidence.
        """
        time_s = float(self._window.ticks()[-1].time_s) if self._window else 0.0

        if self._model is None:
            self._last = FaultClassification(
                time_s=time_s,
                fault_class=FaultClass.HEALTHY,
                confidence=0.0,
                probabilities={FaultClass.HEALTHY: 0.0},
                status=CalibrationStatus.MODEL_NOT_CALIBRATED,
                features_used=FEATURE_NAMES.__len__(),
                notes=["no model loaded; defaulting to HEALTHY with 0 confidence"],
            )
            return self._last

        if len(self._window) == 0:
            self._last = FaultClassification(
                time_s=time_s,
                fault_class=FaultClass.HEALTHY,
                confidence=0.0,
                probabilities={FaultClass.HEALTHY: 0.0},
                status=CalibrationStatus.MODEL_NOT_CALIBRATED,
                features_used=FEATURE_NAMES.__len__(),
                notes=["empty window; defaulting to HEALTHY with 0 confidence"],
            )
            return self._last

        x = self._extractor.extract(self._window).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        # The sklearn model stores the *integer* label list internally
        # in `model.classes_`. We map each integer back to a FaultClass
        # via the TrainedModel.classes tuple (which is in the same order
        # as the labels we passed to fit()).
        model_classes = list(self._model.model.classes_)
        prob_index: dict = {}  # FaultClass -> probability
        for i, label in enumerate(model_classes):
            fc = self._model.classes[int(label)]
            prob_index[fc] = float(proba[i])
        best_fc = max(prob_index.items(), key=lambda kv: kv[1])[0]
        best_p = prob_index[best_fc]
        probs = prob_index

        self._last = FaultClassification(
            time_s=time_s,
            fault_class=best_fc,
            confidence=best_p,
            probabilities=probs,
            status=CalibrationStatus.CALIBRATED,
            features_used=int(x.shape[1]),
            notes=[],
        )
        return self._last

    # ------------------------------------------------------------------
    # Convenience: latency
    # ------------------------------------------------------------------
    def measure_latency(self, n_iter: int = 100) -> float:
        """Return average microseconds per :meth:`classify` call.

        Useful for tests that need a latency budget.
        """
        t0 = _time.perf_counter()
        for _ in range(int(n_iter)):
            self.classify()
        elapsed = _time.perf_counter() - t0
        return 1e6 * elapsed / float(n_iter)


__all__ = ["FaultClassifier"]
