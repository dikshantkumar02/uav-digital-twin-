"""
Evaluation metrics for fault-classification (PHASE 9).

Pure-numpy implementation of accuracy, per-class
precision/recall/F1/support, and a confusion matrix. No
sklearn dependency — we want this to be cheap to import, cheap
to test, and easy to audit. The output is an :class:`EvalReport`
that is also serialisable to ``dict`` for logging / persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .dataset import CLASSES, FaultClass  # noqa: F401 — re-exported


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision / recall / F1 / support for a single class."""

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "support": int(self.support),
        }


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics for one (y_true, y_pred) pair.

    Attributes
    ----------
    accuracy
        Fraction of correct predictions.
    macro_f1
        Unweighted mean of per-class F1.
    weighted_f1
        Support-weighted mean of per-class F1.
    per_class
        Dict mapping class index → :class:`PerClassMetrics`.
    confusion_matrix
        ``(n_classes, n_classes)`` array, rows = true, cols = pred.
    class_names
        Tuple of human-readable class names (length ``n_classes``).
    n_samples
        Number of samples evaluated.
    unknown_count
        Number of samples whose true label fell outside ``classes``
        (or predicted label did, if ``unknown_count_pred`` is also
        reported). Defaults to 0.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[int, PerClassMetrics]
    confusion_matrix: np.ndarray
    class_names: tuple
    n_samples: int
    unknown_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "accuracy": float(self.accuracy),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "n_samples": int(self.n_samples),
            "unknown_count": int(self.unknown_count),
            "class_names": list(self.class_names),
            "per_class": {str(k): v.to_dict() for k, v in self.per_class.items()},
            "confusion_matrix": np.asarray(self.confusion_matrix).tolist(),
        }
        return d


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Optional[Sequence[Any]] = None,
) -> EvalReport:
    """Compute accuracy, per-class metrics, and a confusion matrix.

    Parameters
    ----------
    y_true, y_pred
        Integer label arrays of equal length. Labels are interpreted
        as indices into ``classes``.
    classes
        Sequence of class objects (e.g. ``CLASSES``). If ``None``,
        falls back to :data:`~backend.ml.dataset.CLASSES`.

    Returns
    -------
    :class:`EvalReport`

    Notes
    -----
    Any label that lies outside ``range(n_classes)`` is counted in
    ``unknown_count`` and ignored in the per-class metrics. This
    mirrors the production classifier's policy of returning
    :class:`~backend.ml.types.CalibrationStatus.MODEL_NOT_CALIBRATED`
    when there is no usable output.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if classes is None:
        classes = CLASSES
    n_classes = len(classes)
    class_names = tuple(getattr(c, "value", str(c)) for c in classes)

    n = int(y_true.shape[0])
    if n == 0:
        empty_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        return EvalReport(
            accuracy=0.0, macro_f1=0.0, weighted_f1=0.0,
            per_class={i: PerClassMetrics(0.0, 0.0, 0.0, 0) for i in range(n_classes)},
            confusion_matrix=empty_cm,
            class_names=class_names, n_samples=0, unknown_count=0,
        )

    # Mask out anything outside the class index range. We count
    # them as "unknown" — they're not part of any class.
    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    unknown_count = int((~valid).sum())
    yt = y_true[valid]
    yp = y_pred[valid]

    # Confusion matrix: rows = true, cols = pred.
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(yt, yp):
        cm[int(t), int(p)] += 1

    # Per-class precision/recall/F1/support.
    per_class: Dict[int, PerClassMetrics] = {}
    f1_values: List[float] = []
    weighted_f1_num = 0.0
    total_support = 0
    for c in range(n_classes):
        support = int(cm[c].sum())
        predicted_as_c = int(cm[:, c].sum())
        tp = int(cm[c, c])
        precision = _safe_div(tp, predicted_as_c)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class[c] = PerClassMetrics(precision, recall, f1, support)
        f1_values.append(f1)
        weighted_f1_num += f1 * support
        total_support += support

    correct = int(np.trace(cm))
    accuracy = _safe_div(correct, max(yt.shape[0], 1))
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = _safe_div(weighted_f1_num, total_support)

    return EvalReport(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
        n_samples=int(yt.shape[0]),
        unknown_count=unknown_count,
    )


__all__ = [
    "EvalReport",
    "PerClassMetrics",
    "compute_metrics",
]
