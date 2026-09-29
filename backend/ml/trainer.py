"""
Classifier training (PHASE 9).

Fits a Random Forest (and optionally a Gradient Boosting
classifier) on a :class:`~backend.ml.dataset.DatasetResult` and
returns a serialisable :class:`TrainedModel` artefact.

The training is **deterministic** given the input data + a fixed
seed. The Random Forest is the default because the spec calls for
"interpretable baselines first"; Gradient Boosting is exposed as
an option but is not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .dataset import CLASSES, DatasetResult
from .types import FEATURE_DIM, FEATURE_NAMES
from .evaluation import EvalReport, compute_metrics


# Model version. Bumped when the feature schema, the default
# split policy, or the evaluation contract changes in a way that
# invalidates a previously-saved artefact. Persisted to a sidecar
# ``.version`` file by :func:`backend.ml.persistence.save_model`.
MODEL_VERSION: str = "phase9-ml-1.0.0"


# ---------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------
@dataclass
class ScenarioSplit:
    """One split of a scenario-stratified dataset.

    Each scenario's rows all land in the same split (no leakage).
    """

    X: np.ndarray
    y: np.ndarray
    scenario_ids: np.ndarray
    name: str                       # "train" | "val" | "test"


@dataclass
class ScenarioStratifiedSplit:
    """Three-way scenario-level split of a dataset."""

    train: ScenarioSplit
    val: ScenarioSplit
    test: ScenarioSplit
    scenario_to_split: np.ndarray   # size n_scenarios; 0/1/2 = train/val/test

    @property
    def n_scenarios(self) -> int:
        return int(self.scenario_to_split.shape[0])

    def overlap(self) -> List[int]:
        """Return any scenario id that appears in more than one split.
        Should be empty in a valid split."""
        bad: List[int] = []
        for sid in range(self.n_scenarios):
            in_train = int(np.sum(self.train.scenario_ids == sid)) > 0
            in_val = int(np.sum(self.val.scenario_ids == sid)) > 0
            in_test = int(np.sum(self.test.scenario_ids == sid)) > 0
            if (int(in_train) + int(in_val) + int(in_test)) > 1:
                bad.append(int(sid))
        return bad


def scenario_stratified_split(
    dataset: DatasetResult,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> ScenarioStratifiedSplit:
    """Split a dataset by *scenario*, not by sample.

    A scenario is a single (config, fault) run that produced a
    sequence of windows. All windows from a given scenario land in
    the same split, so the model can never see a window from
    scenario N during training and report accuracy on a window
    from the same scenario at test time. This is the
    data-leakage-prevention contract the project spec mandates.

    Parameters
    ----------
    dataset
        Output of :func:`~backend.ml.dataset.build_dataset`.
    train_frac, val_frac, test_frac
        Target fractions of *scenarios* (not rows) per split.
        ``train_frac + val_frac + test_frac`` should be ~1.0;
        any remainder is added to train.
    seed
        Deterministic seed for the scenario-id shuffle.

    Returns
    -------
    A :class:`ScenarioStratifiedSplit` with disjoint splits.
    """
    if dataset.n_rows == 0:
        empty = ScenarioSplit(
            X=np.zeros((0, FEATURE_DIM), dtype=np.float64),
            y=np.zeros((0,), dtype=np.int64),
            scenario_ids=np.zeros((0,), dtype=np.int64),
            name="",
        )
        return ScenarioStratifiedSplit(
            train=ScenarioSplit(**{**empty.__dict__, "name": "train"}),
            val=ScenarioSplit(**{**empty.__dict__, "name": "val"}),
            test=ScenarioSplit(**{**empty.__dict__, "name": "test"}),
            scenario_to_split=np.zeros((0,), dtype=np.int64),
        )
    if dataset.n_scenarios < 3:
        # Not enough scenarios to three-way split. Fall back to
        # everything-in-train and emit empty val/test; caller
        # can check ``train.scenario_ids.shape[0]``.
        train_idx = np.arange(dataset.n_rows)
        sid_map = np.zeros((dataset.n_scenarios,), dtype=np.int64)
        return ScenarioStratifiedSplit(
            train=_slice(dataset, train_idx, "train"),
            val=_slice(dataset, np.zeros((0,), dtype=np.int64), "val"),
            test=_slice(dataset, np.zeros((0,), dtype=np.int64), "test"),
            scenario_to_split=sid_map,
        )

    # Normalise fractions so they sum to 1.0.
    total = float(train_frac + val_frac + test_frac)
    if total <= 0:
        raise ValueError("at least one split fraction must be > 0")
    train_frac /= total
    val_frac /= total
    test_frac /= total

    rng = np.random.default_rng(int(seed))
    n = int(dataset.n_scenarios)
    sids = np.arange(n)
    rng.shuffle(sids)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    # Cap val so the remainder (>=1) lands in test.
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train_sids = sids[:n_train]
    val_sids = sids[n_train:n_train + n_val]
    test_sids = sids[n_train + n_val:]

    sid_map = np.zeros((n,), dtype=np.int64)
    for s in train_sids:
        sid_map[int(s)] = 0
    for s in val_sids:
        sid_map[int(s)] = 1
    for s in test_sids:
        sid_map[int(s)] = 2

    def _row_idx(scenario_set: np.ndarray) -> np.ndarray:
        if len(scenario_set) == 0:
            return np.zeros((0,), dtype=np.int64)
        mask = np.isin(dataset.scenario_ids, scenario_set)
        return np.flatnonzero(mask)

    return ScenarioStratifiedSplit(
        train=_slice(dataset, _row_idx(train_sids), "train"),
        val=_slice(dataset, _row_idx(val_sids), "val"),
        test=_slice(dataset, _row_idx(test_sids), "test"),
        scenario_to_split=sid_map,
    )


def _slice(dataset: DatasetResult, idx: np.ndarray, name: str) -> ScenarioSplit:
    return ScenarioSplit(
        X=dataset.X[idx] if len(idx) else np.zeros((0, dataset.X.shape[1]), dtype=np.float64),
        y=dataset.y[idx] if len(idx) else np.zeros((0,), dtype=np.int64),
        scenario_ids=dataset.scenario_ids[idx] if len(idx) else np.zeros((0,), dtype=np.int64),
        name=name,
    )


# ---------------------------------------------------------------------
# TrainedModel
# ---------------------------------------------------------------------
@dataclass
class TrainedModel:
    """A trained classifier + metadata."""

    model: object                              # sklearn-compatible classifier
    classes: tuple                             # CLASSES (kept in sync with training)
    feature_names: tuple                       # FEATURE_NAMES
    train_accuracy: float                      # accuracy on the training set
    held_out_accuracy: Optional[float] = None  # accuracy on a held-out split
    n_estimators: int = 100
    model_kind: str = "random_forest"
    seed: int = 0
    model_version: str = MODEL_VERSION          # bumped on breaking changes
    # Per-split evaluation reports. Populated when the split
    # policy provides the necessary data (scenario or sample).
    train_report: Optional["EvalReport"] = None
    val_report: Optional["EvalReport"] = None
    test_report: Optional["EvalReport"] = None
    split_policy: str = "scenario"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def feature_importances(self) -> dict:
        """Return a {feature_name: importance} dict for explainability."""
        if not hasattr(self.model, "feature_importances_"):
            return {}
        importances = np.asarray(self.model.feature_importances_)
        return {name: float(v) for name, v in zip(self.feature_names, importances)}


# ---------------------------------------------------------------------
# Backwards-compatible per-sample split
# ---------------------------------------------------------------------
def _train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A deterministic stratified-ish split (no sklearn dependency).

    Kept for backwards compatibility. New code should use
    :func:`scenario_stratified_split` to avoid data leakage.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(test_frac * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------
def train_classifier(
    dataset: DatasetResult,
    *,
    model_kind: str = "random_forest",
    n_estimators: int = 100,
    seed: int = 0,
    test_frac: float = 0.2,
    split_policy: str = "scenario",
    val_frac: float = 0.15,
) -> TrainedModel:
    """Train a classifier on the dataset and return a :class:`TrainedModel`.

    Parameters
    ----------
    dataset
        Output of :func:`~backend.ml.dataset.build_dataset`.
    model_kind
        ``"random_forest"`` (default) or ``"gradient_boosting"``.
    n_estimators
        Number of trees.
    seed
        Random seed for reproducibility.
    test_frac
        Fraction held out for the reported ``held_out_accuracy``.
        Used only when ``split_policy="sample"`` (legacy per-sample
        split). With ``split_policy="scenario"`` (default) the
        fractions are the *target scenario-level* allocations
        via :func:`scenario_stratified_split`.
    split_policy
        ``"scenario"`` (default, recommended) or ``"sample"``
        (legacy per-sample shuffle; suffers from data leakage).
    val_frac
        Scenario-level validation fraction (only used when
        ``split_policy="scenario"``).
    """
    if dataset.n_rows < 2:
        raise ValueError("dataset too small to train (need at least 2 rows)")

    # Lazy import — keeps the dependency on sklearn contained.
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )

    if split_policy == "scenario":
        splits = scenario_stratified_split(
            dataset,
            train_frac=1.0 - val_frac - 0.15,
            val_frac=val_frac,
            test_frac=0.15,
            seed=seed,
        )
        X_tr, y_tr = splits.train.X, splits.train.y
        X_va, y_va = splits.val.X, splits.val.y
        X_te, y_te = splits.test.X, splits.test.y
    elif split_policy == "sample":
        X_tr, X_te, y_tr, y_te = _train_test_split(
            dataset.X, dataset.y, test_frac, seed,
        )
        X_va = np.zeros((0, X_tr.shape[1]), dtype=np.float64)
        y_va = np.zeros((0,), dtype=np.int64)
    else:
        raise ValueError(f"unknown split_policy: {split_policy!r}")

    if model_kind == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            n_jobs=1,
        )
    elif model_kind == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown model_kind: {model_kind!r}")

    model.fit(X_tr, y_tr)
    train_acc = float((model.predict(X_tr) == y_tr).mean()) if len(X_tr) else 0.0
    held_out_acc = float((model.predict(X_te) == y_te).mean()) if len(X_te) else None

    def _report(X: np.ndarray, y: np.ndarray) -> Optional[EvalReport]:
        if X.shape[0] == 0:
            return None
        y_pred = np.asarray(model.predict(X))
        return compute_metrics(y, y_pred, classes=CLASSES)

    train_report = _report(X_tr, y_tr) if len(X_tr) else None
    val_report = _report(X_va, y_va) if len(X_va) else None
    test_report = _report(X_te, y_te) if len(X_te) else None

    return TrainedModel(
        model=model,
        classes=tuple(CLASSES),
        feature_names=tuple(FEATURE_NAMES),
        train_accuracy=train_acc,
        held_out_accuracy=held_out_acc,
        n_estimators=int(n_estimators),
        model_kind=str(model_kind),
        seed=int(seed),
        model_version=MODEL_VERSION,
        train_report=train_report,
        val_report=val_report,
        test_report=test_report,
        split_policy=str(split_policy),
    )


__all__ = [
    "MODEL_VERSION",
    "ScenarioSplit",
    "ScenarioStratifiedSplit",
    "TrainedModel",
    "scenario_stratified_split",
    "train_classifier",
]
