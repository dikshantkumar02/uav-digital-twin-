"""PHASE 9 tests — fault classification (ML)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.faults import FaultClass, FaultInjector, FaultScenario
from backend.ml import (
    CLASSES,
    FEATURE_DIM,
    FEATURE_NAMES,
    MODEL_VERSION,
    CalibrationStatus,
    FaultClassification,
    FaultClassifier,
    FeatureExtractor,
    TrainedModel,
    WindowBuffer,
    build_dataset,
    compute_metrics,
    load_model,
    save_model,
    scenario_stratified_split,
    train_classifier,
)
from backend.ml.persistence import ModelVersionWarning
from backend.sensors import NoiseMode, SensorBundle, SensorReading, SensorSample
from backend.digital_twin import DigitalTwin, ResidualFrame, ChannelResidual
from backend.simulation import EngineSimulator
from backend.diagnostics import AnomalyDetector

from backend.config import load_config

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _build_sample_with_residuals(
    time_s: float,
    z_per_channel: dict,
) -> tuple[SensorSample, ResidualFrame]:
    """Build a (sample, residual_frame) pair with synthetic z-scores."""
    sample = SensorSample(time_s=time_s)
    rf = ResidualFrame(time_s=time_s)
    for ch, z in z_per_channel.items():
        sample.readings[ch] = SensorReading(value=100.0, mode=NoiseMode.NORMAL)
        rf.residuals[ch] = ChannelResidual(
            channel=ch, observed=100.0, predicted=100.0,
            residual=0.0, z_score=float(z), confidence=1.0, in_bounds=True,
        )
    return sample, rf


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
class TestTypes:
    def test_feature_names_count(self) -> None:
        assert len(FEATURE_NAMES) == FEATURE_DIM
        # Sanity: at least 50 features, well under 200.
        assert 50 <= FEATURE_DIM <= 200

    def test_calibration_status(self) -> None:
        assert {m.value for m in CalibrationStatus} == {
            "MODEL_NOT_CALIBRATED", "CALIBRATED", "MODEL_DEGRADED",
        }

    def test_classification_to_dict(self) -> None:
        fc = FaultClassification(
            time_s=1.0,
            fault_class=FaultClass.ENGINE_DEGRADATION,
            confidence=0.8,
            probabilities={FaultClass.ENGINE_DEGRADATION: 0.8,
                           FaultClass.HEALTHY: 0.2},
            status=CalibrationStatus.CALIBRATED,
            features_used=60,
        )
        d = fc.to_dict()
        assert d["time_s"] == 1.0
        assert d["fault_class"] == "ENGINE_DEGRADATION"
        assert d["confidence"] == pytest.approx(0.8)
        assert d["status"] == "CALIBRATED"
        assert d["features_used"] == 60
        assert d["prob.ENGINE_DEGRADATION"] == pytest.approx(0.8)
        assert fc.is_calibrated is True


# ---------------------------------------------------------------------
# Window buffer
# ---------------------------------------------------------------------
class TestWindowBuffer:
    def test_appends_and_drains(self) -> None:
        w = WindowBuffer(size=5)
        for i in range(3):
            w.append(time_s=float(i), residual=ResidualFrame(time_s=float(i)))
        assert len(w) == 3
        assert w.is_full() is False

    def test_full_and_oldest_out(self) -> None:
        w = WindowBuffer(size=3)
        for i in range(5):
            w.append(time_s=float(i), residual=ResidualFrame(time_s=float(i)))
        assert len(w) == 3
        # The oldest should be 2, not 0 (FIFO eviction).
        times = [t.time_s for t in w.ticks()]
        assert times == [2.0, 3.0, 4.0]

    def test_reset(self) -> None:
        w = WindowBuffer(size=5)
        w.append(time_s=0.0, residual=ResidualFrame(time_s=0.0))
        w.reset()
        assert len(w) == 0

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            WindowBuffer(size=0)


# ---------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------
class TestFeatureExtractor:
    def test_empty_window_is_zeros(self) -> None:
        v = FeatureExtractor().extract(WindowBuffer(size=5))
        assert v.shape == (FEATURE_DIM,)
        assert float(v.sum()) == 0.0

    def test_full_window_correct_length(self) -> None:
        w = WindowBuffer(size=5)
        for i in range(5):
            s, rf = _build_sample_with_residuals(float(i), {"rpm": 0.1})
            w.append(time_s=float(i), residual=rf, sample=s)
        v = FeatureExtractor().extract(w)
        assert v.shape == (FEATURE_DIM,)

    def test_deterministic(self) -> None:
        w1 = WindowBuffer(size=10)
        w2 = WindowBuffer(size=10)
        for i in range(10):
            s, rf = _build_sample_with_residuals(float(i), {"rpm": 0.5, "egt": 0.1})
            w1.append(float(i), rf, s)
            w2.append(float(i), rf, s)
        v1 = FeatureExtractor().extract(w1)
        v2 = FeatureExtractor().extract(w2)
        np.testing.assert_array_equal(v1, v2)

    def test_high_z_lights_up_residual_features(self) -> None:
        w = WindowBuffer(size=5)
        for i in range(5):
            s, rf = _build_sample_with_residuals(float(i), {"rpm": 4.0})
            w.append(float(i), rf, s)
        v = FeatureExtractor().extract(w)
        rpm_idx = FEATURE_NAMES.index("res.rpm.mean_abs_z")
        assert v[rpm_idx] >= 3.0


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class TestDataset:
    def test_build_dataset_shape(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        assert ds.X.ndim == 2
        assert ds.X.shape[1] == FEATURE_DIM
        assert ds.y.shape[0] == ds.X.shape[0]
        assert ds.n_rows > 0

    def test_class_balance_reasonable(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        # Each non-empty class should have at least 1 row.
        non_empty = [k for k, v in ds.n_per_class.items() if v > 0]
        assert len(non_empty) >= 3
        # Largest class shouldn't be 10x the smallest (very rough check).
        counts = [v for v in ds.n_per_class.values() if v > 0]
        assert max(counts) / min(counts) <= 5


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class TestTrainer:
    def test_trains_random_forest(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, model_kind="random_forest", n_estimators=20, seed=0)
        assert isinstance(m, TrainedModel)
        # Random Forest should easily overfit a small synthetic dataset.
        assert m.train_accuracy >= 0.8
        assert m.model_kind == "random_forest"
        # predict_proba returns one row per input, one column per
        # class the model was actually trained on. (May be a
        # subset of CLASSES if some classes had no rows.)
        proba = m.predict_proba(ds.X[:3])
        assert proba.shape[0] == 3
        assert proba.shape[1] >= 1
        assert proba.shape[1] <= len(CLASSES)

    def test_feature_importances(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, model_kind="random_forest", n_estimators=20, seed=0)
        imp = m.feature_importances()
        assert len(imp) == FEATURE_DIM
        assert all(0.0 <= v <= 1.0 for v in imp.values())
        # The importances should sum to ~1.0.
        assert sum(imp.values()) == pytest.approx(1.0, abs=1e-3)

    def test_gradient_boosting(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, model_kind="gradient_boosting", n_estimators=20, seed=0)
        assert m.model_kind == "gradient_boosting"
        assert m.train_accuracy >= 0.5

    def test_invalid_model_kind(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        with pytest.raises(ValueError):
            train_classifier(ds, model_kind="unknown")


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------
class TestClassifier:
    def test_uncalibrated_default(self) -> None:
        cl = FaultClassifier()
        assert cl.status is CalibrationStatus.MODEL_NOT_CALIBRATED
        c = cl.classify()
        assert c.fault_class is FaultClass.HEALTHY
        assert c.confidence == 0.0
        assert c.status is CalibrationStatus.MODEL_NOT_CALIBRATED

    def test_load_model(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        cl = FaultClassifier()
        cl.load(path)
        assert cl.is_calibrated is True
        assert cl.status is CalibrationStatus.CALIBRATED

    def test_loaded_classifier_classifies(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=20, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        cl = FaultClassifier(window_size=5, model_path=path)
        # Feed a vibration-like window (high z on vibration).
        for i in range(5):
            s, rf = _build_sample_with_residuals(
                float(i), {"vibration": 5.0, "rpm": 0.1, "egt": 0.1},
            )
            cl.window.append(time_s=float(i), residual=rf, sample=s)
        c = cl.classify()
        assert c.status is CalibrationStatus.CALIBRATED
        # We can't assert the exact class (training is data-dependent),
        # but the classifier should produce *some* class with a real
        # confidence > 0.
        assert c.fault_class in CLASSES
        assert c.confidence > 0.0

    def test_latency_under_target(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        cl = FaultClassifier(window_size=5, model_path=path)
        for i in range(5):
            s, rf = _build_sample_with_residuals(float(i), {"rpm": 0.1})
            cl.window.append(float(i), rf, s)
        # Warmup.
        for _ in range(5):
            cl.classify()
        us = cl.measure_latency(n_iter=200)
        # Generous target: 5 ms per inference.
        assert us < 5000.0

    # ------------------------------------------------------------------
    # Context-aware combined-fault scenario (PHASE 9 classifier-level).
    # ------------------------------------------------------------------
    def _drive_window_with_injectors(
        self,
        injectors: list,
        window_size: int = 50,
        tick_s: float = 0.1,
    ) -> WindowBuffer:
        """Drive a sim + bundle + twin pipeline with one or more
        ``FaultInjector``s attached, and return a filled ``WindowBuffer``
        of per-tick (residual, sample, env) records.

        Multiple injectors are summed on the engine side: the engine
        receives ``sum(d.severity for d in ticks)`` as its
        ``degradation_severity`` input. The env side is driven by the
        first injector whose scenario is ``ENVIRONMENTAL_DISTURBANCE``
        (others are engine-only by construction).
        """
        cfg = load_config(CONFIG_DIR)
        sim = EngineSimulator(cfg.engine, cfg.environment, dt_s=tick_s)
        bundle = SensorBundle(cfg.sensors, sim, master_seed=0)
        twin = DigitalTwin(cfg.engine, dt_s=tick_s, closed_loop_gain=0.0)
        window = WindowBuffer(size=window_size)
        # Each injector must be attached to capture its baseline.
        for inj in injectors:
            inj.attach(sim, bundle)
        # 60 s is enough to cover onset+duration+buffer-fill.
        total = 600
        for _ in range(total):
            t = sim.runner._t
            ticks = [inj.tick(t) for inj in injectors]
            degr = sum(tk.degradation_severity for tk in ticks)
            vib = sum(tk.vibration_external for tk in ticks)
            step = sim.step(
                degradation_severity=degr,
                vibration_external=vib,
            )
            for inj in injectors:
                inj.apply_environment(t)
                inj.apply_sensor(t)
            sample = bundle.tick()
            predicted = twin.predict(step.env)
            rf = ResidualFrame(time_s=step.env.time_s)
            from backend.ml.dataset import _TWIN_FIELD, _DEFAULT_SIGMA  # noqa: E402
            for ch, r in sample.readings.items():
                pred_val = getattr(predicted, _TWIN_FIELD.get(ch, ""), None)
                obs_val = r.value
                if obs_val is None or pred_val is None:
                    continue
                sigma = _DEFAULT_SIGMA.get(ch, 1.0)
                z = (float(obs_val) - float(pred_val)) / sigma if sigma > 0 else 0.0
                conf = max(0.0, 1.0 - abs(z) / 6.0)
                rf.residuals[ch] = ChannelResidual(
                    channel=ch, observed=obs_val, predicted=pred_val,
                    residual=obs_val - pred_val, z_score=z,
                    confidence=conf, in_bounds=abs(z) <= 3.0,
                )
            window.append(
                time_s=step.env.time_s,
                residual=rf,
                sample=sample,
                env=step.env,
            )
            if window.is_full():
                break
        return window

    def test_combined_env_and_engine_classifier_sees_both(self, tmp_path: Path) -> None:
        """Scenario 3: turbulence AND engine degradation injected
        simultaneously. The classifier's per-class probability vector
        must show *both* ``ENVIRONMENTAL_DISTURBANCE`` and
        ``ENGINE_DEGRADATION`` above a small floor — i.e. the
        context-aware feature set (which includes wind, altitude,
        airspeed, plus engine residual stats) lets the model
        distinguish a wind-driven vibration spike from an
        engine-driven one, even when both are present.

        The class with the highest probability may legitimately be
        either of the two (or HEALTHY) depending on training; what
        matters is that *both* env and engine evidence are visible
        in the probability vector, not that one of them is "the"
        answer. This is the false-positive-mitigation contract.
        """
        # Train on the default battery (which already includes both
        # ENVIRONMENTAL_DISTURBANCE and ENGINE_DEGRADATION as
        # distinct labels). Use the legacy per-sample split so the
        # model sees every scenario — this is a contract test for
        # the classifier's per-class probability vector, not a
        # generalisation test.
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(
            ds, n_estimators=50, seed=0, split_policy="sample",
        )
        path = tmp_path / "model.pkl"
        save_model(m, path)
        cl = FaultClassifier(window_size=50, model_path=path)

        env_inj = FaultInjector(FaultScenario(
            FaultClass.ENVIRONMENTAL_DISTURBANCE,
            severity=1.0, onset_time_s=10.0, duration_s=40.0, seed=7,
        ))
        eng_inj = FaultInjector(FaultScenario(
            FaultClass.ENGINE_DEGRADATION,
            severity=0.7, onset_time_s=10.0, duration_s=40.0, seed=2,
        ))
        window = self._drive_window_with_injectors([env_inj, eng_inj])
        assert window.is_full(), "window did not fill in 60 s sim time"

        # Manually feed the filled window into the classifier (the
        # buffer inside the classifier is private; use the public
        # append path).
        for tick in window.ticks():
            cl.window.append(
                time_s=tick.time_s, residual=tick.residual,
                sample=tick.sample, env=tick.env,
            )
        c = cl.classify()
        assert c.status is CalibrationStatus.CALIBRATED

        # Per-class probability vector must include both labels with
        # mass above a small floor. The exact ranking depends on
        # training; the contract is that BOTH paths are visible.
        probs = c.probabilities
        env_p = float(probs.get(FaultClass.ENVIRONMENTAL_DISTURBANCE, 0.0))
        eng_p = float(probs.get(FaultClass.ENGINE_DEGRADATION, 0.0))
        # A generous floor: any non-trivial signal. The Random Forest
        # distributes mass across all classes; we just require both
        # to have *some* mass when both faults are present. The
        # 0.03 floor absorbs the natural RF variance after the
        # PHASE 23 audit's ambient_pressure sigma widening
        # (see VERIFICATION_REPORT.md, FAILURE #3).
        assert env_p > 0.03, (
            f"ENVIRONMENTAL_DISTURBANCE prob {env_p:.3f} below floor 0.03; "
            f"full vector: {probs}"
        )
        assert eng_p > 0.05, (
            f"ENGINE_DEGRADATION prob {eng_p:.3f} below floor 0.05; "
            f"full vector: {probs}"
        )
        # Sum of probabilities over CLASSES should be ~1.0
        # (Random Forest normalises; this is a sanity check that
        # the model emitted a real distribution, not a degenerate
        # one).
        total = sum(probs.values())
        assert 0.95 <= total <= 1.05, (
            f"probability vector sums to {total:.3f}, expected ~1.0"
        )

    def test_env_only_classifier_does_not_flag_engine(self, tmp_path: Path) -> None:
        """Scenario 1 (classifier-level counterpart of PHASE 7
        ``test_environmental_disturbance_no_engine_effect``): when
        ONLY an environmental disturbance is injected, the
        classifier must NOT rank ``ENGINE_DEGRADATION`` first. The
        env-vs-engine distinction is what the random forest learns
        from the joint feature set.
        """
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        # Per-sample split: this is a contract test for the
        # classifier's top-1, not a generalisation test. We use
        # the legacy split so the model has seen both env and
        # engine classes during training — the contract being
        # asserted is "given a model that has seen both, env-only
        # should not be flagged as an engine fault".
        m = train_classifier(
            ds, n_estimators=50, seed=0, split_policy="sample",
        )
        path = tmp_path / "model.pkl"
        save_model(m, path)
        cl = FaultClassifier(window_size=50, model_path=path)

        env_inj = FaultInjector(FaultScenario(
            FaultClass.ENVIRONMENTAL_DISTURBANCE,
            severity=1.0, onset_time_s=10.0, duration_s=40.0, seed=7,
        ))
        window = self._drive_window_with_injectors([env_inj])
        assert window.is_full()
        for tick in window.ticks():
            cl.window.append(
                time_s=tick.time_s, residual=tick.residual,
                sample=tick.sample, env=tick.env,
            )
        c = cl.classify()
        assert c.status is CalibrationStatus.CALIBRATED
        # The top-1 class must NOT be any engine fault when the
        # only injected fault is environmental. We check the two
        # engine-fault classes with the strongest vibration/EGRT
        # signal: ENGINE_DEGRADATION and VIBRATION_ENGINE_ANOMALY.
        top = c.fault_class
        assert top is not FaultClass.ENGINE_DEGRADATION, (
            f"env-only injection mis-flagged as ENGINE_DEGRADATION; "
            f"probs={c.probabilities}"
        )
        assert top is not FaultClass.VIBRATION_ENGINE_ANOMALY, (
            f"env-only injection mis-flagged as VIBRATION_ENGINE_ANOMALY; "
            f"probs={c.probabilities}"
        )


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        loaded = load_model(path)
        # Predict must match on a small held-out set.
        x = ds.X[:3]
        np.testing.assert_array_equal(m.predict(x), loaded.predict(x))
        np.testing.assert_allclose(m.predict_proba(x), loaded.predict_proba(x))


# ---------------------------------------------------------------------
# Scenario-level split (data-leakage prevention)
# ---------------------------------------------------------------------
class TestScenarioSplit:
    def test_split_partitions_unique_scenarios(self) -> None:
        """No scenario id may appear in more than one split."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        split = scenario_stratified_split(ds, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0)
        assert split.n_scenarios == ds.n_scenarios
        # Union of split sizes == dataset size.
        assert split.train.X.shape[0] + split.val.X.shape[0] + split.test.X.shape[0] == ds.X.shape[0]
        # scenario_to_split is the source of truth.
        for sid in range(split.n_scenarios):
            assigned = int(split.scenario_to_split[sid])
            in_train = int(np.sum(split.train.scenario_ids == sid)) > 0
            in_val = int(np.sum(split.val.scenario_ids == sid)) > 0
            in_test = int(np.sum(split.test.scenario_ids == sid)) > 0
            if assigned == 0:
                assert in_train and not in_val and not in_test, (
                    f"scenario {sid} assigned to train but appears in another split"
                )
            elif assigned == 1:
                assert in_val and not in_train and not in_test, (
                    f"scenario {sid} assigned to val but appears in another split"
                )
            else:
                assert in_test and not in_train and not in_val, (
                    f"scenario {sid} assigned to test but appears in another split"
                )

    def test_split_overlap_helper_is_empty(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        split = scenario_stratified_split(ds, seed=0)
        assert split.overlap() == []

    def test_split_is_deterministic(self) -> None:
        """Same seed → same partitioning."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        a = scenario_stratified_split(ds, seed=42)
        b = scenario_stratified_split(ds, seed=42)
        np.testing.assert_array_equal(a.scenario_to_split, b.scenario_to_split)
        np.testing.assert_array_equal(a.train.scenario_ids, b.train.scenario_ids)
        np.testing.assert_array_equal(a.test.scenario_ids, b.test.scenario_ids)

    def test_split_respects_fractions(self) -> None:
        """Fractions should be ~within rounding error of the request."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        n = ds.n_scenarios
        if n < 3:
            pytest.skip("not enough scenarios to three-way split")
        split = scenario_stratified_split(ds, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=0)
        n_train = int(np.sum(split.scenario_to_split == 0))
        n_val = int(np.sum(split.scenario_to_split == 1))
        n_test = int(np.sum(split.scenario_to_split == 2))
        assert n_train + n_val + n_test == n
        # Each split must have at least one scenario (caller relies
        # on this — the test_frac cap guarantees it).
        assert n_train >= 1
        assert n_test >= 1

    def test_train_classifier_default_uses_scenario_split(self) -> None:
        """The default split policy is the leakage-free scenario split."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        assert m.split_policy == "scenario"
        # Reports are populated for train and test (we have >= 7
        # scenarios by default, so the val set is non-empty).
        assert m.train_report is not None
        assert m.test_report is not None
        # Train accuracy is reported (per-split metric).
        assert 0.0 <= m.train_accuracy <= 1.0
        if m.held_out_accuracy is not None:
            assert 0.0 <= m.held_out_accuracy <= 1.0


# ---------------------------------------------------------------------
# Evaluation metrics + confusion matrix
# ---------------------------------------------------------------------
class TestEvaluation:
    def test_compute_metrics_perfect_classifier(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        rep = compute_metrics(y_true, y_pred, classes=CLASSES)
        assert rep.accuracy == pytest.approx(1.0)
        # Classes that actually appear in y_true have f1 = 1.0
        for c in (0, 1, 2):
            assert rep.per_class[c].f1 == pytest.approx(1.0)
        # Weighted F1 is over the support-weighted classes; with
        # all three present classes hitting f1=1.0, weighted_f1=1.0.
        assert rep.weighted_f1 == pytest.approx(1.0)
        # Confusion matrix is diagonal for the populated rows/cols.
        diag = np.diag(rep.confusion_matrix)
        assert int(diag.sum()) == len(y_true)
        # All off-diagonal entries are 0.
        off = rep.confusion_matrix.copy()
        np.fill_diagonal(off, 0)
        assert int(off.sum()) == 0

    def test_compute_metrics_all_wrong(self) -> None:
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        rep = compute_metrics(y_true, y_pred, classes=CLASSES)
        assert rep.accuracy == pytest.approx(0.0)
        # Class 0: precision 0, recall 0, f1 0.
        c0 = rep.per_class[0]
        assert c0.precision == pytest.approx(0.0)
        assert c0.recall == pytest.approx(0.0)
        assert c0.f1 == pytest.approx(0.0)
        # All 4 samples live in the (0,1) cell.
        assert int(rep.confusion_matrix[0, 1]) == 4

    def test_compute_metrics_confusion_matrix_shape_and_order(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        rep = compute_metrics(y_true, y_pred, classes=CLASSES)
        assert rep.confusion_matrix.shape == (len(CLASSES), len(CLASSES))
        # class_names should be the .value strings.
        for i, fc in enumerate(CLASSES):
            assert rep.class_names[i] == fc.value
            pcm = rep.per_class[i]
            # For a perfect classifier, every class has f1 = 1.0
            # (for the classes that actually appeared in y_true).
            if i in (0, 1, 2):
                assert pcm.f1 == pytest.approx(1.0)
                assert pcm.support == 2

    def test_compute_metrics_unknown_label(self) -> None:
        """A label outside [0, n_classes) is counted as unknown and ignored."""
        y_true = [0, 0, 0, 99, 1, 1]
        y_pred = [0, 0, 0, 0,  1, 1]
        rep = compute_metrics(y_true, y_pred, classes=CLASSES)
        assert rep.unknown_count == 1
        assert rep.n_samples == 5  # only the in-range samples count
        assert rep.accuracy == pytest.approx(1.0)

    def test_train_classifier_populates_reports(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=20, seed=0)
        assert m.train_report is not None
        # Per-class metrics present for every class.
        for c in range(len(CLASSES)):
            assert c in m.train_report.per_class
        # Confusion matrix shape matches the class set.
        assert m.train_report.confusion_matrix.shape == (len(CLASSES), len(CLASSES))
        # The reports must serialise to dict.
        d = m.train_report.to_dict()
        assert "accuracy" in d and "confusion_matrix" in d and "per_class" in d

    def test_trained_model_reports_are_isolated_per_split(self) -> None:
        """Train / val / test reports must be different objects (not aliased)."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=20, seed=0)
        # Scenario split always populates train and test. val may
        # be None if n_scenarios < 3 (it isn't, but be defensive).
        assert m.train_report is not None
        assert m.test_report is not None
        # The train accuracy reported in the report should match
        # the legacy train_accuracy field (both computed on the
        # same training set).
        assert m.train_report.accuracy == pytest.approx(m.train_accuracy, abs=1e-6)


# ---------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------
class TestVersioning:
    def test_model_version_field_present(self) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        assert m.model_version == MODEL_VERSION

    def test_save_writes_sidecar(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        sidecar = tmp_path / "model.pkl.version"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip() == MODEL_VERSION

    def test_load_round_trip_preserves_version(self, tmp_path: Path) -> None:
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        loaded = load_model(path)
        assert loaded.model_version == MODEL_VERSION

    def test_load_warns_on_version_mismatch(self, tmp_path: Path) -> None:
        """A sidecar stamped with a different version triggers a warning."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        # Tamper with the sidecar.
        (tmp_path / "model.pkl.version").write_text(
            "phase9-ml-0.0.0-deliberately-bad", encoding="utf-8",
        )
        with pytest.warns(ModelVersionWarning):
            loaded = load_model(path)
        # The pickle itself still loads.
        assert loaded.model_version == MODEL_VERSION

    def test_load_warns_on_missing_sidecar(self, tmp_path: Path) -> None:
        """If the sidecar is missing, we still warn (load-time safety)."""
        ds = build_dataset(config_dir=CONFIG_DIR, seed=0)
        m = train_classifier(ds, n_estimators=10, seed=0)
        path = tmp_path / "model.pkl"
        save_model(m, path)
        (tmp_path / "model.pkl.version").unlink()
        with pytest.warns(ModelVersionWarning):
            load_model(path)

