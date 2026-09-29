"""PHASE 21 tests — controlled fault-injection framework.

The 14 tests in this file cover:

* the FaultCategory / FaultType taxonomy (3 categories, 17 types,
  every type maps to a real FaultClass);
* the per-type metadata (affected_parameters, sensor_mode, env_knob);
* the FaultRecord dataclass (roundtrip, required fields);
* the new temporal patterns (exponential, pulse) on top of
  the existing linear / step;
* the ScenarioGenerator (enumeration count, scenario_id uniqueness,
  no two scenarios identical);
* the SplitPolicy (deterministic, hash-based, no leakage);
* the DatasetWriter (writes manifest + traces, no contamination).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from backend.datasets import (
    DatasetManifest,
    DatasetSpec,
    DatasetWriter,
    SplitPolicy,
)
from backend.environment import MissionTemplate
from backend.faults import (
    DEFAULT_SEVERITY_LEVELS,
    DEFAULT_TEMPORAL_PATTERNS,
    FAULT_TAXONOMY,
    FaultCategory,
    FaultClass,
    FaultRecord,
    FaultScenarioBundle,
    FaultType,
    ScenarioGenerator,
    by_category,
    by_class,
    get_meta,
)
from backend.faults.progression import _VALID_MODELS, severity_at
from backend.sensors import NoiseMode, SENSOR_CHANNELS

pytestmark = pytest.mark.phase21


# ---------------------------------------------------------------------
# Test 1: FaultCategory has exactly three members
# ---------------------------------------------------------------------
class TestFaultCategorySpec:
    def test_fault_category_has_three_members(self) -> None:
        assert {c.value for c in FaultCategory} == {
            "ENGINE", "SENSOR", "ENVIRONMENT",
        }


# ---------------------------------------------------------------------
# Test 2: FaultType count + spec'd names
# ---------------------------------------------------------------------
class TestFaultTypeSpec:
    def test_engine_category_has_five_types(self) -> None:
        engine = by_category(FaultCategory.ENGINE)
        names = {ft.value for ft in engine}
        assert names == {
            "GRADUAL_PERFORMANCE_DEGRADATION",
            "THERMAL_DEGRADATION",
            "PRESSURE_RELATED_ANOMALY",
            "VIBRATION_RELATED_ANOMALY",
            "EFFICIENCY_LOSS",
        }

    def test_sensor_category_has_six_types(self) -> None:
        sensor = by_category(FaultCategory.SENSOR)
        names = {ft.value for ft in sensor}
        assert names == {
            "SENSOR_BIAS",
            "SENSOR_DRIFT",
            "SENSOR_DROPOUT",
            "SENSOR_STUCK",
            "SENSOR_SPIKE",
            "SENSOR_NOISE_INCREASE",
        }

    def test_environment_category_has_at_least_five_types(self) -> None:
        env = by_category(FaultCategory.ENVIRONMENT)
        names = {ft.value for ft in env}
        # The spec named 5; we added a 6th (pressure_shift).
        assert {
            "ENV_TURBULENCE",
            "ENV_GUST",
            "ENV_TEMP_SHIFT",
            "ENV_ALTITUDE_TRANSITION",
            "ENV_RAPID_THROTTLE",
        }.issubset(names)
        assert len(env) >= 5

    def test_fault_type_count_matches_registry(self) -> None:
        # 5 + 6 + 6 (or 5) = 17 (or 16). Locked to 17 by taxonomy
        # definition.
        assert len(FAULT_TAXONOMY) == 17
        assert len(list(FaultType)) == 17


# ---------------------------------------------------------------------
# Test 3: every FaultType maps to a real FaultClass
# ---------------------------------------------------------------------
class TestFaultTypeMappings:
    @pytest.mark.parametrize("ft", list(FaultType))
    def test_every_type_maps_to_existing_class(self, ft: FaultType) -> None:
        meta = get_meta(ft)
        assert meta.mapped_class in set(FaultClass)
        # And the same type can be found by_class().
        assert ft in by_class(meta.mapped_class)


# ---------------------------------------------------------------------
# Test 4: every FaultType has affected_parameters
# ---------------------------------------------------------------------
class TestFaultTypeMetadata:
    @pytest.mark.parametrize("ft", list(FaultType))
    def test_every_type_has_affected_parameters(self, ft: FaultType) -> None:
        meta = get_meta(ft)
        assert len(meta.affected_parameters) >= 1
        for p in meta.affected_parameters:
            assert isinstance(p, str)
            assert len(p) > 0

    @pytest.mark.parametrize("ft", list(FaultType))
    def test_every_type_has_valid_progression(self, ft: FaultType) -> None:
        meta = get_meta(ft)
        assert meta.default_progression in _VALID_MODELS

    @pytest.mark.parametrize("ft", list(FaultType))
    def test_every_type_has_positive_duration(self, ft: FaultType) -> None:
        meta = get_meta(ft)
        assert meta.default_duration_s > 0

    def test_sensor_types_have_sensor_mode(self) -> None:
        for ft in by_category(FaultCategory.SENSOR):
            meta = get_meta(ft)
            assert meta.sensor_mode is not None
            assert meta.sensor_mode in set(NoiseMode)

    def test_engine_types_have_no_sensor_mode(self) -> None:
        for ft in by_category(FaultCategory.ENGINE):
            meta = get_meta(ft)
            assert meta.sensor_mode is None

    def test_environment_types_have_env_knob(self) -> None:
        for ft in by_category(FaultCategory.ENVIRONMENT):
            meta = get_meta(ft)
            assert meta.env_knob is not None
            assert isinstance(meta.env_knob, str)


# ---------------------------------------------------------------------
# Test 5: FaultRecord roundtrip
# ---------------------------------------------------------------------
class TestFaultRecordRoundtrip:
    def _sample(self) -> FaultRecord:
        return FaultRecord(
            fault_id="flt-abcdef0123456789",
            fault_type=FaultType.THERMAL_DEGRADATION,
            start_time=100.0,
            end_time=300.0,
            severity=0.5,
            affected_parameters=("egt", "cht"),
            ground_truth={
                "fault_class": "OVERHEATING",
                "category": "ENGINE",
                "peak_severity": 0.5,
            },
            scenario_id="scn-1234567890abcdef",
            temporal_pattern="linear",
            severity_level="medium",
        )

    def test_required_fields_present(self) -> None:
        rec = self._sample()
        d = rec.to_dict()
        for key in (
            "fault_id", "fault_type", "start_time", "end_time",
            "severity", "affected_parameters", "ground_truth",
            "scenario_id", "temporal_pattern", "severity_level",
        ):
            assert key in d

    def test_roundtrip_lossless(self) -> None:
        rec = self._sample()
        d = rec.to_dict()
        rec2 = FaultRecord.from_dict(d)
        assert rec2 == rec


# ---------------------------------------------------------------------
# Test 6: severity levels are exactly three
# ---------------------------------------------------------------------
class TestSeverityLevels:
    def test_default_severity_levels_are_three(self) -> None:
        names = [name for name, _ in DEFAULT_SEVERITY_LEVELS]
        assert names == ["low", "medium", "high"]

    def test_default_severity_peaks_in_range(self) -> None:
        for name, peak in DEFAULT_SEVERITY_LEVELS:
            assert 0.0 < peak < 1.0
            assert name in ("low", "medium", "high")

    def test_dataset_spec_default_severities(self) -> None:
        spec = DatasetSpec()
        assert spec.severity_levels == ("low", "medium", "high")


# ---------------------------------------------------------------------
# Test 7: temporal patterns include exponential and pulse
# ---------------------------------------------------------------------
class TestTemporalPatterns:
    def test_default_patterns_are_four(self) -> None:
        assert len(DEFAULT_TEMPORAL_PATTERNS) == 4
        assert set(DEFAULT_TEMPORAL_PATTERNS) == {
            "linear", "step", "exponential", "pulse",
        }

    def test_exponential_rises_then_decays(self) -> None:
        peak = 1.0
        onset, dur = 0.0, 30.0
        # At onset, severity is 0.
        assert severity_at(onset, onset, dur, peak, "exponential") == 0.0
        # At onset + 1s (after the 1s ramp), severity is at peak.
        assert severity_at(onset + 1.0, onset, dur, peak, "exponential") == pytest.approx(peak, abs=1e-9)
        # Past the active window, severity is 0.
        assert severity_at(onset + dur + 1.0, onset, dur, peak, "exponential") == 0.0
        # In the middle of the window, severity is positive but
        # strictly less than peak.
        mid = onset + dur / 2.0
        v = severity_at(mid, onset, dur, peak, "exponential")
        assert 0.0 < v < peak

    def test_pulse_has_three_segments(self) -> None:
        peak = 1.0
        onset, dur = 0.0, 10.0
        # 0..20% ramps 0 -> 0.6*peak
        assert severity_at(onset, onset, dur, peak, "pulse") == 0.0
        assert severity_at(onset + dur * 0.5, onset, dur, peak, "pulse") == pytest.approx(0.6, abs=1e-9)
        # 80%..100% ramps 0.6 -> 0.4
        v = severity_at(onset + dur * 0.9, onset, dur, peak, "pulse")
        assert 0.4 < v < 0.6

    def test_linear_and_step_unchanged(self) -> None:
        # Sanity: the existing patterns still work.
        assert severity_at(0.0, 0.0, 10.0, 1.0, "linear") == 0.0
        assert severity_at(5.0, 0.0, 10.0, 1.0, "linear") == pytest.approx(0.5)
        assert severity_at(5.0, 0.0, 10.0, 0.7, "step") == pytest.approx(0.7)


# ---------------------------------------------------------------------
# Test 8: ScenarioGenerator counts and enumerates correctly
# ---------------------------------------------------------------------
class TestScenarioGeneratorEnumeration:
    def _gen(
        self,
        templates: List[MissionTemplate],
        fault_types: List[FaultType],
        severities: List[str] = ("low", "medium"),
        patterns: List[str] = ("linear", "step"),
        phases: int = 2,
        seed: int = 42,
    ) -> ScenarioGenerator:
        return ScenarioGenerator(
            mission_templates=templates,
            fault_types=fault_types,
            severity_levels=severities,
            temporal_patterns=patterns,
            phases_per_template=phases,
            base_seed=seed,
        )

    def test_count_matches_combinatorial_product(self) -> None:
        gen = self._gen(
            [MissionTemplate.NORMAL, MissionTemplate.HOT_WEATHER],
            [FaultType.THERMAL_DEGRADATION, FaultType.ENV_TURBULENCE],
        )
        expected = (
            2  # templates
            * 2  # fault types
            * 2  # severities
            * 2  # patterns
            * 2  # phases
        )
        assert gen.total_count == expected

    def test_enumerate_yields_expected_count(self) -> None:
        gen = self._gen(
            [MissionTemplate.NORMAL],
            [FaultType.THERMAL_DEGRADATION, FaultType.ENV_TURBULENCE],
        )
        n = sum(1 for _ in gen.enumerate_scenarios())
        assert n == gen.total_count


# ---------------------------------------------------------------------
# Test 9: scenario_id is unique
# ---------------------------------------------------------------------
class TestScenarioIdUniqueness:
    def test_all_scenario_ids_unique_in_a_small_catalog(self) -> None:
        gen = ScenarioGenerator(
            mission_templates=[MissionTemplate.NORMAL],
            fault_types=[
                FaultType.THERMAL_DEGRADATION,
                FaultType.ENV_TURBULENCE,
                FaultType.SENSOR_BIAS,
            ],
            severity_levels=["low", "medium", "high"],
            temporal_patterns=["linear", "step", "exponential", "pulse"],
            phases_per_template=4,
            base_seed=42,
        )
        seen: set = set()
        for bundle in gen.enumerate_scenarios():
            assert bundle.scenario_id not in seen
            seen.add(bundle.scenario_id)
        assert len(seen) == gen.total_count


# ---------------------------------------------------------------------
# Test 10: SplitPolicy is deterministic
# ---------------------------------------------------------------------
class TestSplitPolicyDeterminism:
    def _scenarios(self) -> List[FaultScenarioBundle]:
        gen = ScenarioGenerator(
            mission_templates=[MissionTemplate.NORMAL],
            fault_types=[FaultType.THERMAL_DEGRADATION],
            severity_levels=["low", "medium", "high"],
            temporal_patterns=["linear", "step"],
            phases_per_template=3,
            base_seed=42,
        )
        return list(gen.enumerate_scenarios())

    def test_same_spec_same_assignment(self) -> None:
        bundles = self._scenarios()
        sids = [b.scenario_id for b in bundles]
        spec = DatasetSpec(
            name="d1",
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=42,
        )
        p1 = SplitPolicy(spec, sids)
        p2 = SplitPolicy(spec, sids)
        for sid in sids:
            assert p1.assign(sid) == p2.assign(sid)

    def test_different_seed_different_assignment(self) -> None:
        bundles = self._scenarios()
        sids = [b.scenario_id for b in bundles]
        # At least one scenario must flip splits when the seed changes.
        spec_a = DatasetSpec(
            name="d2", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=1,
        )
        spec_b = DatasetSpec(
            name="d2", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
            seed=999,
        )
        p_a = SplitPolicy(spec_a, sids)
        p_b = SplitPolicy(spec_b, sids)
        # Find at least one id where the two assignments differ.
        differs = [sid for sid in sids if p_a.assign(sid) != p_b.assign(sid)]
        # The hash-based split should not be identical across seeds.
        assert len(differs) > 0


# ---------------------------------------------------------------------
# Test 11: SplitPolicy no leakage
# ---------------------------------------------------------------------
class TestSplitPolicyNoLeakage:
    def test_no_scenario_in_two_splits(self) -> None:
        gen = ScenarioGenerator(
            mission_templates=[MissionTemplate.NORMAL],
            fault_types=[FaultType.THERMAL_DEGRADATION, FaultType.ENV_TURBULENCE],
            severity_levels=["low", "medium"],
            temporal_patterns=["linear", "step"],
            phases_per_template=4,
            base_seed=42,
        )
        sids = [b.scenario_id for b in gen.enumerate_scenarios()]
        spec = DatasetSpec(
            name="d3",
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            seed=42,
        )
        p = SplitPolicy(spec, sids)
        p.validate_no_leakage()  # must not raise

        # Manually verify disjointness.
        seen: dict = {}
        for sid in sids:
            bucket = p.assign(sid)
            if sid in seen:
                assert seen[sid] == bucket
            seen[sid] = bucket

    def test_ratios_summing_to_one_required(self) -> None:
        with pytest.raises(ValueError):
            DatasetSpec(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_train_ratio_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DatasetSpec(train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)


# ---------------------------------------------------------------------
# Test 12: DatasetWriter produces manifest + traces
# ---------------------------------------------------------------------
class TestDatasetWriterEndToEnd:
    def test_writer_produces_manifest_and_stub_traces(self) -> None:
        spec = DatasetSpec(
            name="phase21_smoke",
            mission_templates=(MissionTemplate.NORMAL,),
            fault_types=(FaultType.THERMAL_DEGRADATION,),
            severity_levels=("low", "medium"),
            temporal_patterns=("linear", "step"),
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            seed=42,
            phases_per_template=2,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "phase21_smoke"
            manifest = DatasetWriter(out, spec, run_scenarios=False).write()

            # Manifest exists and parses.
            assert (out / "manifest.json").exists()
            roundtrip = DatasetManifest.from_json(manifest.to_json())
            assert roundtrip.dataset_name == "phase21_smoke"
            assert roundtrip.total_scenarios() == 8

            # Every split has the right number of ids.
            total = sum(len(v) for v in manifest.splits.values())
            assert total == 8

            # Every recorded scenario_id appears in the manifest
            # and in the splits.
            all_ids = {sid for ids in manifest.splits.values() for sid in ids}
            assert len(all_ids) == 8
            for sid in all_ids:
                assert sid in manifest.scenario_paths

            # The npz stub files exist.
            traces_dir = out / "traces"
            for sid in all_ids:
                assert (traces_dir / f"{sid}.npz").exists()

    def test_writer_no_contamination_same_seed(self) -> None:
        # Two runs with the same spec must produce the exact same
        # split assignment.
        spec = DatasetSpec(
            name="phase21_no_contam",
            mission_templates=(MissionTemplate.NORMAL,),
            fault_types=(FaultType.THERMAL_DEGRADATION,),
            severity_levels=("low",),
            temporal_patterns=("linear",),
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            seed=42,
            phases_per_template=2,
        )
        with tempfile.TemporaryDirectory() as td:
            out1 = Path(td) / "a"
            out2 = Path(td) / "b"
            m1 = DatasetWriter(out1, spec, run_scenarios=False).write()
            m2 = DatasetWriter(out2, spec, run_scenarios=False).write()
            assert m1.splits == m2.splits


# ---------------------------------------------------------------------
# Test 13: no two scenarios are identical
# ---------------------------------------------------------------------
class TestNoTwoScenariosIdentical:
    def test_unique_combinations(self) -> None:
        gen = ScenarioGenerator(
            mission_templates=[MissionTemplate.NORMAL],
            fault_types=[
                FaultType.THERMAL_DEGRADATION,
                FaultType.ENV_TURBULENCE,
                FaultType.SENSOR_BIAS,
            ],
            severity_levels=["low", "medium", "high"],
            temporal_patterns=["linear", "step", "exponential", "pulse"],
            phases_per_template=2,
            base_seed=42,
        )
        seen: set = set()
        for bundle in gen.enumerate_scenarios():
            rec = bundle.fault_records[0]
            key = (
                rec.fault_type,
                rec.severity_level,
                rec.temporal_pattern,
                rec.start_time,
            )
            assert key not in seen, f"duplicate scenario: {key}"
            seen.add(key)


# ---------------------------------------------------------------------
# Test 14: severity levels correspond to peaks
# ---------------------------------------------------------------------
class TestSeverityLevelPeaks:
    def test_peaks_match(self) -> None:
        gen = ScenarioGenerator(
            mission_templates=[MissionTemplate.NORMAL],
            fault_types=[FaultType.THERMAL_DEGRADATION],
            severity_levels=["low", "medium", "high"],
            temporal_patterns=["linear"],
            phases_per_template=1,
            base_seed=42,
        )
        expected = {"low": 0.2, "medium": 0.5, "high": 0.85}
        seen_levels: set = set()
        for bundle in gen.enumerate_scenarios():
            rec = bundle.fault_records[0]
            seen_levels.add(rec.severity_level)
            assert rec.severity == pytest.approx(expected[rec.severity_level])
            # Also: the underlying FaultScenario carries the same
            # peak.
            sc = bundle.fault_scenarios[0]
            assert sc.severity == pytest.approx(expected[rec.severity_level])
        assert seen_levels == {"low", "medium", "high"}
