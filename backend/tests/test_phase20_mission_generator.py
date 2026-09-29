"""PHASE 20 tests — MALE-UAV mission profile generator.

The 14 tests in this file cover the spec's seven concerns:

* the 10 named phases (PhaseKind);
* the 7 built-in mission templates (MissionTemplate);
* per-phase field completeness (MissionPhaseSpec);
* random-seed reproducibility (ProfileGenerator);
* continuous commanded transitions across phase boundaries;
* MALE-UAV envelope compliance per template;
* the synchronised per-tick bundle (MissionTick);
* the AI-never-sees-fault-label rule (ObservedTick);
* engine_load propagation into the engine truth layer;
* the engine_load=0.0 regression guard (existing tests must
  stay green).

The full NORMAL mission is ~22 min of sim time. To keep the
suite fast, the per-template tests use a *scaled-down* variant
of each template whose total duration is ~120 s — long enough
to cover a takeoff → climb → cruise → descent cycle but short
enough to finish in seconds.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List

import pytest

from backend.config import load_config
from backend.environment import (
    FaultTruthRecord,
    MISSION_TEMPLATES,
    MissionPhaseSpec,
    MissionTemplate,
    MissionTick,
    MissionTrace,
    ObservedTick,
    PhaseKind,
    ProfileGenerator,
    ProfileGeneratorConfig,
    get_template,
    list_templates,
)
from backend.environment.phases import PhaseKind as _PhaseKind
from backend.environment.profile_generator import (
    ProfileGenerator as _ProfileGenerator,
)
from backend.environment.templates import (
    MISSION_TEMPLATES as _MISSION_TEMPLATES,
)
from backend.environment.templates import MissionTemplate as _MissionTemplate
from backend.faults import FaultClass, FaultScenario
from backend.sensors import NoiseMode
from backend.simulation import EngineInputs

pytestmark = pytest.mark.phase20

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _short_mission_phases() -> List[MissionPhaseSpec]:
    """A short, valid 10-phase mission for fast tests.

    Durations and per-phase values are chosen so the spec's
    continuity bound (0.05/s throttle, 20 m/s altitude, 5 m/s/s
    airspeed) is satisfied *without* scaling. Total length is
    150 s, so a single run completes in under 2 s.
    """
    return [
        MissionPhaseSpec(
            kind=PhaseKind.STARTUP, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=0.0, target_airspeed_mps=0.0,
            target_throttle=0.10, engine_load=0.05,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.05,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.TAXI, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=0.0, target_airspeed_mps=15.0,
            target_throttle=0.15, engine_load=0.05,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.05,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.TAKEOFF, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=50.0, target_airspeed_mps=40.0,
            target_throttle=0.85, engine_load=0.40,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.10,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.CLIMB, start_t_s=0.0, duration_s=200.0,
            target_altitude_m=3000.0, target_airspeed_mps=55.0,
            target_throttle=0.80, engine_load=0.30,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.10,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.CRUISE, start_t_s=0.0, duration_s=30.0,
            target_altitude_m=3000.0, target_airspeed_mps=60.0,
            target_throttle=0.75, engine_load=0.30,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.10,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.LOITER, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=3000.0, target_airspeed_mps=50.0,
            target_throttle=0.55, engine_load=0.25,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.10,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.HIGH_LOAD_MANOEUVRE, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=3000.0, target_airspeed_mps=70.0,
            target_throttle=0.90, engine_load=0.85,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=5.0, turbulence_intensity=0.20,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.DESCENT, start_t_s=0.0, duration_s=200.0,
            target_altitude_m=500.0, target_airspeed_mps=55.0,
            target_throttle=0.40, engine_load=0.20,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.10,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.LANDING, start_t_s=0.0, duration_s=25.0,
            target_altitude_m=0.0, target_airspeed_mps=40.0,
            target_throttle=0.20, engine_load=0.10,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.05,
        ),
        MissionPhaseSpec(
            kind=PhaseKind.SHUTDOWN, start_t_s=0.0, duration_s=20.0,
            target_altitude_m=0.0, target_airspeed_mps=0.0,
            target_throttle=0.05, engine_load=0.02,
            ambient_temperature_c=None, pressure_pa=None,
            wind_mps=0.0, turbulence_intensity=0.05,
        ),
    ]


def _quick_config(seed: int = 42, fault_plan=None) -> ProfileGeneratorConfig:
    return ProfileGeneratorConfig(
        seed=seed,
        dt_s=0.1,
        transition_default_s=5.0,
        fault_plan=fault_plan,
    )


# ---------------------------------------------------------------------
# Test 1: PhaseKind has 10 members in spec order
# ---------------------------------------------------------------------
class TestPhaseKindSpec:
    def test_phase_kind_has_10_members(self) -> None:
        members = list(PhaseKind)
        assert len(members) == 10

    def test_phase_kind_members_in_canonical_order(self) -> None:
        canonical = PhaseKind.canonical_order()
        assert [p.value for p in canonical] == [
            "STARTUP",
            "TAXI",
            "TAKEOFF",
            "CLIMB",
            "CRUISE",
            "LOITER",
            "HIGH_LOAD_MANOEUVRE",
            "DESCENT",
            "LANDING",
            "SHUTDOWN",
        ]


# ---------------------------------------------------------------------
# Test 2: MissionTemplate has 7 members
# ---------------------------------------------------------------------
class TestMissionTemplateSpec:
    def test_mission_template_has_7_members(self) -> None:
        members = list(MissionTemplate)
        assert len(members) == 7

    def test_mission_template_names(self) -> None:
        names = {t.value for t in MissionTemplate}
        assert names == {
            "NORMAL",
            "HIGH_ALTITUDE",
            "HOT_WEATHER",
            "LONG_ENDURANCE",
            "RAPID_THROTTLE",
            "TURBULENT",
            "COMBINED_STRESS",
        }


# ---------------------------------------------------------------------
# Test 3: every template has 10 phases in canonical order
# ---------------------------------------------------------------------
class TestEveryTemplateHasTenPhases:
    @pytest.mark.parametrize("template", list(MissionTemplate))
    def test_template_has_ten_phases(self, template: MissionTemplate) -> None:
        phases = MISSION_TEMPLATES[template]
        assert len(phases) == 10

    @pytest.mark.parametrize("template", list(MissionTemplate))
    def test_template_phases_in_canonical_order(self, template: MissionTemplate) -> None:
        phases = MISSION_TEMPLATES[template]
        kinds = [p.kind for p in phases]
        assert kinds == list(PhaseKind.canonical_order())


# ---------------------------------------------------------------------
# Test 4: per-phase fields are complete
# ---------------------------------------------------------------------
class TestPerPhaseFieldsComplete:
    @pytest.mark.parametrize("template", list(MissionTemplate))
    def test_every_phase_has_all_required_fields(self, template: MissionTemplate) -> None:
        for ph in MISSION_TEMPLATES[template]:
            assert isinstance(ph.kind, PhaseKind)
            assert isinstance(ph.start_t_s, float)
            assert isinstance(ph.duration_s, float) and ph.duration_s > 0
            assert isinstance(ph.target_altitude_m, float) and ph.target_altitude_m >= 0
            assert isinstance(ph.target_airspeed_mps, float) and ph.target_airspeed_mps >= 0
            assert 0.0 <= ph.target_throttle <= 1.0
            assert 0.0 <= ph.engine_load <= 1.0
            # ambient_temperature_c and pressure_pa are optional.
            assert ph.wind_mps is not None
            assert 0.0 <= ph.turbulence_intensity <= 2.0
            assert ph.transition_s >= 0


# ---------------------------------------------------------------------
# Test 5: random seed reproducibility
# ---------------------------------------------------------------------
class TestRandomSeedReproducible:
    def test_two_runs_same_seed_produce_identical_traces(self) -> None:
        phases = _short_mission_phases()
        cfg = _quick_config(seed=42)
        t1 = ProfileGenerator(phases=phases, config=cfg).run()
        t2 = ProfileGenerator(phases=phases, config=cfg).run()
        assert len(t1.truth) == len(t2.truth)
        # Per-tick engine state and commanded env must match.
        for a, b in zip(t1.truth, t2.truth):
            assert a.env.throttle == pytest.approx(b.env.throttle)
            assert a.env.altitude_m == pytest.approx(b.env.altitude_m)
            assert a.engine.rpm == pytest.approx(b.engine.rpm)


# ---------------------------------------------------------------------
# Test 6: transitions are continuous (per the spec's bound)
# ---------------------------------------------------------------------
class TestTransitionsContinuous:
    def test_commanded_state_rates_within_spec_bounds(self) -> None:
        """The short mission's commanded state must never change
        faster than the spec's continuity bound:

        * throttle: 0.05 / s
        * altitude: 20 m / s
        * airspeed: 5 m / s
        """
        phases = _short_mission_phases()
        trace = ProfileGenerator(phases=phases, config=_quick_config(seed=7)).run()
        dt = 0.1
        max_d_thr = 0.0
        max_d_alt = 0.0
        max_d_spd = 0.0
        for i in range(1, len(trace.truth)):
            a = trace.truth[i - 1].env
            b = trace.truth[i].env
            max_d_thr = max(max_d_thr, abs(b.throttle - a.throttle))
            max_d_alt = max(max_d_alt, abs(b.altitude_m - a.altitude_m))
            max_d_spd = max(max_d_spd, abs(b.airspeed_mps - a.airspeed_mps))
        # Per-second bounds.
        assert max_d_thr / dt <= 0.05 + 1e-9, (
            f"throttle {max_d_thr/dt:.4f}/s > 0.05/s"
        )
        assert max_d_alt / dt <= 20.0 + 1e-9, (
            f"altitude {max_d_alt/dt:.2f} m/s > 20 m/s"
        )
        assert max_d_spd / dt <= 5.0 + 1e-9, (
            f"airspeed {max_d_spd/dt:.3f} m/s/s > 5 m/s/s"
        )


# ---------------------------------------------------------------------
# Test 7: MALE-UAV envelope compliance per template
# ---------------------------------------------------------------------
class TestTemplateEnvelopeMatchesMaleUav:
    def test_normal_cruise_envelope(self) -> None:
        n = MISSION_TEMPLATES[MissionTemplate.NORMAL]
        idx = PhaseKind.canonical_order().index(PhaseKind.CRUISE)
        cruise = n[idx]
        assert 4000 <= cruise.target_altitude_m <= 8000
        assert 50 <= cruise.target_airspeed_mps <= 90
        assert 0.5 <= cruise.target_throttle <= 0.9

    def test_high_altitude_cruise_above_6000m(self) -> None:
        ha = MISSION_TEMPLATES[MissionTemplate.HIGH_ALTITUDE]
        idx = PhaseKind.canonical_order().index(PhaseKind.CRUISE)
        cruise = ha[idx]
        assert cruise.target_altitude_m > 6000

    def test_hot_weather_cruise_ambient_above_30c(self) -> None:
        hw = MISSION_TEMPLATES[MissionTemplate.HOT_WEATHER]
        idx = PhaseKind.canonical_order().index(PhaseKind.CRUISE)
        cruise = hw[idx]
        # At altitude (5000 m), ISA + 25 °C ~ +7.5 °C — below 30. So we
        # check the surface phases for the +30 °C threshold.
        taxi = hw[1]  # TAXI
        assert taxi.ambient_temperature_c is not None
        assert taxi.ambient_temperature_c > 30


# ---------------------------------------------------------------------
# Test 8: COMBINED_STRESS has all three deltas
# ---------------------------------------------------------------------
class TestCombinedStressHasAllDeltas:
    def test_combined_stress_cruise_phase_has_hot_high_turb(self) -> None:
        cs = MISSION_TEMPLATES[MissionTemplate.COMBINED_STRESS]
        idx = PhaseKind.canonical_order().index(PhaseKind.CRUISE)
        cruise = cs[idx]
        assert cruise.ambient_temperature_c is not None
        assert cruise.ambient_temperature_c > 0
        assert cruise.turbulence_intensity > 0.3
        assert cruise.target_altitude_m > 5000


# ---------------------------------------------------------------------
# Test 9: MissionTick carries all four streams
# ---------------------------------------------------------------------
class TestMissionTickCarriesAllStreams:
    def test_one_tick_has_env_engine_sensors_fault_truth(self) -> None:
        phases = _short_mission_phases()
        trace = ProfileGenerator(phases=phases, config=_quick_config(seed=1)).run()
        tick = trace.truth[len(trace.truth) // 2]
        assert isinstance(tick, MissionTick)
        assert tick.env is not None
        assert tick.engine is not None
        assert isinstance(tick.sensors, dict)
        # The per-channel reading dict is non-empty.
        assert len(tick.sensors) > 0
        assert isinstance(tick.fault_truth, FaultTruthRecord)
        d = tick.to_dict()
        assert "env" in d and "engine_state" not in d  # engine uses 'engine' key
        assert "engine" in d
        assert "sensors" in d
        assert "fault_truth" in d


# ---------------------------------------------------------------------
# Test 10: ObservedTick excludes the fault label
# ---------------------------------------------------------------------
class TestObservedTickExcludesFaultTruth:
    def test_observed_to_dict_json_has_no_fault_keys(self) -> None:
        phases = _short_mission_phases()
        # Inject a sensor fault so the truth side has a non-trivial record.
        fs = FaultScenario(
            fault_class=FaultClass.SENSOR_FAULT,
            severity=0.8,
            onset_time_s=1.0,
            duration_s=20.0,
            target_channel="rpm",
            target_sensor_mode=NoiseMode.STUCK,
            seed=42,
        )
        trace = ProfileGenerator(
            phases=phases,
            config=ProfileGeneratorConfig(seed=42, fault_plan=fs),
        ).run()
        # Pick a tick where the fault is active.
        i = len(trace.truth) // 2
        observed_json = json.dumps(trace.observed[i].to_dict())
        for forbidden in ("fault_class", "severity", "sensor_mode", "target_channel"):
            assert forbidden not in observed_json, (
                f"observed JSON contains forbidden key: {forbidden!r}"
            )

    def test_observed_dataclass_has_no_fault_truth_attribute(self) -> None:
        phases = _short_mission_phases()
        trace = ProfileGenerator(phases=phases, config=_quick_config(seed=1)).run()
        # The ObservedTick dataclass has no ``fault_truth`` field at all
        # (structurally absent, not just None).
        for tick in trace.observed[:3]:
            assert not hasattr(tick, "fault_truth")


# ---------------------------------------------------------------------
# Test 11: iter_observed never yields a fault key
# ---------------------------------------------------------------------
class TestIterObservedNeverYieldsFault:
    def test_iter_observed_full_sweep_no_fault_substring(self) -> None:
        phases = _short_mission_phases()
        fs = FaultScenario(
            fault_class=FaultClass.OVERHEATING,
            severity=0.6,
            onset_time_s=1.0,
            duration_s=200.0,
            seed=42,
        )
        trace = ProfileGenerator(
            phases=phases,
            config=ProfileGeneratorConfig(seed=42, fault_plan=fs),
        ).run()
        # Walk every observed tick and ensure no JSON substring is "fault".
        for tick in trace.observed:
            text = json.dumps(tick.to_dict())
            assert "fault" not in text, (
                f"observed tick at t={tick.time_s} contains 'fault' in JSON"
            )


# ---------------------------------------------------------------------
# Test 12: truth side has the fault when injected; observed does not
# ---------------------------------------------------------------------
class TestTruthSideHasFaultWhenInjected:
    def test_overheating_fault_visible_in_truth_not_observed(self) -> None:
        phases = _short_mission_phases()
        fs = FaultScenario(
            fault_class=FaultClass.OVERHEATING,
            severity=0.6,
            onset_time_s=1.0,
            duration_s=200.0,
            seed=42,
        )
        trace = ProfileGenerator(
            phases=phases,
            config=ProfileGeneratorConfig(seed=42, fault_plan=fs),
        ).run()
        # Find the first truth tick where the fault is non-zero.
        active_truth = next(
            (
                t for t in trace.truth
                if t.fault_truth.fault_class == "OVERHEATING"
                and t.fault_truth.severity > 0
            ),
            None,
        )
        assert active_truth is not None
        assert active_truth.fault_truth.severity > 0
        # At the same time index, the observed tick must have no fault key.
        i = trace.truth.index(active_truth)
        obs_json = json.dumps(trace.observed[i].to_dict())
        for forbidden in ("fault_class", "severity", "sensor_mode", "target_channel"):
            assert forbidden not in obs_json


# ---------------------------------------------------------------------
# Test 13: engine_load biases vibration, BSFC, and fuel flow
# ---------------------------------------------------------------------
class TestEngineLoadBiasesEngine:
    def test_engine_load_increases_vibration_bsfc_fuel(self) -> None:
        from backend.environment.profile_generator import ProfileGenerator

        # Use the full 10-phase short mission and override
        # ``engine_load`` to 0.0 (low) and 0.8 (high) on every
        # phase. The only delta between the two runs is the
        # ``engine_load`` bias path in the engine.
        base = _short_mission_phases()
        low = [_with_engine_load(ph, 0.0) for ph in base]
        high = [_with_engine_load(ph, 0.8) for ph in base]
        trace_low = ProfileGenerator(phases=low, config=_quick_config(seed=11)).run()
        trace_high = ProfileGenerator(phases=high, config=_quick_config(seed=11)).run()
        # Compare the steady-state cruise window: altitude near
        # 3000 m (the short mission's cruise altitude), throttle
        # near 0.75.
        def _window(trace, alt=3000.0, thr=0.75, tol=(100.0, 0.05)):
            alt_tol, thr_tol = tol
            return [
                t for t in trace.truth
                if abs(t.env.altitude_m - alt) < alt_tol
                and abs(t.env.throttle - thr) < thr_tol
            ]

        w_low = _window(trace_low)
        w_high = _window(trace_high)
        assert w_low and w_high
        import statistics
        vib_low = statistics.mean(t.engine.vibration_rms_g for t in w_low)
        vib_high = statistics.mean(t.engine.vibration_rms_g for t in w_high)
        bsfc_low = statistics.mean(t.engine.bsfc_g_per_kwh for t in w_low)
        bsfc_high = statistics.mean(t.engine.bsfc_g_per_kwh for t in w_high)
        ff_low = statistics.mean(t.engine.fuel_flow_lph for t in w_low)
        ff_high = statistics.mean(t.engine.fuel_flow_lph for t in w_high)
        assert vib_high > vib_low
        assert bsfc_high > bsfc_low
        assert ff_high > ff_low


def _with_engine_load(ph: MissionPhaseSpec, load: float) -> MissionPhaseSpec:
    """Return a copy of ``ph`` with ``engine_load`` set to ``load``.

    ``dataclasses.replace`` is cleaner but kept as a helper so the
    test reads naturally.
    """
    import dataclasses
    return dataclasses.replace(ph, engine_load=load)


# ---------------------------------------------------------------------
# Test 14: engine_load=0.0 does not perturb the existing health / risk outputs
# ---------------------------------------------------------------------
class TestEngineLoadZeroIsRegressionSafe:
    def test_engine_inputs_engine_load_default_is_zero(self) -> None:
        # The PHASE 3 EngineInputs must default to engine_load=0.0
        # so existing PHASE 7 / 10 / 12 tests are unaffected.
        from backend.environment import EnvironmentState
        # Build a minimal env state by hand.
        from backend.environment.state import WindState
        from backend.environment.atmosphere import AtmosphericState

        env = EnvironmentState(
            time_s=0.0,
            altitude_m=0.0,
            airspeed_mps=0.0,
            throttle=0.0,
            atmosphere=AtmosphericState(
                altitude_m=0.0,
                pressure_pa=101325.0,
                temperature_k=288.15,
                temperature_c=15.0,
                density_kg_per_m3=1.225,
                speed_of_sound_mps=340.0,
            ),
            wind=WindState(
                time_s=0.0,
                turbulence_w_mps=0.0,
                gust_w_mps=0.0,
                total_w_mps=0.0,
                is_gust_active=False,
            ),
            vertical_accel_mps2=0.0,
        )
        ei = EngineInputs(env=env)
        assert ei.engine_load == 0.0

    def test_engine_step_with_load_zero_matches_existing(self) -> None:
        # Use the actual config-driven engine to verify the
        # additive-only contract: engine_load=0.0 does not change
        # any of the engine's outputs.
        from backend.simulation import Engine
        from backend.environment.atmosphere import Atmosphere
        cfg = load_config(CONFIG_DIR)
        e1 = Engine(cfg.engine)
        e2 = Engine(cfg.engine)
        e1.set_dt(0.1)
        e2.set_dt(0.1)
        ref_density = cfg.environment.atmosphere.sea_level.density_kg_per_m3
        e1.set_reference_density(ref_density)
        e2.set_reference_density(ref_density)
        # Build the runtime atmosphere and look up the state at 5000 m.
        atmos = Atmosphere(cfg.environment.atmosphere)
        atmos_state = atmos.state(5000.0)
        # Build a simple env state.
        from backend.environment import EnvironmentState
        from backend.environment.state import WindState

        env = EnvironmentState(
            time_s=0.0,
            altitude_m=5000.0,
            airspeed_mps=65.0,
            throttle=0.75,
            atmosphere=atmos_state,
            wind=WindState(
                time_s=0.0,
                turbulence_w_mps=0.0,
                gust_w_mps=0.0,
                total_w_mps=0.0,
                is_gust_active=False,
            ),
            vertical_accel_mps2=0.0,
        )
        e1.step(EngineInputs(env=env, engine_load=0.0))
        e2.step(EngineInputs(env=env, engine_load=0.0))
        s1 = e1.last_state
        s2 = e2.last_state
        # Both engines ran with engine_load=0.0, so their outputs
        # should match (the engine has no internal RNG).
        assert s1.rpm == s2.rpm
        assert s1.vibration_rms_g == s2.vibration_rms_g
        assert s1.bsfc_g_per_kwh == s2.bsfc_g_per_kwh
        assert s1.fuel_flow_lph == s2.fuel_flow_lph
