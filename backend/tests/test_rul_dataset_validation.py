"""Automated validation suite verifying Step 8 criteria for calibrated Rotax 914 RUL."""

import pytest
from backend.rul.model import HybridRulModel, load_engine_limits


def test_step8_rul_decreases_with_runtime():
    model = HybridRulModel()
    res_0h = model.evaluate(engine_age_hours=50.0, wear=0.02)
    res_500h = model.evaluate(engine_age_hours=600.0, wear=0.30)
    res_1000h = model.evaluate(engine_age_hours=1100.0, wear=0.55)
    res_1800h = model.evaluate(engine_age_hours=1850.0, wear=0.92)

    assert res_0h["remaining_hours"] > res_500h["remaining_hours"]
    assert res_500h["remaining_hours"] > res_1000h["remaining_hours"]
    assert res_1000h["remaining_hours"] > res_1800h["remaining_hours"]
    # Check calibrated lifecycle bands
    assert 1800.0 <= res_0h["remaining_hours"] <= 2000.0   # New
    assert 800.0 <= res_500h["remaining_hours"] <= 1500.0 # Mid-life
    assert 10.0 <= res_1800h["remaining_hours"] <= 400.0  # Degraded / Critical


def test_step8_high_cht_accelerates_degradation():
    model = HybridRulModel()
    res_normal = model.evaluate(cht_c=118.0, engine_age_hours=500.0, wear=0.25)
    res_high_cht = model.evaluate(cht_c=165.0, engine_age_hours=500.0, wear=0.25)

    assert res_high_cht["remaining_hours"] < res_normal["remaining_hours"]
    assert res_high_cht["health_index"] < res_normal["health_index"]


def test_step8_low_oil_pressure_reduces_rul():
    model = HybridRulModel()
    res_nominal_p = model.evaluate(oil_pressure_bar=4.2, engine_age_hours=600.0, wear=0.30)
    res_low_p = model.evaluate(oil_pressure_bar=2.1, engine_age_hours=600.0, wear=0.30)

    assert res_low_p["remaining_hours"] < res_nominal_p["remaining_hours"]


def test_step8_high_vibration_reduces_rul():
    model = HybridRulModel()
    res_nominal_vib = model.evaluate(vibration_rms_g=0.45, engine_age_hours=500.0, wear=0.25)
    res_high_vib = model.evaluate(vibration_rms_g=1.65, engine_age_hours=500.0, wear=0.25)

    assert res_high_vib["remaining_hours"] < res_nominal_vib["remaining_hours"]


def test_step8_healthy_vs_fault_scenario():
    model = HybridRulModel()
    # Healthy mission
    healthy_res = model.evaluate(
        rpm=4800.0, cht_c=120.0, egt_c=780.0, oil_pressure_bar=4.0,
        oil_temp_c=95.0, fuel_flow_lph=16.0, vibration_rms_g=0.65,
        engine_age_hours=200.0, wear=0.10, fault_severity=0.0,
    )
    # Fault scenario (overheating + abnormal vibration)
    fault_res = model.evaluate(
        rpm=5200.0, cht_c=168.0, egt_c=860.0, oil_pressure_bar=2.4,
        oil_temp_c=122.0, fuel_flow_lph=22.0, vibration_rms_g=1.55,
        engine_age_hours=200.0, wear=0.10, fault_severity=0.85,
    )

    assert healthy_res["remaining_hours"] > 1600.0
    assert fault_res["remaining_hours"] < 150.0  # Rapid critical drop
    assert fault_res["remaining_hours"] < healthy_res["remaining_hours"] * 0.15


def test_step8_benchmark_target_alignment():
    """Verify Master Prompt example: 842.6 h, 241 cycles, 0.92 confidence, 81.4% health index, 842.6 ± 28.4 h."""
    model = HybridRulModel()
    # Mid-life calibrated reference point
    res = model.evaluate(
        rpm=4750.0,
        cht_c=125.0,
        egt_c=785.0,
        oil_pressure_bar=3.95,
        oil_temp_c=102.0,
        fuel_flow_lph=16.2,
        vibration_rms_g=0.70,
        engine_age_hours=960.0,
        wear=0.48,
        health_score=0.84,
    )

    assert 800.0 <= res["remaining_hours"] <= 900.0
    assert 220 <= res["remaining_cycles"] <= 260
    assert 0.88 <= res["confidence"] <= 0.96
    assert 75.0 <= res["health_index"] <= 86.0
    assert res["lower_hours"] < res["remaining_hours"] < res["upper_hours"]
    assert res["uncertainty_hours"] < 40.0
