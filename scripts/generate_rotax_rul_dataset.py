"""Rotax 912 High-Fidelity Synthetic RUL Dataset Generator for MALE UAV Digital Twin.

Generates 50,000 1-Hz records across 120+ missions with Rotax 912 physics envelopes,
nonlinear RUL degradation modeling, and full train/val/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "rotax_912_rul_dataset.csv"

TOTAL_ROWS = 50_000
NUM_MISSIONS = 135
NOMINAL_OVERHAUL_HOURS = 2000.0

PHASES = ["Takeoff", "Climb", "Cruise", "Loiter", "Descent", "Landing"]
MISSION_TYPES = ["ISR", "Maritime", "Endurance", "Communication Relay"]

FAULT_TYPES = [
    "Normal",
    "Misfire",
    "Injector abnormality",
    "Cooling degradation",
    "Lubrication issue",
    "Sensor drift",
    "Combustion instability",
    "Abnormal vibration",
    "Overheating",
]

def generate():
    rng = np.random.default_rng(seed=42)

    # Allocate mission distributions
    # 50,000 seconds total distributed across 135 missions (averaging ~370s simulated telemetry windows from 2-12h flights)
    rows_per_mission = rng.multinomial(TOTAL_ROWS, [1.0 / NUM_MISSIONS] * NUM_MISSIONS)
    
    records = []
    
    # Mission types and baseline environmental profiles per mission
    env_profiles = [
        {"temp_base": 15.0, "alt_max": 4500, "hum_base": 50, "scenario": "standard"},
        {"temp_base": 42.0, "alt_max": 3000, "hum_base": 20, "scenario": "hot_desert"},
        {"temp_base": -8.0, "alt_max": 6000, "hum_base": 65, "scenario": "cold_high_alt"},
        {"temp_base": 28.0, "alt_max": 2500, "hum_base": 88, "scenario": "maritime"},
        {"temp_base": 35.0, "alt_max": 4000, "hum_base": 30, "scenario": "desert_dust"},
    ]

    global_second = 0

    for m_idx in range(NUM_MISSIONS):
        m_id = f"MSN-{m_idx + 1:04d}"
        n_pts = rows_per_mission[m_idx]
        if n_pts == 0:
            continue
            
        m_type = rng.choice(MISSION_TYPES)
        profile = env_profiles[m_idx % len(env_profiles)]
        
        # Base engine age between 50h and 1950h
        base_age_hours = rng.uniform(50.0, 1950.0)
        cycles = int(base_age_hours * rng.uniform(0.8, 1.2))
        
        # Decide if this mission has a fault
        has_fault = rng.random() < 0.38
        m_fault = rng.choice(FAULT_TYPES[1:]) if has_fault else "Normal"
        m_severity = rng.uniform(0.35, 1.0) if has_fault else 0.0
        
        # Generate timeline fractions
        t_frac = np.linspace(0, 1, n_pts)
        
        # Phase boundaries: Takeoff (0-0.08), Climb (0.08-0.25), Cruise (0.25-0.65), Loiter (0.65-0.85), Descent (0.85-0.95), Landing (0.95-1.0)
        phase_labels = []
        for f in t_frac:
            if f < 0.08:
                phase_labels.append("Takeoff")
            elif f < 0.25:
                phase_labels.append("Climb")
            elif f < 0.65:
                phase_labels.append("Cruise")
            elif f < 0.85:
                phase_labels.append("Loiter")
            elif f < 0.95:
                phase_labels.append("Descent")
            else:
                phase_labels.append("Landing")
                
        for i, (f, phase) in enumerate(zip(t_frac, phase_labels)):
            global_second += 1
            timestamp = f"2026-09-30T{global_second // 3600:02d}:{(global_second % 3600) // 60:02d}:{global_second % 60:02d}Z"
            
            # Mission Phase Physics & Throttle Profiles
            if phase == "Takeoff":
                throttle = rng.uniform(92.0, 100.0)
                rpm = rng.uniform(5400.0, 5800.0)
                load = rng.uniform(90.0, 100.0)
                alt = rng.uniform(0.0, 300.0)
            elif phase == "Climb":
                throttle = rng.uniform(78.0, 92.0)
                rpm = rng.uniform(5100.0, 5500.0)
                load = rng.uniform(80.0, 92.0)
                alt = 300.0 + f * (profile["alt_max"] - 300.0)
            elif phase == "Cruise":
                throttle = rng.uniform(62.0, 75.0)
                rpm = rng.uniform(4700.0, 5100.0)
                load = rng.uniform(65.0, 78.0)
                alt = profile["alt_max"] + rng.normal(0, 20.0)
            elif phase == "Loiter":
                throttle = rng.uniform(48.0, 60.0)
                rpm = rng.uniform(4000.0, 4600.0)
                load = rng.uniform(48.0, 62.0)
                alt = profile["alt_max"] * 0.8 + rng.normal(0, 15.0)
            elif phase == "Descent":
                throttle = rng.uniform(22.0, 40.0)
                rpm = rng.uniform(2800.0, 3800.0)
                load = rng.uniform(25.0, 45.0)
                alt = max(50.0, profile["alt_max"] * (1.0 - (f - 0.85) / 0.10))
            else: # Landing
                throttle = rng.uniform(10.0, 22.0)
                rpm = rng.uniform(1400.0, 2200.0)
                load = rng.uniform(12.0, 25.0)
                alt = max(0.0, 50.0 * (1.0 - (f - 0.95) / 0.05))

            # Environmental telemetry
            lapse = 0.0065 * alt
            ambient_temp = profile["temp_base"] - lapse + rng.normal(0, 0.4)
            humidity = np.clip(profile["hum_base"] + rng.normal(0, 1.5), 10.0, 98.0)
            p_ambient = 101.325 * (1.0 - 2.25577e-5 * alt) ** 5.25588
            air_density = max(0.5, (p_ambient * 1000.0) / (287.05 * (ambient_temp + 273.15)))
            wind_speed = max(0.0, rng.uniform(2.0, 14.0) + rng.normal(0, 0.5))

            # Engine core telemetry (Rotax 912 physical characteristics)
            map_kpa = np.clip(38.0 + (throttle / 100.0) * (p_ambient - 38.0) + rng.normal(0, 0.3), 38.0, 101.5)
            cht = np.clip(92.0 + (load / 100.0) * 55.0 + (ambient_temp - 15.0) * 0.35 + rng.normal(0, 1.0), 90.0, 178.0)
            egt = np.clip(660.0 + (load / 100.0) * 190.0 + rng.normal(0, 3.0), 650.0, 895.0)
            oil_temp = np.clip(86.0 + (load / 100.0) * 32.0 + (ambient_temp - 15.0) * 0.25 + rng.normal(0, 0.6), 85.0, 128.0)
            oil_press = np.clip(2.0 + (rpm / 5800.0) * 2.8 - (oil_temp - 85.0) * 0.012 + rng.normal(0, 0.04), 2.0, 5.0)
            fuel_flow = np.clip(4.2 + (rpm / 5800.0) * (throttle / 100.0) * 22.0 + rng.normal(0, 0.2), 4.0, 27.8)
            vibration = np.clip(0.04 + (rpm / 5800.0) ** 1.8 * 0.95 + rng.normal(0, 0.02), 0.02, 1.75)
            batt_volt = np.clip(13.8 + rng.normal(0, 0.15) - (0.4 if rpm < 1800 else 0.0), 12.8, 14.4)
            alt_current = np.clip(8.0 + (throttle / 100.0) * 18.0 + rng.normal(0, 0.5), 5.0, 32.0)
            inj_timing = np.clip(24.0 + (rpm / 5800.0) * 8.0 - (throttle / 100.0) * 4.0 + rng.normal(0, 0.2), 20.0, 34.0)

            # Fault injection shifts
            active_fault = "Normal"
            active_severity = 0.0
            is_anomaly = 0
            
            # Fault activates during cruise / loiter / climb after onset
            if has_fault and f >= 0.20:
                active_fault = m_fault
                active_severity = m_severity
                is_anomaly = 1
                
                if m_fault == "Misfire":
                    rpm -= 250.0 * active_severity
                    vibration += 0.45 * active_severity
                    egt -= 95.0 * active_severity
                elif m_fault == "Injector abnormality":
                    fuel_flow += 3.8 * active_severity
                    egt += 65.0 * active_severity
                elif m_fault == "Cooling degradation":
                    cht += 28.0 * active_severity
                    oil_temp += 12.0 * active_severity
                elif m_fault == "Lubrication issue":
                    oil_press -= 1.4 * active_severity
                    oil_temp += 18.0 * active_severity
                elif m_fault == "Sensor drift":
                    cht += 22.0 * active_severity
                elif m_fault == "Combustion instability":
                    vibration += 0.35 * active_severity
                    egt += rng.normal(0, 25.0) * active_severity
                elif m_fault == "Abnormal vibration":
                    vibration += 0.70 * active_severity
                elif m_fault == "Overheating":
                    cht += 35.0 * active_severity
                    oil_temp += 20.0 * active_severity
                    egt += 40.0 * active_severity

            # Clamp after fault modification
            rpm = np.clip(rpm, 1400.0, 5800.0)
            cht = np.clip(cht, 90.0, 180.0)
            egt = np.clip(egt, 650.0, 900.0)
            oil_temp = np.clip(oil_temp, 85.0, 130.0)
            oil_press = np.clip(oil_press, 2.0, 5.0)
            fuel_flow = np.clip(fuel_flow, 4.0, 28.0)
            vibration = np.clip(vibration, 0.02, 1.8)

            # Digital Twin Estimator Health Features [0, 1]
            thermal_eff = np.clip(0.38 - (cht - 120.0) * 0.0015 - (active_severity * 0.06 if active_fault in ["Overheating", "Misfire"] else 0.0), 0.18, 0.42)
            comb_idx = np.clip(0.95 - (active_severity * 0.38 if active_fault in ["Misfire", "Combustion instability", "Injector abnormality"] else 0.0) + rng.normal(0, 0.01), 0.20, 0.99)
            lub_health = np.clip(0.96 - (oil_temp - 95.0) * 0.006 - (active_severity * 0.45 if active_fault == "Lubrication issue" else 0.0) - (base_age_hours / 2000.0) * 0.15, 0.15, 0.99)
            vib_health = np.clip(0.98 - (vibration / 1.8) * 0.45 - (active_severity * 0.45 if active_fault in ["Abnormal vibration", "Misfire"] else 0.0), 0.10, 0.99)
            cool_eff = np.clip(0.94 - (cht - 115.0) * 0.007 - (active_severity * 0.50 if active_fault in ["Cooling degradation", "Overheating"] else 0.0), 0.15, 0.98)
            health_idx = np.clip(0.25 * comb_idx + 0.25 * lub_health + 0.25 * vib_health + 0.25 * cool_eff, 0.05, 0.99)

            # Physical Degradation Accrual
            current_age = base_age_hours + (i / 3600.0)
            wear_factor = np.clip((current_age / NOMINAL_OVERHAUL_HOURS) ** 1.35 + (1.0 - health_idx) * 0.35, 0.02, 0.99)
            carbon_dep = np.clip(0.08 + (current_age / NOMINAL_OVERHAUL_HOURS) * 0.65 + (0.20 if active_fault in ["Injector abnormality", "Misfire"] else 0.0), 0.05, 0.98)
            bearing_wear = np.clip(0.05 + (current_age / NOMINAL_OVERHAUL_HOURS) * 0.70 + (0.25 if active_fault in ["Lubrication issue", "Abnormal vibration"] else 0.0), 0.03, 0.99)

            # Nonlinear Condition-Based RUL Calculation
            # Baseline remaining life: 2000 - age
            baseline_rul = max(0.0, NOMINAL_OVERHAUL_HOURS - current_age)
            # Condition penalty factor based on combined health and active fault severity
            condition_mult = (health_idx ** 1.8) * (1.0 - 0.75 * active_severity)
            rul_hours = np.clip(baseline_rul * condition_mult + rng.normal(0, 12.0), 0.0, 2000.0)
            
            # Map category thresholds as specified in prompt
            if is_anomaly:
                if active_severity > 0.7:
                    rul_hours = min(rul_hours, rng.uniform(15.0, 145.0)) # Critical
                else:
                    rul_hours = min(rul_hours, rng.uniform(450.0, 1150.0)) # Moderate

            conf_pct = np.clip(94.0 - active_severity * 22.0 - (current_age / 2000.0) * 15.0 + rng.normal(0, 1.0), 45.0, 99.0)

            # Maintenance Recommendation
            if rul_hours < 150.0:
                rec = "CRITICAL: Immediate abort / RTB and overhaul inspection required."
            elif rul_hours < 600.0:
                rec = "MAINTENANCE DUE: Schedule borescope, oil filter check, and injector bench test."
            elif rul_hours < 1200.0:
                rec = "MONITOR: Advisory trend observed. Inspect lubrication & cooling margins post-flight."
            else:
                rec = "NORMAL: System healthy. Standard 100-hour scheduled inspection interval applies."

            records.append({
                "mission_id": m_id,
                "timestamp": timestamp,
                "mission_phase": phase,
                "mission_type": m_type,
                "ambient_temperature_C": round(ambient_temp, 2),
                "altitude_m": round(alt, 1),
                "humidity_percent": round(humidity, 1),
                "air_density": round(air_density, 4),
                "wind_speed_mps": round(wind_speed, 2),
                "rpm": round(rpm, 1),
                "manifold_pressure_kPa": round(map_kpa, 2),
                "throttle_percent": round(throttle, 1),
                "cht_C": round(cht, 1),
                "egt_C": round(egt, 1),
                "oil_temp_C": round(oil_temp, 1),
                "oil_pressure_bar": round(oil_press, 2),
                "fuel_flow_Lph": round(fuel_flow, 2),
                "battery_voltage": round(batt_volt, 2),
                "alternator_current_A": round(alt_current, 1),
                "vibration_rms": round(vibration, 4),
                "injection_timing_deg": round(inj_timing, 1),
                "engine_load_percent": round(load, 1),
                "thermal_efficiency": round(thermal_eff, 4),
                "combustion_index": round(comb_idx, 4),
                "lubrication_health": round(lub_health, 4),
                "vibration_health": round(vib_health, 4),
                "cooling_efficiency": round(cool_eff, 4),
                "health_index": round(health_idx, 4),
                "engine_age_hours": round(current_age, 2),
                "cumulative_cycles": cycles,
                "wear_factor": round(wear_factor, 4),
                "carbon_deposit_index": round(carbon_dep, 4),
                "bearing_wear_index": round(bearing_wear, 4),
                "anomaly": is_anomaly,
                "fault_type": active_fault,
                "fault_severity": round(active_severity, 2),
                "rul_hours": round(rul_hours, 1),
                "confidence_percent": round(conf_pct, 1),
                "maintenance_recommendation": rec,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {len(df)} records across {NUM_MISSIONS} missions to {OUTPUT_CSV}")

    # Generate Train (70%) / Validation (15%) / Test (15%) split by mission_id (preventing mission data leakage)
    unique_missions = df["mission_id"].unique()
    rng.shuffle(unique_missions)
    n_train = int(len(unique_missions) * 0.70)
    n_val = int(len(unique_missions) * 0.15)
    
    train_m = set(unique_missions[:n_train])
    val_m = set(unique_missions[n_train:n_train + n_val])
    test_m = set(unique_missions[n_train + n_val:])
    
    df_train = df[df["mission_id"].isin(train_m)]
    df_val = df[df["mission_id"].isin(val_m)]
    df_test = df[df["mission_id"].isin(test_m)]
    
    df_train.to_csv(DATA_DIR / "rotax_912_rul_train.csv", index=False)
    df_val.to_csv(DATA_DIR / "rotax_912_rul_val.csv", index=False)
    df_test.to_csv(DATA_DIR / "rotax_912_rul_test.csv", index=False)
    
    print(f"Splits saved: Train={len(df_train)} rows, Val={len(df_val)} rows, Test={len(df_test)} rows")

    # Generate summary stats & correlation
    numeric_cols = [
        "rpm", "manifold_pressure_kPa", "throttle_percent", "cht_C", "egt_C",
        "oil_temp_C", "oil_pressure_bar", "fuel_flow_Lph", "vibration_rms",
        "health_index", "engine_age_hours", "wear_factor", "rul_hours"
    ]
    summary = df[numeric_cols].describe().round(2).to_dict()
    corr = df[numeric_cols].corr().round(3).to_dict()
    
    with open(DATA_DIR / "rotax_912_dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "correlation": corr}, fh, indent=2)
    print("Dataset stats & correlation exported.")

if __name__ == "__main__":
    generate()
