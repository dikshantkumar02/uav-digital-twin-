"""PHASE 1 tests — project skeleton + configuration.

These tests do NOT need NumPy, Pandas or scikit-learn. They cover:
* YAML files exist and parse
* Pydantic validation succeeds
* Required fields and provenance labels are present
* Loader is deterministic and re-entrant
* CLI runs in --validate-only mode
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.config import (
    ConfigError,
    LoadedConfig,
    Provenance,
    get_default_config,
    load_config,
    reload_config,
)
from backend.config.cli import main as cli_main

pytestmark = pytest.mark.phase1


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def test_required_yaml_files_exist() -> None:
    for name in ("engine", "environment", "sensors", "faults"):
        assert (CONFIG_DIR / f"{name}.yaml").is_file(), f"missing {name}.yaml"


def test_load_config_returns_loaded_config() -> None:
    cfg = load_config(CONFIG_DIR)
    assert isinstance(cfg, LoadedConfig)
    assert cfg.engine.model_id == "REPR-OPP-180"
    assert cfg.environment.atmosphere.model == "ISA1976"
    assert "rpm" in cfg.sensors
    assert "ENGINE_DEGRADATION" in cfg.faults


def test_engine_maps_have_correct_shape() -> None:
    cfg = load_config(CONFIG_DIR)
    perf = cfg.engine.performance_maps
    rpm_axis = perf["bsfc_g_per_kwh"]["rpm_axis"]
    map_axis = perf["bsfc_g_per_kwh"]["map_axis"]
    for table in ("bsfc_g_per_kwh", "egt_map_c", "cht_map_c"):
        rows = perf[table]["values"]
        assert len(rows) == len(rpm_axis)
        for row in rows:
            assert len(row) == len(map_axis)


def test_provenance_default_is_set_on_engine() -> None:
    cfg = load_config(CONFIG_DIR)
    assert cfg.engine.provenance_default == Provenance.SYNTHETIC


def test_limits_are_self_consistent() -> None:
    cfg = load_config(CONFIG_DIR)
    lim = cfg.engine.limits
    assert lim.oil_temp_min_c < lim.oil_temp_max_c
    assert lim.oil_pressure_min_psi < lim.oil_pressure_max_psi
    assert lim.egt_max_c > 0
    assert lim.cht_max_c > 0


def test_sensors_have_required_fields_and_probs_in_unit_interval() -> None:
    cfg = load_config(CONFIG_DIR)
    for name, s in cfg.sensors.items():
        assert s.sample_rate_hz > 0
        for p in (s.dropout_prob, s.spike_prob, s.stuck_prob, s.calibration_error):
            assert 0.0 <= p <= 1.0, f"sensor {name} probability out of range"


def test_faults_have_engine_degradation_entry() -> None:
    cfg = load_config(CONFIG_DIR)
    degradation = cfg.faults["ENGINE_DEGRADATION"]
    assert degradation.enabled is True
    assert "bsfc_multiplier" in degradation.parameters["effects"]


def test_mission_profile_is_monotonic_in_time() -> None:
    cfg = load_config(CONFIG_DIR)
    profile = cfg.environment.mission.profile
    for a, b in zip(profile, profile[1:]):
        assert b.t_s > a.t_s, "mission profile waypoints must be strictly increasing in time"


def test_extra_keys_rejected() -> None:
    """Pydantic must reject unknown keys (extra='forbid' in the schema)."""
    import yaml

    bad = {
        "engine": {
            "model_id": "X",
            "description": "Y",
            "provenance_default": "SYNTHETIC",
            "this_key_does_not_exist": 1,  # forbidden
            "geometry": {
                "cylinders": 4,
                "displacement_l": 3.6,
                "compression_ratio": 8.5,
                "bore_mm": 95.2,
                "stroke_mm": 95.2,
                "redline_rpm": 2700,
                "idle_rpm": 650,
                "rated_rpm": 2400,
                "rated_power_kw": 134,
            },
            "performance_maps": {
                "bsfc_g_per_kwh": {
                    "rpm_axis": [1000, 2000],
                    "map_axis": [20.0, 30.0],
                    "values": [[400, 400], [400, 400]],
                },
                "egt_map_c": {
                    "rpm_axis": [1000, 2000],
                    "map_axis": [20.0, 30.0],
                    "values": [[500, 500], [500, 500]],
                },
                "cht_map_c": {
                    "rpm_axis": [1000, 2000],
                    "map_axis": [20.0, 30.0],
                    "values": [[200, 200], [200, 200]],
                },
            },
            "throttle_to_map": {
                "gain_inhg_per_unit": 1.0,
                "idle_map_inhg": 12.0,
                "max_map_inhg": 29.92,
                "dead_zone": 0.02,
            },
            "dynamics": {
                "rpm_time_constant_s": 1.5,
                "thermal_time_constant_s": 90.0,
                "oil_time_constant_s": 60.0,
            },
            "limits": {
                "egt_max_c": 870,
                "cht_max_c": 260,
                "oil_temp_min_c": 40,
                "oil_temp_max_c": 120,
                "oil_pressure_min_psi": 25,
                "oil_pressure_max_psi": 100,
                "vibration_rms_max_g": 6.0,
            },
            "lubrication": {
                "oil_capacity_l": 7.5,
                "nominal_pressure_psi": 60,
                "pressure_rpm_slope": 0.02,
                "pressure_temp_drop_per_c": 0.3,
            },
            "degradation": {
                "base_wear_per_hour": 1e-5,
                "wear_temp_factor_per_c": 2e-6,
                "wear_rpm_factor_per_rpm": 1e-9,
                "max_wear": 1.0,
            },
        },
        "environment": yaml.safe_load((CONFIG_DIR / "environment.yaml").read_text()),
        "sensors": yaml.safe_load((CONFIG_DIR / "sensors.yaml").read_text()),
        "faults": yaml.safe_load((CONFIG_DIR / "faults.yaml").read_text()),
    }
    with pytest.raises(Exception):
        from backend.config.schemas import AppConfig

        AppConfig.model_validate(bad)


def test_loader_is_memoised() -> None:
    a = get_default_config()
    b = get_default_config()
    assert a is b
    c = reload_config()
    # After reload, the in-memory object is replaced (different identity).
    assert c is not a


def test_load_raises_on_missing_dir() -> None:
    with pytest.raises(ConfigError):
        load_config("/no/such/dir")


def test_cli_validate_only_exits_zero() -> None:
    rc = cli_main(["--config", str(CONFIG_DIR), "--validate-only"])
    assert rc == 0


def test_cli_runs_via_subprocess() -> None:
    # Smoke test: the entry point must at least import and exit cleanly.
    result = subprocess.run(
        [sys.executable, "-m", "backend.config.cli", "--config", str(CONFIG_DIR), "--validate-only"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_environment_yaml_contains_disturbances() -> None:
    cfg = load_config(CONFIG_DIR)
    assert cfg.environment.disturbances.turbulence.enabled is True
    assert cfg.environment.disturbances.gusts.enabled is True


def test_config_serialisable_to_json_for_dashboard() -> None:
    """The dashboard later needs a JSON dump of the engine metadata."""
    cfg = load_config(CONFIG_DIR)
    blob = json.dumps(
        {
            "model_id": cfg.engine.model_id,
            "rated_rpm": cfg.engine.geometry.rated_rpm,
            "rated_power_kw": cfg.engine.geometry.rated_power_kw,
        }
    )
    assert "REPR-OPP-180" in blob
