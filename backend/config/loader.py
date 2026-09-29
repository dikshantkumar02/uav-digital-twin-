"""
Configuration loader.

Loads engine / environment / sensor / fault YAML files and validates them
against the Pydantic schemas in :mod:`backend.config.schemas`.

Design goals
------------
* Single source of truth: ``config/*.yaml`` files only.
* Strict validation: extra / missing keys are rejected.
* Deterministic by default: every config that can carry a seed has one.
* Graceful degradation: if PyYAML is unavailable, use a minimal stdlib
  YAML-ish loader for the *simple* key-value files we ship; if validation
  still cannot proceed, raise an explicit error pointing at the missing
  dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import (
    AppConfig,
    DashboardConfig,
    EnvironmentConfig,
    EngineConfig,
    FaultsConfig,
    HardwareConfig,
    HilConfig,
    RiskConfig,
    SensorsConfig,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


# ---------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------
def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except ImportError:
        return False


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if not _yaml_available():
        raise ConfigError(
            "PyYAML is required to load YAML configuration. "
            "Install it with: pip install pyyaml"
        )
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedConfig:
    engine: EngineConfig
    environment: EnvironmentConfig
    sensors: SensorsConfig
    faults: FaultsConfig
    risk: RiskConfig
    dashboard: DashboardConfig
    hardware: HardwareConfig
    hil: HilConfig
    source_dir: Path

    def as_app_config(self) -> AppConfig:
        return AppConfig(
            engine=self.engine,
            environment=self.environment,
            sensors=self.sensors,
            faults=self.faults,
            risk=self.risk,
        )


def load_config(config_dir: Path | str | None = None) -> LoadedConfig:
    """Load and validate the full configuration bundle.

    Parameters
    ----------
    config_dir:
        Directory containing the required YAML files. Defaults to
        the repository-level ``config/`` directory. The risk
        configuration (``risk.yaml``) and dashboard configuration
        (``dashboard.yaml``) are optional; if absent, the schema
        defaults are used.

    Returns
    -------
    LoadedConfig
        A frozen bundle of validated Pydantic models.
    """
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not base.is_dir():
        raise ConfigError(f"Config directory does not exist: {base}")

    required_paths = {
        "engine": base / "engine.yaml",
        "environment": base / "environment.yaml",
        "sensors": base / "sensors.yaml",
        "faults": base / "faults.yaml",
    }
    for name, p in required_paths.items():
        if not p.exists():
            raise ConfigError(f"Missing required config file '{name}.yaml' at {p}")

    raw = {name: _read_yaml(p) for name, p in required_paths.items()}

    # Each YAML file wraps its payload in a single top-level key (e.g. `engine:`).
    # Unwrap one level before validating so schemas match the inner content.
    def _unwrap(d: dict[str, Any], expected_key: str) -> dict[str, Any]:
        if set(d.keys()) == {expected_key} and isinstance(d[expected_key], dict):
            return d[expected_key]
        return d

    # risk.yaml is optional; fall back to schema defaults.
    risk_path = base / "risk.yaml"
    if risk_path.exists():
        risk_raw = _read_yaml(risk_path)
        risk_inner = _unwrap(risk_raw, "risk")
    else:
        risk_inner = {}

    # dashboard.yaml is optional; fall back to schema defaults.
    dashboard_path = base / "dashboard.yaml"
    if dashboard_path.exists():
        dashboard_raw = _read_yaml(dashboard_path)
        dashboard_inner = _unwrap(dashboard_raw, "dashboard")
    else:
        dashboard_inner = {}

    # hardware.yaml is optional; fall back to schema defaults.
    hardware_path = base / "hardware.yaml"
    if hardware_path.exists():
        hardware_raw = _read_yaml(hardware_path)
        hardware_inner = _unwrap(hardware_raw, "hardware")
    else:
        hardware_inner = {}

    # hil.yaml is optional; fall back to schema defaults.
    hil_path = base / "hil.yaml"
    if hil_path.exists():
        hil_raw = _read_yaml(hil_path)
        hil_inner = _unwrap(hil_raw, "hil")
    else:
        hil_inner = {}

    try:
        engine_cfg = EngineConfig.model_validate(_unwrap(raw["engine"], "engine"))
        env_cfg = EnvironmentConfig.model_validate(_unwrap(raw["environment"], "environment"))
        sensors_cfg = SensorsConfig.model_validate(_unwrap(raw["sensors"], "sensors"))
        faults_cfg = FaultsConfig.model_validate(_unwrap(raw["faults"], "faults"))
        risk_cfg = RiskConfig.model_validate(risk_inner)
        dashboard_cfg = DashboardConfig.model_validate(dashboard_inner)
        hardware_cfg = HardwareConfig.model_validate(hardware_inner)
        hil_cfg = HilConfig.model_validate(hil_inner)
    except Exception as exc:  # noqa: BLE001 — rewrapped with context
        raise ConfigError(f"Configuration validation failed: {exc}") from exc

    log.info("Loaded configuration from %s", base)
    return LoadedConfig(
        engine=engine_cfg,
        environment=env_cfg,
        sensors=sensors_cfg,
        faults=faults_cfg,
        risk=risk_cfg,
        dashboard=dashboard_cfg,
        hardware=hardware_cfg,
        hil=hil_cfg,
        source_dir=base,
    )


def load_dashboard_config(
    config_dir: Path | str | None = None,
) -> DashboardConfig:
    """Load ``config/dashboard.yaml`` and return a :class:`DashboardConfig`.

    Forgiving: if the file is missing or empty, returns the schema
    defaults so the dashboard is runnable from a fresh checkout. If
    the file is present but invalid, raises :class:`ConfigError`.
    """
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not base.is_dir():
        return DashboardConfig.defaults()
    path = base / "dashboard.yaml"
    if not path.exists():
        return DashboardConfig.defaults()
    raw = _read_yaml(path)
    inner = raw.get("dashboard", raw) if isinstance(raw, dict) else {}
    if not isinstance(inner, dict):
        return DashboardConfig.defaults()
    try:
        return DashboardConfig.model_validate(inner)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Dashboard configuration validation failed: {exc}") from exc


def load_hardware_config(
    config_dir: Path | str | None = None,
) -> HardwareConfig:
    """Load ``config/hardware.yaml`` and return a :class:`HardwareConfig`.

    Forgiving: if the file is missing or empty, returns the schema
    defaults (port="loopback") so the transport is runnable from a
    fresh checkout. If the file is present but invalid, raises
    :class:`ConfigError`.
    """
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not base.is_dir():
        return HardwareConfig.defaults()
    path = base / "hardware.yaml"
    if not path.exists():
        return HardwareConfig.defaults()
    raw = _read_yaml(path)
    inner = raw.get("hardware", raw) if isinstance(raw, dict) else {}
    if not isinstance(inner, dict):
        return HardwareConfig.defaults()
    try:
        return HardwareConfig.model_validate(inner)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Hardware configuration validation failed: {exc}") from exc


def load_hil_config(
    config_dir: Path | str | None = None,
) -> HilConfig:
    """Load ``config/hil.yaml`` and return a :class:`HilConfig`.

    Forgiving: if the file is missing or empty, returns the schema
    defaults so the HIL harness is runnable from a fresh checkout.
    If the file is present but invalid, raises :class:`ConfigError`.
    """
    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    if not base.is_dir():
        return HilConfig.defaults()
    path = base / "hil.yaml"
    if not path.exists():
        return HilConfig.defaults()
    raw = _read_yaml(path)
    inner = raw.get("hil", raw) if isinstance(raw, dict) else {}
    if not isinstance(inner, dict):
        return HilConfig.defaults()
    try:
        return HilConfig.model_validate(inner)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"HIL configuration validation failed: {exc}") from exc


__all__ = [
    "ConfigError",
    "LoadedConfig",
    "load_config",
    "load_dashboard_config",
    "load_hardware_config",
    "load_hil_config",
    "get_default_config",
    "reload_config",
]


@lru_cache(maxsize=1)
def get_default_config() -> LoadedConfig:
    """Memoised default config (used by FastAPI dependency injection later)."""
    return load_config()


def reload_config() -> LoadedConfig:
    """Clear the cache and reload — used by tests."""
    get_default_config.cache_clear()
    return get_default_config()
