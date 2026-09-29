"""
Configuration package.

Public re-exports so callers can write::

    from backend.config import load_config, get_default_config
"""

from .loader import (
    ConfigError,
    LoadedConfig,
    get_default_config,
    load_config,
    load_dashboard_config,
    load_hardware_config,
    load_hil_config,
    reload_config,
)
from .schemas import (
    AppConfig,
    Atmosphere,
    ChtMap,
    DashboardConfig,
    EngineConfig,
    EnvironmentConfig,
    EgtMap,
    FaultEntry,
    FaultsConfig,
    GustsCfg,
    HardwareConfig,
    HilConfig,
    HilReplayConfig,
    HilToleranceConfig,
    Mission,
    Provenance,
    RiskConfig,
    SeaLevel,
    SensorConfig,
    SensorsConfig,
    TurbulenceCfg,
    Waypoint,
)

__all__ = [
    "AppConfig",
    "Atmosphere",
    "ChtMap",
    "ConfigError",
    "DashboardConfig",
    "EngineConfig",
    "EnvironmentConfig",
    "EgtMap",
    "FaultEntry",
    "FaultsConfig",
    "GustsCfg",
    "HardwareConfig",
    "HilConfig",
    "HilReplayConfig",
    "HilToleranceConfig",
    "LoadedConfig",
    "Mission",
    "Provenance",
    "RiskConfig",
    "SeaLevel",
    "SensorConfig",
    "SensorsConfig",
    "TurbulenceCfg",
    "Waypoint",
    "get_default_config",
    "load_config",
    "load_dashboard_config",
    "load_hardware_config",
    "load_hil_config",
    "reload_config",
]
