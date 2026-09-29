"""
Faults package — declarative fault scenarios + per-tick injection.

Public re-exports::

    from backend.faults import (
        FaultClass, FaultScenario, FaultInjector, FaultTick,
        SensorFaultPlan, SeverityPlan, severity_at,
    )
    # PHASE 21 — fault-injection framework
    from backend.faults import (
        FaultCategory, FaultType, FaultTypeMeta, FAULT_TAXONOMY,
        FaultRecord, FaultScenarioBundle, ScenarioGenerator,
    )
"""

from .framework import (
    DEFAULT_SEVERITY_LEVELS,
    DEFAULT_TEMPORAL_PATTERNS,
    FaultRecord,
    FaultScenarioBundle,
    ScenarioGenerator,
)
from .injector import FaultInjector, FaultTick
from .plan import FaultClass, FaultScenario, SensorFaultPlan, SeverityPlan
from .progression import severity_at
from .taxonomy import (
    FAULT_TAXONOMY,
    FaultCategory,
    FaultType,
    FaultTypeMeta,
    by_category,
    by_class,
    get_meta,
)

__all__ = [
    "DEFAULT_SEVERITY_LEVELS",
    "DEFAULT_TEMPORAL_PATTERNS",
    "FAULT_TAXONOMY",
    "FaultCategory",
    "FaultClass",
    "FaultInjector",
    "FaultRecord",
    "FaultScenario",
    "FaultScenarioBundle",
    "FaultTick",
    "FaultType",
    "FaultTypeMeta",
    "ScenarioGenerator",
    "SensorFaultPlan",
    "SeverityPlan",
    "by_category",
    "by_class",
    "get_meta",
    "severity_at",
]
