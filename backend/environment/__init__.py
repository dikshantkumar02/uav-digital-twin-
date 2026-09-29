"""
Environment package — atmosphere + disturbances.

Public re-exports for the rest of the system::

    from backend.environment import Atmosphere, TurbulenceModel, GustModel
    from backend.environment import MissionProfile, EnvironmentState
    # PHASE 20 — mission profile generator
    from backend.environment import (
        MissionTemplate, MissionPhaseSpec, PhaseKind,
        ProfileGenerator, ProfileGeneratorConfig,
    )
"""

from .atmosphere import Atmosphere, AtmosphericState
from .disturbances import GustEvent, GustModel, TurbulenceModel
from .mission import (
    EnvironmentState,
    FaultTruthRecord,
    MissionProfile,
    MissionRunner,
    MissionTick,
    MissionTrace,
    ObservedTick,
)
from .phases import MissionPhaseSpec, PhaseKind
from .profile_generator import ProfileGenerator, ProfileGeneratorConfig
from .state import WindState
from .templates import MISSION_TEMPLATES, MissionTemplate, get_template, list_templates

__all__ = [
    "Atmosphere",
    "AtmosphericState",
    "GustEvent",
    "GustModel",
    "MISSION_TEMPLATES",
    "MissionPhaseSpec",
    "MissionProfile",
    "MissionRunner",
    "MissionTemplate",
    "MissionTick",
    "MissionTrace",
    "ObservedTick",
    "PhaseKind",
    "ProfileGenerator",
    "ProfileGeneratorConfig",
    "FaultTruthRecord",
    "TurbulenceModel",
    "WindState",
    "EnvironmentState",
    "get_template",
    "list_templates",
]
