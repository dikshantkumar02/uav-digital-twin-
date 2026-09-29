"""
Simulation package — engine + (later) UAV.

Public re-exports::

    from backend.simulation import (
        Engine, EngineState, EngineInputs, DegradationState,
        EngineSimulator, SimulatorStep,
    )
"""

from .engine import (
    DegradationState,
    Engine,
    EngineInputs,
    EngineState,
)
from .interp import bilinear_interp, interp_axis, map_evaluator
from .simulator import EngineSimulator, SimulatorStep

__all__ = [
    "DegradationState",
    "Engine",
    "EngineInputs",
    "EngineSimulator",
    "EngineState",
    "SimulatorStep",
    "bilinear_interp",
    "interp_axis",
    "map_evaluator",
]
