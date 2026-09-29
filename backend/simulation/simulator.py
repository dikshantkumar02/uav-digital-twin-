"""
Unified simulator: mission runner + engine.

This is the single object the rest of the system uses to advance
"one tick" — it composes the environment and the engine into one
``SimulatorStep`` and offers a step-by-step API as well as a
batch ``run()`` that returns a list of steps for analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from backend.config import EngineConfig, EnvironmentConfig
from backend.environment import EnvironmentState, MissionRunner

from .engine import Engine, EngineInputs, EngineState


@dataclass(frozen=True)
class SimulatorStep:
    """One tick of the unified simulator."""

    env: EnvironmentState
    engine: EngineState

    def to_dict(self) -> dict:
        d = self.engine.to_dict()
        # Add environment fields with a prefix to avoid clashes.
        d["env_time_s"] = self.env.time_s
        d["env_pressure_pa"] = self.env.atmosphere.pressure_pa
        d["env_temperature_c"] = self.env.atmosphere.temperature_c
        d["env_density_kg_per_m3"] = self.env.atmosphere.density_kg_per_m3
        d["env_wind_total_mps"] = self.env.wind.total_w_mps
        d["env_vertical_accel_mps2"] = self.env.vertical_accel_mps2
        return d


class EngineSimulator:
    """Drives the engine from a mission profile."""

    def __init__(
        self,
        engine_cfg: EngineConfig,
        env_cfg: EnvironmentConfig,
        dt_s: float = 0.1,
    ) -> None:
        self._engine = Engine(engine_cfg)
        self._engine.set_dt(dt_s)
        self._runner = MissionRunner(env_cfg, dt_s=dt_s)
        # Provide a sea-level reference density for altitude corrections.
        self._engine.set_reference_density(env_cfg.atmosphere.sea_level.density_kg_per_m3)

    # ------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def runner(self) -> MissionRunner:
        return self._runner

    @property
    def dt_s(self) -> float:
        return self._engine.dt_s

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._engine.reset()
        self._runner.reset()

    def step(
        self,
        degradation_severity: float = 0.0,
        vibration_external: float = 0.0,
    ) -> SimulatorStep:
        env = self._runner.step()
        es = self._engine.step(
            EngineInputs(
                env=env,
                degradation_severity=degradation_severity,
                vibration_external=vibration_external,
            )
        )
        return SimulatorStep(env=env, engine=es)

    def run(
        self,
        degradation_severity: float = 0.0,
        vibration_external: Optional[List[float]] = None,
    ) -> List[SimulatorStep]:
        """Run a full mission and return every step."""
        self.reset()
        steps: List[SimulatorStep] = []
        # We rely on the runner's `run()` to keep things deterministic.
        env_states = self._runner.run()
        if vibration_external is not None and len(vibration_external) != len(env_states):
            raise ValueError("vibration_external length must match number of env states")
        for i, env in enumerate(env_states):
            ve = 0.0 if vibration_external is None else float(vibration_external[i])
            es = self._engine.step(
                EngineInputs(
                    env=env,
                    degradation_severity=degradation_severity,
                    vibration_external=ve,
                )
            )
            steps.append(SimulatorStep(env=env, engine=es))
        return steps


__all__ = ["EngineSimulator", "SimulatorStep"]
