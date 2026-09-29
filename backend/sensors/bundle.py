"""
Sensor bundle — the single object the rest of the system reads from.

The bundle owns one :class:`Channel` per configured sensor. On each
``tick()`` it advances the simulator by one step and produces a
:class:`SensorSample` containing:

* a per-channel reading (float or ``None`` for dropout)
* the per-channel noise mode at this tick
* the ground-truth engine + environment state (so the Digital Twin
  can later compare observation vs. truth when training/evaluating)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.config import SensorsConfig
from backend.environment import EnvironmentState
from backend.simulation import EngineSimulator, EngineState, SimulatorStep

from .channels import Channel, SENSOR_CHANNELS, build_channel
from .noise import NoiseMode


# ---------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------
@dataclass
class SensorReading:
    """One sensor's reading at one tick."""

    value: Optional[float]   # None == dropout
    mode: NoiseMode


@dataclass
class SensorSample:
    """A single tick of the sensor bundle."""

    time_s: float
    readings: Dict[str, SensorReading] = field(default_factory=dict)
    engine: Optional[EngineState] = None
    env: Optional[EnvironmentState] = None

    def get(self, channel: str) -> Optional[float]:
        r = self.readings.get(channel)
        return None if r is None else r.value

    def mode(self, channel: str) -> Optional[NoiseMode]:
        r = self.readings.get(channel)
        return None if r is None else r.mode

    def is_dropped(self, channel: str) -> bool:
        r = self.readings.get(channel)
        return r is not None and r.value is None

    def is_stuck(self, channel: str) -> bool:
        r = self.readings.get(channel)
        return r is not None and r.mode == NoiseMode.STUCK

    def to_dict(self) -> dict:
        out = {"time_s": self.time_s}
        for name, r in self.readings.items():
            out[f"sensor.{name}"] = r.value
            out[f"sensor.{name}.mode"] = r.mode.value
        if self.engine is not None:
            for k, v in self.engine.to_dict().items():
                if k != "time_s":
                    out[f"truth.{k}"] = v
        if self.env is not None:
            for k, v in self.env.to_dict().items():
                if k != "time_s":
                    out[f"truth.env.{k}"] = v
        return out


# ---------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------
class SensorBundle:
    """Virtual sensor bundle over a unified simulator."""

    def __init__(
        self,
        sensors_cfg: SensorsConfig,
        simulator: EngineSimulator,
        master_seed: int = 12345,
    ) -> None:
        self._sim = simulator
        self._master_seed = int(master_seed)
        self._channels: Dict[str, Channel] = {
            name: build_channel(name, sensors_cfg[name], self._master_seed)
            for name in SENSOR_CHANNELS
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def channels(self) -> Dict[str, Channel]:
        return self._channels

    def channel(self, name: str) -> Channel:
        return self._channels[name]

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def tick(
        self,
        degradation_severity: float = 0.0,
        vibration_external: float = 0.0,
    ) -> SensorSample:
        """Advance the simulator by one step and read every channel."""
        step = self._sim.step(
            degradation_severity=degradation_severity,
            vibration_external=vibration_external,
        )
        return self._read_step(step)

    def run(
        self,
        degradation_severity: float = 0.0,
        vibration_external: Optional[List[float]] = None,
    ) -> List[SensorSample]:
        """Run the full mission, returning one sample per simulator step."""
        from backend.simulation import EngineInputs

        self._sim.reset()
        env_states = self._sim.runner.run()
        if vibration_external is not None and len(vibration_external) != len(env_states):
            raise ValueError("vibration_external length must match number of env states")
        samples: List[SensorSample] = []
        self._sim.engine.reset()
        for i, env in enumerate(env_states):
            ve = 0.0 if vibration_external is None else float(vibration_external[i])
            es = self._sim.engine.step(
                EngineInputs(env=env, degradation_severity=degradation_severity, vibration_external=ve)
            )
            step = SimulatorStep(env=env, engine=es)
            samples.append(self._read_step(step))
        return samples

    def _read_step(self, step: SimulatorStep) -> SensorSample:
        sample = SensorSample(
            time_s=step.env.time_s,
            engine=step.engine,
            env=step.env,
        )
        dt = self._sim.dt_s
        for name, ch in self._channels.items():
            value, mode = ch.read(step.engine, step.env, step.env.time_s, dt)
            sample.readings[name] = SensorReading(value=value, mode=mode)
        return sample

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------
    def inject_fault(self, channel: str, mode: NoiseMode, start_t: Optional[float] = None) -> None:
        """Inject a fault on one channel. See :class:`Channel.inject_fault`."""
        self._channels[channel].inject_fault(mode, start_t or 0.0)

    def clear_fault(self, channel: Optional[str] = None) -> None:
        if channel is None:
            for ch in self._channels.values():
                ch.clear_fault()
        else:
            self._channels[channel].clear_fault()

    def reset(self) -> None:
        for ch in self._channels.values():
            ch.reset()
        self._sim.reset()


__all__ = ["SensorBundle", "SensorReading", "SensorSample"]
