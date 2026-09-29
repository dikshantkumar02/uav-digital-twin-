"""
Per-channel virtual sensor.

A channel owns:
* its own RNG (deterministic, derived from a master seed)
* its own drift accumulator
* its own stuck / fault state
* the per-sample cadence (down-sampled from the simulator's dt)

Channels are constructed via :func:`build_channel` from a
:class:`backend.config.SensorConfig`. The set of channels used by the
bundle is enumerated in :data:`SENSOR_CHANNELS` — a mapping from
channel name to a small callable that extracts the "true" value from
the current ``EngineState`` + ``EnvironmentState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from backend.config import SensorConfig
from backend.environment import EnvironmentState
from backend.simulation import EngineState

from .noise import (
    NoiseMode,
    apply_bias,
    apply_calibration_error,
    apply_dropout,
    apply_drift,
    apply_gaussian_noise,
    apply_spike,
    apply_stuck,
)


# ---------------------------------------------------------------------
# Truth extractors — what the channel measures in the (engine, env) state
# ---------------------------------------------------------------------
def _true_value(
    name: str,
    engine: EngineState,
    env: EnvironmentState,
) -> float:
    """Extract the *true* value of ``name`` from the engine + environment.

    Returns a float. Channels that conceptually measure environment
    variables ignore the engine state; engine channels ignore the env
    except for the altitude/airspeed channels.
    """
    if name == "rpm":
        return engine.rpm
    if name == "egt":
        return engine.egt_c
    if name == "cht":
        return engine.cht_c
    if name == "oil_pressure":
        return engine.oil_pressure_psi
    if name == "oil_temperature":
        return engine.oil_temperature_c
    if name == "fuel_flow":
        return engine.fuel_flow_lph
    if name == "vibration":
        return engine.vibration_rms_g
    if name == "imu_accel":
        return env.vertical_accel_mps2
    if name == "altitude":
        return env.altitude_m
    if name == "airspeed":
        return env.airspeed_mps
    if name == "ambient_temperature":
        return env.atmosphere.temperature_c
    if name == "ambient_pressure":
        return env.atmosphere.pressure_pa
    raise KeyError(f"unknown sensor channel: {name}")


# Mapping of channel name -> SensorConfig key (they happen to match for now).
SENSOR_CHANNELS: tuple[str, ...] = (
    "rpm",
    "egt",
    "cht",
    "oil_pressure",
    "oil_temperature",
    "fuel_flow",
    "vibration",
    "imu_accel",
    "altitude",
    "airspeed",
    "ambient_temperature",
    "ambient_pressure",
)


# ---------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------
@dataclass
class Channel:
    """A virtual sensor channel."""

    name: str
    cfg: SensorConfig
    rng: np.random.Generator
    drift_accumulator: float = 0.0
    stuck: bool = False
    fault_mode: NoiseMode = NoiseMode.NORMAL
    fault_started_t: Optional[float] = None
    last_value: Optional[float] = None
    last_value_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Configuration / injection
    # ------------------------------------------------------------------
    def inject_fault(self, mode: NoiseMode, start_t: float) -> None:
        """Externally inject a sensor fault."""
        self.fault_mode = mode
        self.fault_started_t = start_t
        if mode == NoiseMode.STUCK:
            self.stuck = True
        else:
            self.stuck = False

    def clear_fault(self) -> None:
        self.fault_mode = NoiseMode.NORMAL
        self.fault_started_t = None
        self.stuck = False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def read(
        self,
        engine: EngineState,
        env: EnvironmentState,
        t_s: float,
        dt_s: float,
    ) -> tuple[Optional[float], NoiseMode]:
        """Read the sensor at time ``t_s``.

        Returns (value, mode). ``value`` is ``None`` for a dropout. The
        mode reflects the dominant reason the sample was perturbed, so
        the diagnostic engine can later attribute it to a sensor fault.
        """
        cfg = self.cfg

        # Time-based drift integration: ``drift_per_hour`` is the slope,
        # the running accumulator is updated by ``drift * dt``.
        if cfg.drift_per_hour > 0.0:
            self.drift_accumulator += cfg.drift_per_hour * dt_s / 3600.0

        # STUCK is a hard override: ignore truth and all noise processes.
        if self.fault_mode == NoiseMode.STUCK or self.stuck:
            self.stuck = True
            self.last_value = float(cfg.stuck_value)
            self.last_value_time = t_s
            return float(cfg.stuck_value), NoiseMode.STUCK

        # Choose a base mode for this sample.
        mode = NoiseMode.NORMAL
        if self.fault_mode == NoiseMode.DRIFTING:
            mode = NoiseMode.DRIFTING
            # Boost drift when fault is active.
            self.drift_accumulator += 0.5 * dt_s
        elif self.fault_mode == NoiseMode.FAULT:
            mode = NoiseMode.FAULT

        true_v = _true_value(self.name, engine, env)
        v = float(true_v)
        v = apply_bias(v, cfg.bias)
        v = apply_drift(v, self.drift_accumulator)
        v = apply_calibration_error(v, cfg.calibration_error)

        # Spikes (with base probability or fault-burst).
        spike_prob = cfg.spike_prob
        spike_amp = 3.0 * float(cfg.noise_std + 1e-9)
        if mode == NoiseMode.FAULT:
            spike_prob = max(spike_prob, 0.05)
            spike_amp *= 4.0
        v = apply_spike(self.rng, v, spike_prob, spike_amp)

        # Dropouts (with base probability or fault-burst).
        dropout_prob = cfg.dropout_prob
        if mode == NoiseMode.FAULT:
            dropout_prob = max(dropout_prob, 0.10)
        v_out = apply_dropout(self.rng, v, dropout_prob)
        if v_out is None:
            mode = NoiseMode.DROPPED
            self.last_value = None
            self.last_value_time = t_s
            return None, mode

        # Gaussian noise.
        v_out = apply_gaussian_noise(self.rng, v_out, cfg.noise_std)

        self.last_value = float(v_out)
        self.last_value_time = t_s
        return float(v_out), mode

    def reset(self) -> None:
        self.drift_accumulator = 0.0
        self.stuck = False
        self.fault_mode = NoiseMode.NORMAL
        self.fault_started_t = None
        self.last_value = None
        self.last_value_time = None


def build_channel(
    name: str,
    cfg: SensorConfig,
    seed: int,
) -> Channel:
    """Construct a :class:`Channel` with a deterministic RNG.

    The RNG seed is derived from ``seed`` and the channel name, so each
    channel is independent but the bundle as a whole is reproducible.
    """
    sub_seed = int(seed) ^ (hash(name) & 0xFFFFFFFF)
    rng = np.random.default_rng(sub_seed)
    return Channel(name=name, cfg=cfg, rng=rng)


__all__ = ["Channel", "SENSOR_CHANNELS", "build_channel", "_true_value"]
