"""
PHASE 21 — Dataset writer.

Given a :class:`~backend.datasets.split.DatasetSpec`, enumerate
the full scenario catalog, run each scenario through a
:class:`~backend.environment.profile_generator.ProfileGenerator`,
and write a labelled dataset to disk:

  <output_dir>/
    manifest.json
    traces/
      <scenario_id>.npz

The writer is **idempotent**: re-running with the same spec and
seed overwrites the same files. The split assignment is
frozen per ``(dataset_name, seed)`` so the same scenario_id
always lands in the same split across runs and across hosts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.environment import MissionTemplate, ProfileGenerator
from backend.environment.profile_generator import ProfileGeneratorConfig
from backend.faults import FaultScenario
from backend.faults.framework import FaultScenarioBundle, ScenarioGenerator
from backend.faults.taxonomy import FaultType

from .manifest import DatasetManifest
from .split import DatasetSpec, SplitPolicy


# ---------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------
def _write_trace(
    path: Path,
    bundle: FaultScenarioBundle,
) -> None:
    """Run a single scenario and write the trace to ``path``.

    The trace is saved as a single ``.npz`` with the following
    arrays:

    * ``time_s`` (T,) — mission time in seconds
    * ``env_altitude_m`` (T,)
    * ``env_airspeed_mps`` (T,)
    * ``env_throttle`` (T,)
    * ``env_ambient_temp_c`` (T,)
    * ``engine_rpm`` (T,)
    * ``engine_vibration`` (T,) — proxy (vibration_external * gain)
    * ``sensor_<channel>`` (T,) — one array per sensor channel
    * ``fault_fault_class`` (T,) — string per tick (HEALTHY when no fault)
    * ``fault_severity`` (T,) — float in [0, 1]
    * ``fault_scenario_id`` — single string in the file attrs
    """
    template = bundle.mission_template
    # The first fault in the bundle is the one we run with
    # (multi-fault is not exercised by the default generator).
    fault_scen = bundle.fault_scenarios[0] if bundle.fault_scenarios else None

    # Build a ProfileGenerator with the template + a quick config.
    cfg = ProfileGeneratorConfig(
        seed=int(bundle.seed),
        dt_s=0.1,
        transition_default_s=5.0,
        fault_plan=fault_scen,
    )
    gen = ProfileGenerator(template=template, config=cfg)
    trace = gen.run()

    # Extract aligned numpy arrays from the truth side.
    n = len(trace.truth)
    time_s = np.zeros(n, dtype=np.float64)
    env_alt = np.zeros(n, dtype=np.float64)
    env_spd = np.zeros(n, dtype=np.float64)
    env_thr = np.zeros(n, dtype=np.float64)
    env_amb = np.zeros(n, dtype=np.float64)
    env_pressure_pa = np.zeros(n, dtype=np.float64)
    engine_rpm = np.zeros(n, dtype=np.float64)
    engine_egt = np.zeros(n, dtype=np.float64)
    engine_cht = np.zeros(n, dtype=np.float64)
    engine_vibration = np.zeros(n, dtype=np.float64)
    engine_oil_pressure = np.zeros(n, dtype=np.float64)
    engine_oil_temp = np.zeros(n, dtype=np.float64)
    engine_fuel_flow = np.zeros(n, dtype=np.float64)
    engine_bsfc = np.zeros(n, dtype=np.float64)
    fault_class_arr = np.empty(n, dtype=object)
    fault_severity_arr = np.zeros(n, dtype=np.float64)

    # Collect per-sensor arrays lazily.
    sensor_arrays: Dict[str, np.ndarray] = {}

    for i, tick in enumerate(trace.truth):
        time_s[i] = float(tick.time_s)
        env_alt[i] = float(tick.env.altitude_m)
        env_spd[i] = float(tick.env.airspeed_mps)
        env_thr[i] = float(tick.env.throttle)
        env_amb[i] = float(tick.env.atmosphere.temperature_c)
        env_pressure_pa[i] = float(tick.env.atmosphere.pressure_pa)
        # Engine side
        e = tick.engine
        engine_rpm[i] = float(getattr(e, "rpm", 0.0))
        engine_egt[i] = float(getattr(e, "egt_c", 0.0))
        engine_cht[i] = float(getattr(e, "cht_c", 0.0))
        engine_vibration[i] = float(getattr(e, "vibration_rms_g", 0.0))
        engine_oil_pressure[i] = float(getattr(e, "oil_pressure_psi", 0.0))
        engine_oil_temp[i] = float(getattr(e, "oil_temperature_c", 0.0))
        engine_fuel_flow[i] = float(getattr(e, "fuel_flow_lph", 0.0))
        engine_bsfc[i] = float(getattr(e, "bsfc_g_per_kwh", 0.0))
        # Sensor readings
        for name, reading in tick.sensors.items():
            arr = sensor_arrays.setdefault(
                name, np.zeros(n, dtype=np.float64)
            )
            arr[i] = float(reading.value)
        # Fault truth
        if tick.fault_truth is not None:
            fault_class_arr[i] = str(tick.fault_truth.fault_class)
            fault_severity_arr[i] = float(tick.fault_truth.severity)

    # Build the npz payload.
    payload: Dict[str, np.ndarray] = {
        "time_s": time_s,
        "env_altitude_m": env_alt,
        "env_airspeed_mps": env_spd,
        "env_throttle": env_thr,
        "env_ambient_temp_c": env_amb,
        "env_pressure_pa": env_pressure_pa,
        "engine_rpm": engine_rpm,
        "engine_egt_c": engine_egt,
        "engine_cht_c": engine_cht,
        "engine_vibration_g": engine_vibration,
        "engine_oil_pressure_psi": engine_oil_pressure,
        "engine_oil_temp_c": engine_oil_temp,
        "engine_fuel_flow_lph": engine_fuel_flow,
        "engine_bsfc_g_per_kwh": engine_bsfc,
        "fault_fault_class": fault_class_arr,
        "fault_severity": fault_severity_arr,
    }
    for name, arr in sensor_arrays.items():
        # Names like "rpm" become "sensor_rpm" for clarity.
        payload[f"sensor_{name}"] = arr

    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez requires a plain string or Path; it does NOT support
    # arbitrary kwargs for attributes. We store the scenario_id
    # as a sidecar JSON file next to the npz — small and robust.
    np.savez(path, **payload)
    sidecar = path.with_suffix(path.suffix + ".scenario.json")
    sidecar.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------
# DatasetWriter
# ---------------------------------------------------------------------
class DatasetWriter:
    """Generate the dataset described by ``spec`` into ``output_dir``."""

    def __init__(
        self,
        output_dir: "Path | str",
        spec: DatasetSpec,
        *,
        generator: "ScenarioGenerator | None" = None,
        # If False, do not actually run scenarios — useful for tests
        # that want to inspect the manifest + split without paying
        # the simulation cost.
        run_scenarios: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._spec = spec
        self._run = run_scenarios
        # Build the generator from the spec.
        if generator is None:
            templates = tuple(
                MissionTemplate(t) for t in spec.mission_templates
            ) if spec.mission_templates else tuple(MissionTemplate)
            fault_types = tuple(
                FaultType(t) for t in spec.fault_types
            ) if spec.fault_types else tuple(FaultType)
            generator = ScenarioGenerator(
                mission_templates=templates,
                fault_types=fault_types,
                severity_levels=spec.severity_levels,
                temporal_patterns=spec.temporal_patterns,
                phases_per_template=spec.phases_per_template,
                base_seed=spec.seed,
            )
        self._generator = generator
        # Pre-compute the scenario_ids so the split is locked
        # *before* anything is run.
        self._scenario_ids: List[str] = []
        for bundle in self._generator.enumerate_scenarios():
            self._scenario_ids.append(bundle.scenario_id)
        self._split = SplitPolicy(spec, scenario_ids=self._scenario_ids)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    @property
    def total_count(self) -> int:
        return self._generator.total_count

    @property
    def split_policy(self) -> SplitPolicy:
        return self._split

    def write(self) -> DatasetManifest:
        """Run every scenario, write traces + manifest, return the manifest."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = self._output_dir / "traces"
        scenario_paths: Dict[str, str] = {}
        splits: Dict[str, List[str]] = {
            "train": [], "val": [], "test": [],
        }

        for bundle in self._generator.enumerate_scenarios():
            sid = bundle.scenario_id
            split = self._split.assign(sid)
            splits[split].append(sid)
            rel = Path("traces") / f"{sid}.npz"
            out = self._output_dir / rel
            if self._run:
                _write_trace(out, bundle)
            else:
                # Write a stub file so the manifest is still
                # navigable. Tests opt into this.
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, time_s=np.zeros(1))
                sidecar = out.with_suffix(out.suffix + ".scenario.json")
                sidecar.write_text(json.dumps(bundle.to_dict(), indent=2))
            scenario_paths[sid] = str(rel)

        # Sanity check.
        self._split.validate_no_leakage()
        manifest = DatasetManifest(
            dataset_name=self._spec.name,
            seed=self._spec.seed,
            spec=self._spec,
            splits=splits,
            scenario_paths=scenario_paths,
            description=self._spec.description,
        )
        (self._output_dir / "manifest.json").write_text(
            manifest.to_json(indent=2)
        )
        return manifest


__all__ = ["DatasetWriter"]
