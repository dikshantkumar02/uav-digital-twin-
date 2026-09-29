"""
PHASE 21 — Fault-injection framework.

A *research-dataset* layer on top of the existing PHASE 7
declarative fault system. The framework:

* defines the standardised :class:`FaultRecord` the user asked
  for (``fault_id``, ``fault_type``, ``start_time``,
  ``end_time``, ``severity``, ``affected_parameters``,
  ``ground_truth``, ``scenario_id``);
* bundles one or more :class:`FaultRecord` with their matching
  PHASE 7 :class:`~backend.faults.plan.FaultScenario` objects
  into a :class:`FaultScenarioBundle`;
* enumerates the full catalog of unique scenarios via
  :class:`ScenarioGenerator`.

The framework does *not* run anything — that is the
:class:`~backend.datasets.writer.DatasetWriter`'s job. The
framework is a *catalog + record* layer.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from backend.environment import MissionTemplate
from backend.sensors import NoiseMode, SENSOR_CHANNELS

from .plan import FaultClass, FaultScenario
from .taxonomy import FAULT_TAXONOMY, FaultType, get_meta


# ---------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------
# Three discrete levels: low / medium / high. The framework maps
# each level to a peak severity in [0, 1].
DEFAULT_SEVERITY_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("low", 0.20),
    ("medium", 0.50),
    ("high", 0.85),
)

DEFAULT_TEMPORAL_PATTERNS: Tuple[str, ...] = (
    "linear", "step", "exponential", "pulse",
)


# ---------------------------------------------------------------------
# FaultRecord
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaultRecord:
    """The standardised experiment-level fault record.

    Mirrors the user's spec::

        fault_id, fault_type, start_time, end_time, severity,
        affected_parameters, ground_truth, scenario_id

    Plus two framework-only fields that are required for the
    dataset to be reproducible:

    * ``temporal_pattern`` — one of ``"linear" | "step" |
      "exponential" | "pulse"``
    * ``severity_level`` — one of ``"low" | "medium" | "high"``
    """

    fault_id: str
    fault_type: FaultType
    start_time: float
    end_time: float
    severity: float
    affected_parameters: Tuple[str, ...]
    ground_truth: Dict[str, Any]
    scenario_id: str
    temporal_pattern: str
    severity_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "severity": float(self.severity),
            "affected_parameters": list(self.affected_parameters),
            "ground_truth": dict(self.ground_truth),
            "scenario_id": self.scenario_id,
            "temporal_pattern": self.temporal_pattern,
            "severity_level": self.severity_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FaultRecord":
        return cls(
            fault_id=str(d["fault_id"]),
            fault_type=FaultType(str(d["fault_type"])),
            start_time=float(d["start_time"]),
            end_time=float(d["end_time"]),
            severity=float(d["severity"]),
            affected_parameters=tuple(d["affected_parameters"]),
            ground_truth=dict(d["ground_truth"]),
            scenario_id=str(d["scenario_id"]),
            temporal_pattern=str(d["temporal_pattern"]),
            severity_level=str(d["severity_level"]),
        )


# ---------------------------------------------------------------------
# FaultScenarioBundle
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class FaultScenarioBundle:
    """One labelled experiment = (records, scenarios, mission, seed)."""

    scenario_id: str
    mission_template: MissionTemplate
    fault_records: Tuple[FaultRecord, ...]
    fault_scenarios: Tuple[FaultScenario, ...]
    seed: int
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "mission_template": self.mission_template.value,
            "fault_records": [r.to_dict() for r in self.fault_records],
            "seed": int(self.seed),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FaultScenarioBundle":
        return cls(
            scenario_id=str(d["scenario_id"]),
            mission_template=MissionTemplate(str(d["mission_template"])),
            fault_records=tuple(
                FaultRecord.from_dict(r) for r in d["fault_records"]
            ),
            fault_scenarios=tuple(),  # rebuilt by writer from records
            seed=int(d["seed"]),
            description=str(d.get("description", "")),
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _scenario_id_for(
    template: MissionTemplate,
    fault_type: FaultType,
    severity_level: str,
    pattern: str,
    offset_index: int,
    master_seed: int,
) -> str:
    """Deterministic scenario_id from the combinatorial key.

    Two calls with the same arguments always produce the same id,
    regardless of the order or the host. Used both as the
    framework's catalogue key and the SplitPolicy's hash input.
    """
    h = hashlib.sha256()
    h.update(master_seed.to_bytes(8, "big", signed=True))
    h.update(template.value.encode("utf-8"))
    h.update(b"|")
    h.update(fault_type.value.encode("utf-8"))
    h.update(b"|")
    h.update(severity_level.encode("utf-8"))
    h.update(b"|")
    h.update(pattern.encode("utf-8"))
    h.update(b"|")
    h.update(int(offset_index).to_bytes(4, "big", signed=True))
    return "scn-" + h.hexdigest()[:16]


def _fault_id_for(scenario_id: str, n: int) -> str:
    """Stable fault_id derived from the scenario_id and the index."""
    h = hashlib.sha256()
    h.update(scenario_id.encode("utf-8"))
    h.update(b"|")
    h.update(int(n).to_bytes(4, "big", signed=True))
    return "flt-" + h.hexdigest()[:16]


def _resolve_severity_peak(level: str) -> float:
    """Map a severity-level name to its peak severity in [0, 1]."""
    for name, peak in DEFAULT_SEVERITY_LEVELS:
        if name == level:
            return float(peak)
    raise ValueError(
        f"unknown severity level {level!r}; valid: "
        f"{[n for n, _ in DEFAULT_SEVERITY_LEVELS]}"
    )


def _mission_template_duration_s(template: MissionTemplate) -> float:
    """The total mission length for a given template."""
    from backend.environment import MISSION_TEMPLATES
    from backend.environment.profile_generator import (
        _resolve_phase_start_times,
    )

    raw_phases = MISSION_TEMPLATES[template]
    resolved = _resolve_phase_start_times(list(raw_phases), 5.0)
    return float(resolved[-1].end_t_s)


def _build_fault_scenario(
    *,
    fault_type: FaultType,
    severity_peak: float,
    onset_time_s: float,
    duration_s: float,
    progression: str,
    seed: int,
    sensor_channel: Optional[str] = None,
) -> FaultScenario:
    """Build the PHASE 7 FaultScenario that realises a framework fault.

    The mapping is per the FAULT_TAXONOMY:
    * SENSOR_FAULT: needs target_channel + target_sensor_mode
    * all others: no extra fields
    """
    meta = get_meta(fault_type)
    if meta.mapped_class is FaultClass.SENSOR_FAULT:
        if sensor_channel is None:
            raise ValueError(
                f"{fault_type.value} requires a target sensor channel"
            )
        if meta.sensor_mode is None:
            raise ValueError(
                f"{fault_type.value} has no sensor_mode in FAULT_TAXONOMY"
            )
        return FaultScenario(
            fault_class=FaultClass.SENSOR_FAULT,
            severity=float(severity_peak),
            onset_time_s=float(onset_time_s),
            duration_s=float(duration_s),
            progression=str(progression),
            seed=int(seed),
            target_channel=str(sensor_channel),
            target_sensor_mode=meta.sensor_mode,
        )
    return FaultScenario(
        fault_class=meta.mapped_class,
        severity=float(severity_peak),
        onset_time_s=float(onset_time_s),
        duration_s=float(duration_s),
        progression=str(progression),
        seed=int(seed),
    )


def _build_ground_truth(
    *,
    fault_type: FaultType,
    severity_peak: float,
    sensor_channel: Optional[str],
) -> Dict[str, Any]:
    """Build the per-fault ground-truth dict for the record."""
    meta = get_meta(fault_type)
    gt: Dict[str, Any] = {
        "fault_class": meta.mapped_class.value,
        "category": meta.category.value,
        "peak_severity": float(severity_peak),
    }
    if meta.mapped_class is FaultClass.SENSOR_FAULT:
        gt["sensor_mode"] = (
            meta.sensor_mode.value if meta.sensor_mode is not None else None
        )
        gt["target_channel"] = sensor_channel
    if meta.env_knob is not None:
        gt["env_knob"] = meta.env_knob
    return gt


# ---------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------
class ScenarioGenerator:
    """Enumerate the full scenario catalog for a given spec.

    Each scenario is a unique combination of:

    * mission template (PHASE 20)
    * fault type (one of the 17 :class:`FaultType` values)
    * severity level (low / medium / high)
    * temporal pattern (linear / step / exponential / pulse)
    * offset index (0..phases_per_template-1) — the framework
      distributes fault onsets across the mission so two
      scenarios with the same (template, fault, severity,
      pattern) but different offsets are *not* the same.
    """

    def __init__(
        self,
        mission_templates: Sequence[MissionTemplate],
        fault_types: Sequence[FaultType],
        severity_levels: Sequence[str] = DEFAULT_SEVERITY_LEVELS.__class__(
            (name for name, _ in DEFAULT_SEVERITY_LEVELS)
        ),
        temporal_patterns: Sequence[str] = DEFAULT_TEMPORAL_PATTERNS,
        phases_per_template: int = 10,
        base_seed: int = 42,
        sensor_channels: Sequence[str] = SENSOR_CHANNELS,
    ) -> None:
        if not mission_templates:
            raise ValueError("at least one mission template required")
        if not fault_types:
            raise ValueError("at least one fault type required")
        if not severity_levels:
            raise ValueError("at least one severity level required")
        if not temporal_patterns:
            raise ValueError("at least one temporal pattern required")
        if phases_per_template < 1:
            raise ValueError("phases_per_template must be >= 1")
        # Validate all severity levels are real.
        valid_levels = [name for name, _ in DEFAULT_SEVERITY_LEVELS]
        for lvl in severity_levels:
            if lvl not in valid_levels:
                raise ValueError(
                    f"unknown severity level {lvl!r}; valid: {valid_levels}"
                )
        # Validate all temporal patterns are supported.
        from .progression import _VALID_MODELS
        for pat in temporal_patterns:
            if pat not in _VALID_MODELS:
                raise ValueError(
                    f"unknown temporal pattern {pat!r}; valid: "
                    f"{list(_VALID_MODELS)}"
                )
        self._templates: Tuple[MissionTemplate, ...] = tuple(mission_templates)
        self._fault_types: Tuple[FaultType, ...] = tuple(fault_types)
        self._severity_levels: Tuple[str, ...] = tuple(severity_levels)
        self._temporal_patterns: Tuple[str, ...] = tuple(temporal_patterns)
        self._phases_per_template: int = int(phases_per_template)
        self._base_seed: int = int(base_seed)
        self._sensor_channels: Tuple[str, ...] = tuple(sensor_channels)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def total_count(self) -> int:
        """The number of unique scenarios in the catalog."""
        return (
            len(self._templates)
            * len(self._fault_types)
            * len(self._severity_levels)
            * len(self._temporal_patterns)
            * self._phases_per_template
        )

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------
    def enumerate_scenarios(self) -> Iterator[FaultScenarioBundle]:
        """Yield every FaultScenarioBundle in the catalog.

        The order is deterministic: templates are yielded in the
        order they were given, then fault types, then severity
        levels, then patterns, then offset indices.
        """
        for t_idx, template in enumerate(self._templates):
            total_dur = _mission_template_duration_s(template)
            for f_idx, ft in enumerate(self._fault_types):
                meta = get_meta(ft)
                for s_idx, level in enumerate(self._severity_levels):
                    sev_peak = _resolve_severity_peak(level)
                    for p_idx, pattern in enumerate(self._temporal_patterns):
                        for off in range(self._phases_per_template):
                            yield self._make_one(
                                template=template,
                                ft=ft,
                                meta=meta,
                                level=level,
                                pattern=pattern,
                                severity_peak=sev_peak,
                                offset_index=off,
                                total_duration_s=total_dur,
                                indices=(t_idx, f_idx, s_idx, p_idx, off),
                            )

    def get_scenario(self, scenario_id: str) -> Optional[FaultScenarioBundle]:
        """Look up a single scenario by id. Returns None if unknown."""
        for bundle in self.enumerate_scenarios():
            if bundle.scenario_id == scenario_id:
                return bundle
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _make_one(
        self,
        *,
        template: MissionTemplate,
        ft: FaultType,
        meta: Any,
        level: str,
        pattern: str,
        severity_peak: float,
        offset_index: int,
        total_duration_s: float,
        indices: Tuple[int, int, int, int, int],
    ) -> FaultScenarioBundle:
        t_idx, f_idx, s_idx, p_idx, off = indices
        # Deterministic scenario_id (does not depend on enumeration
        # order — see _scenario_id_for).
        scenario_id = _scenario_id_for(
            template, ft, level, pattern, offset_index, self._base_seed
        )
        # Per-bundle RNG seed: derived from the scenario_id so the
        # sensor channel choice is reproducible.
        seed_hash = hashlib.sha256(scenario_id.encode("utf-8")).digest()
        seed = int.from_bytes(seed_hash[:4], "big", signed=False)
        # Distribute fault onset across the mission: skip the first
        # 5% (warmup) and the last 5% (cooldown) so the active
        # window sits inside the mission body.
        onset_time_s = max(
            0.0, 0.05 * total_duration_s + offset_index
            * (0.9 * total_duration_s / max(1, self._phases_per_template))
        )
        # Duration: meta default, scaled to fit if the mission is
        # shorter than 2x the default.
        duration_s = min(meta.default_duration_s, 0.6 * total_duration_s)
        end_time_s = onset_time_s + duration_s
        # Pick a sensor channel if this is a SENSOR_* fault.
        sensor_channel: Optional[str] = None
        if meta.mapped_class is FaultClass.SENSOR_FAULT:
            # Stable pick from the seed.
            ch_idx = seed % len(self._sensor_channels)
            sensor_channel = self._sensor_channels[ch_idx]
        # Build the record + scenario.
        record = FaultRecord(
            fault_id=_fault_id_for(scenario_id, 0),
            fault_type=ft,
            start_time=float(onset_time_s),
            end_time=float(end_time_s),
            severity=float(severity_peak),
            affected_parameters=tuple(meta.affected_parameters),
            ground_truth=_build_ground_truth(
                fault_type=ft,
                severity_peak=severity_peak,
                sensor_channel=sensor_channel,
            ),
            scenario_id=scenario_id,
            temporal_pattern=pattern,
            severity_level=level,
        )
        scenario = _build_fault_scenario(
            fault_type=ft,
            severity_peak=severity_peak,
            onset_time_s=onset_time_s,
            duration_s=duration_s,
            progression=pattern,
            seed=seed,
            sensor_channel=sensor_channel,
        )
        description = (
            f"{template.value} | {ft.value} | {level} | {pattern} "
            f"| onset={onset_time_s:.0f}s dur={duration_s:.0f}s"
        )
        return FaultScenarioBundle(
            scenario_id=scenario_id,
            mission_template=template,
            fault_records=(record,),
            fault_scenarios=(scenario,),
            seed=seed,
            description=description,
        )


__all__ = [
    "DEFAULT_SEVERITY_LEVELS",
    "DEFAULT_TEMPORAL_PATTERNS",
    "FaultRecord",
    "FaultScenarioBundle",
    "ScenarioGenerator",
]
