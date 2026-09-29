"""
PHASE 21 — Dataset manifest.

A small frozen container that records everything a researcher
needs to consume a generated dataset: the spec, the
``scenario_id`` → split assignment, and the relative path of
each scenario's trace artefact on disk.

The manifest serialises to / from JSON for portability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

from .split import DatasetSpec


@dataclass(frozen=True)
class DatasetManifest:
    """One manifest = one written dataset."""

    dataset_name: str
    seed: int
    spec: DatasetSpec
    splits: Dict[str, List[str]]
    scenario_paths: Dict[str, str]
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "seed": int(self.seed),
            "spec": self.spec.to_dict(),
            "splits": {
                k: sorted(v) for k, v in self.splits.items()
            },
            "scenario_paths": dict(self.scenario_paths),
            "description": self.description,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DatasetManifest":
        d = json.loads(text)
        spec = DatasetSpec(
            name=str(d["spec"]["name"]),
            mission_templates=tuple(d["spec"].get("mission_templates", ())),
            fault_types=tuple(d["spec"].get("fault_types", ())),
            severity_levels=tuple(d["spec"]["severity_levels"]),
            temporal_patterns=tuple(d["spec"]["temporal_patterns"]),
            train_ratio=float(d["spec"]["train_ratio"]),
            val_ratio=float(d["spec"]["val_ratio"]),
            test_ratio=float(d["spec"]["test_ratio"]),
            seed=int(d["spec"]["seed"]),
            phases_per_template=int(d["spec"]["phases_per_template"]),
            description=str(d["spec"].get("description", "")),
        )
        return cls(
            dataset_name=str(d["dataset_name"]),
            seed=int(d["seed"]),
            spec=spec,
            splits={k: list(v) for k, v in d["splits"].items()},
            scenario_paths=dict(d["scenario_paths"]),
            description=str(d.get("description", "")),
        )

    def total_scenarios(self) -> int:
        return sum(len(v) for v in self.splits.values())


__all__ = ["DatasetManifest"]
