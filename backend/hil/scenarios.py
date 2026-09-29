"""
HIL scenario registry (PHASE 15).

The HIL harness only cares about a small subset of the dashboard
scenarios: the five canonical fault classes (healthy, engine
degradation, overheating, sensor fault, vibration anomaly), each
run for the default 60-second mission. The names come from
:data:`backend.dashboard.scenarios.SCENARIO_REGISTRY` — the HIL
config (``config/hil.yaml``) lists which subset to cover.

This module is a thin layer that:

* reads the configured scenario names from
  :func:`backend.config.load_hil_config`;
* resolves each name against ``SCENARIO_REGISTRY``;
* raises a clear error if the operator asked for a scenario that
  does not exist (catches typos at boot, not during a 60-second
  run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.dashboard.scenarios import SCENARIO_REGISTRY, ScenarioSpec

from ..config import load_hil_config
from .manifest import Manifest


# Default scenarios — also baked into HilConfig so the harness is
# usable even when config/hil.yaml is missing.
DEFAULT_SCENARIOS: List[str] = [
    "healthy_60s",
    "engine_degradation_60s",
    "overheating_60s",
    "sensor_fault_60s",
    "vibration_anomaly_60s",
]


# ---------------------------------------------------------------------
# HILScenario
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HILScenario:
    """One scenario wired into the HIL harness.

    Attributes
    ----------
    name:
        Scenario name (must match a key in ``SCENARIO_REGISTRY``).
    spec:
        The :class:`ScenarioSpec` from the dashboard registry.
    duration_s:
        How long to run the scenario in seconds. The dashboard's
        scenarios are 60 s by default; the HIL harness reuses that.
    sample_rate_hz:
        Pipeline tick rate the recorder will use. Matches the
        :class:`PipelineRunner` config.
    """

    name: str
    spec: ScenarioSpec
    duration_s: float = 60.0
    sample_rate_hz: float = 10.0


# ---------------------------------------------------------------------
# HILScenarioRegistry
# ---------------------------------------------------------------------
class HILScenarioRegistry:
    """Resolves HIL scenario names to :class:`HILScenario` objects."""

    def __init__(self, scenarios: Optional[List[str]] = None) -> None:
        if scenarios is None:
            scenarios = list(DEFAULT_SCENARIOS)
        self._scenarios: Dict[str, HILScenario] = {}
        for name in scenarios:
            spec = SCENARIO_REGISTRY.get(name)
            if spec is None:
                raise KeyError(
                    f"scenario {name!r} not found in SCENARIO_REGISTRY; "
                    f"available: {sorted(SCENARIO_REGISTRY)}"
                )
            self._scenarios[name] = HILScenario(name=name, spec=spec)

    @classmethod
    def from_config(cls, config_dir: Optional[str] = None) -> "HILScenarioRegistry":
        """Build the registry from :func:`load_hil_config`."""
        cfg = load_hil_config(config_dir)
        return cls(scenarios=list(cfg.scenarios or DEFAULT_SCENARIOS))

    # ------------------------------------------------------------------
    @property
    def names(self) -> List[str]:
        return list(self._scenarios)

    def __contains__(self, name: str) -> bool:
        return name in self._scenarios

    def __iter__(self):
        return iter(self._scenarios.values())

    def get(self, name: str) -> HILScenario:
        if name not in self._scenarios:
            raise KeyError(
                f"scenario {name!r} not covered by HIL harness; "
                f"covered: {sorted(self._scenarios)}"
            )
        return self._scenarios[name]

    # ------------------------------------------------------------------
    def manifest_for(
        self,
        name: str,
        *,
        crc_policy: str = "ccitt",
        notes: Optional[List[str]] = None,
    ) -> Manifest:
        """Build a fresh :class:`Manifest` for ``name``."""
        sc = self.get(name)
        return Manifest(
            scenario_name=sc.name,
            sample_rate_hz=float(sc.sample_rate_hz),
            duration_s=float(sc.duration_s),
            crc_policy=str(crc_policy),
            notes=list(notes or []),
        )


__all__ = [
    "DEFAULT_SCENARIOS",
    "HILScenario",
    "HILScenarioRegistry",
]
