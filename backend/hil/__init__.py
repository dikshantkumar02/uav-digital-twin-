"""
HIL validation harness (PHASE 15).

Public re-exports so callers can write::

    from backend.hil import (
        HilRunner, WireRecorder, WireReplayer, SnapshotCapture,
        GoldenRun, Tolerance, ComparisonReport, Mismatch, compare_runs,
        Manifest, HILScenarioRegistry, HILScenario, load_golden,
    )

The package covers the three HIL flows:

* **Record** — :class:`WireRecorder` + :meth:`HilRunner.record`
* **Replay** — :class:`WireReplayer` + :meth:`HilRunner.validate`
* **Compare** — :func:`compare_runs` + :class:`ComparisonReport`
"""

from .capture import SnapshotCapture
from .comparison import ComparisonReport, Mismatch, Tolerance, compare_runs
from .golden import GoldenRun, load_golden
from .manifest import Manifest
from .recorder import WireRecorder
from .replayer import WireReplayer
from .runner import HilRunner, ValidateResult, make_default_runner
from .scenarios import (
    DEFAULT_SCENARIOS,
    HILScenario,
    HILScenarioRegistry,
)

__all__ = [
    "ComparisonReport",
    "DEFAULT_SCENARIOS",
    "GoldenRun",
    "HILScenario",
    "HILScenarioRegistry",
    "HilRunner",
    "Manifest",
    "Mismatch",
    "SnapshotCapture",
    "Tolerance",
    "ValidateResult",
    "WireRecorder",
    "WireReplayer",
    "compare_runs",
    "load_golden",
    "make_default_runner",
]
