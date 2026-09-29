"""
Golden-run loader (PHASE 15).

A :class:`GoldenRun` is the canonical record of a recorded HIL
scenario: the manifest describing it, and the list of
``DashboardSnapshot.to_dict()`` payloads captured during recording.

The JSONL file is the source of truth for the snapshots; the YAML
manifest is metadata (scenario name, sample rate, recorded_at,
git_rev, n_frames). Both live in the same directory so a replayer
can locate everything from a single root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import Manifest


# ---------------------------------------------------------------------
# GoldenRun
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GoldenRun:
    """An immutable view of one golden-recorded HIL run.

    Attributes
    ----------
    root:
        Directory holding ``manifest.yaml``, ``wire.bin`` and
        ``snapshots.jsonl`` for this scenario.
    manifest:
        The parsed :class:`Manifest` (with ``n_frames`` populated
        by the recorder).
    snapshots:
        The list of snapshot dicts loaded from ``snapshots.jsonl``,
        one per tick, in recording order.
    """

    root: Path
    manifest: Manifest
    snapshots: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.snapshots)

    @property
    def bin_path(self) -> Path:
        return self.root / "wire.bin"

    @property
    def jsonl_path(self) -> Path:
        return self.root / "snapshots.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.yaml"


# ---------------------------------------------------------------------
# load_golden
# ---------------------------------------------------------------------
def load_golden(root: Path, *, manifest: Optional[Manifest] = None) -> GoldenRun:
    """Load a golden run from ``root``.

    Parameters
    ----------
    root:
        Directory containing ``manifest.yaml`` (optional) and
        ``snapshots.jsonl`` (required). The wire bytes are not
        loaded — only the manifest metadata and the JSONL snapshot
        dicts.
    manifest:
        Optional pre-parsed :class:`Manifest` to use instead of
        reading ``manifest.yaml``. Useful for tests that build the
        manifest in memory.

    Raises
    ------
    FileNotFoundError
        If ``snapshots.jsonl`` does not exist under ``root``.
    """
    root = Path(root)
    jsonl_path = root / "snapshots.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"snapshots.jsonl not found in {root}")
    if manifest is None:
        mpath = root / "manifest.yaml"
        if mpath.is_file():
            manifest = Manifest.read_yaml(mpath)
        else:
            manifest = Manifest(
                scenario_name=root.name,
                sample_rate_hz=0.0,
                duration_s=0.0,
            )
    snapshots: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            snapshots.append(json.loads(line))
    return GoldenRun(root=root, manifest=manifest, snapshots=snapshots)


__all__ = ["GoldenRun", "load_golden"]
