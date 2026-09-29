"""
SnapshotCapture — writes :class:`DashboardSnapshot` dicts to a JSONL file.

Each line is a single ``json.dumps(snap.to_dict(), sort_keys=True)``
so two runs of the same scenario with the same seed produce
byte-identical JSONL files. This is the property the comparator
relies on for its golden-frame reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from backend.dashboard.snapshot import DashboardSnapshot


class SnapshotCapture:
    """Append-only JSONL writer for :class:`DashboardSnapshot`."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        self._n = 0

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def frames_captured(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    def capture(self, snapshot: DashboardSnapshot) -> None:
        """Append one snapshot's dict to the JSONL file."""
        d: Dict[str, Any] = snapshot.to_dict()
        self._fh.write(json.dumps(d, sort_keys=True) + "\n")
        self._fh.flush()
        self._n += 1

    def capture_dict(self, d: Dict[str, Any]) -> None:
        """Append a pre-serialised dict (for tests + replay path)."""
        self._fh.write(json.dumps(d, sort_keys=True) + "\n")
        self._fh.flush()
        self._n += 1

    # ------------------------------------------------------------------
    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


__all__ = ["SnapshotCapture"]
