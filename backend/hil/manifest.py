"""
Manifest for a recorded HIL run (PHASE 15).

A :class:`Manifest` describes one captured scenario: the scenario
name, the sample rate, the duration, the CRC policy used for
recording, and a small block of metadata. The manifest lives next
to the recorded ``wire.bin`` and ``snapshots.jsonl`` so a replayer
can reconstruct the run without a separate database.

Round-tripping is supported through :meth:`to_dict`,
:meth:`from_dict`, :meth:`write_yaml`, and :meth:`read_yaml`. The
YAML format is human-editable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Manifest:
    """Metadata for one recorded HIL run.

    Attributes
    ----------
    scenario_name:
        Stable identifier — must match a key in
        :data:`backend.dashboard.scenarios.SCENARIO_REGISTRY`.
    sample_rate_hz:
        Pipeline tick rate (Hz) used during recording.
    duration_s:
        Total recorded duration in seconds.
    crc_policy:
        ``"ccitt"`` or ``"none"`` — the CRC policy used when
        :class:`~backend.hardware.FrameCodec` encoded the wire
        bytes. ``"ccitt"`` is the PHASE 14 default.
    recorded_at:
        ISO-8601 timestamp at which the recording finished.
    git_rev:
        Optional git commit hash of the code under test.
    notes:
        Free-form operator notes.
    n_frames:
        Number of wire frames captured. Filled in by the recorder
        on close; default 0 for a fresh manifest.
    """

    scenario_name: str
    sample_rate_hz: float
    duration_s: float
    crc_policy: str = "ccitt"
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    git_rev: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    n_frames: int = 0

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_s": self.duration_s,
            "crc_policy": self.crc_policy,
            "recorded_at": self.recorded_at,
            "git_rev": self.git_rev,
            "notes": list(self.notes),
            "n_frames": self.n_frames,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        if not isinstance(d, dict):
            raise TypeError(f"manifest dict must be a mapping, got {type(d).__name__}")
        return cls(
            scenario_name=str(d.get("scenario_name", "")),
            sample_rate_hz=float(d.get("sample_rate_hz", 0.0)),
            duration_s=float(d.get("duration_s", 0.0)),
            crc_policy=str(d.get("crc_policy", "ccitt")),
            recorded_at=str(d.get("recorded_at", "")),
            git_rev=d.get("git_rev"),
            notes=list(d.get("notes", []) or []),
            n_frames=int(d.get("n_frames", 0)),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def write_yaml(path: Path, manifest: "Manifest") -> None:
        """Write ``manifest`` to ``path`` in YAML form (stdlib PyYAML)."""
        import yaml  # local import — keeps `manifest` importable without PyYAML

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest.to_dict(), fh, sort_keys=True)

    @staticmethod
    def read_yaml(path: Path) -> "Manifest":
        """Read a manifest from ``path``."""
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return Manifest.from_dict(data if isinstance(data, dict) else {})

    # ------------------------------------------------------------------
    def with_frames(self, n_frames: int) -> "Manifest":
        """Return a copy of this manifest with ``n_frames`` set."""
        return Manifest(
            scenario_name=self.scenario_name,
            sample_rate_hz=self.sample_rate_hz,
            duration_s=self.duration_s,
            crc_policy=self.crc_policy,
            recorded_at=self.recorded_at,
            git_rev=self.git_rev,
            notes=list(self.notes),
            n_frames=int(n_frames),
        )


__all__ = ["Manifest"]
