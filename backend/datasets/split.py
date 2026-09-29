"""
PHASE 21 — Dataset split policy.

A deterministic, hash-based, leakage-free split of a catalog of
``scenario_id`` strings into ``train | val | test`` buckets.

The policy is **frozen per ``(dataset_name, seed)``**: every
caller running the same spec on the same machine gets the
exact same assignment, and a given ``scenario_id`` is assigned
to *exactly one* split (no test contamination across the
generated dataset).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """A *request* for a labelled fault-injection dataset.

    Defaults to a small smoke-test configuration.
    """

    name: str = "dataset"
    mission_templates: Tuple[str, ...] = ()  # filled by writer
    fault_types: Tuple[str, ...] = ()        # filled by writer
    severity_levels: Tuple[str, ...] = ("low", "medium", "high")
    temporal_patterns: Tuple[str, ...] = (
        "linear", "step", "exponential", "pulse",
    )
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    phases_per_template: int = 10
    description: str = ""

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, "
                f"test={self.test_ratio})"
            )
        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio <= 0:
            raise ValueError(
                "train_ratio must be > 0; val_ratio and test_ratio must be >= 0"
            )
        if self.phases_per_template < 1:
            raise ValueError("phases_per_template must be >= 1")
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not self.severity_levels:
            raise ValueError("severity_levels must be non-empty")
        if not self.temporal_patterns:
            raise ValueError("temporal_patterns must be non-empty")

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "mission_templates": list(self.mission_templates),
            "fault_types": list(self.fault_types),
            "severity_levels": list(self.severity_levels),
            "temporal_patterns": list(self.temporal_patterns),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "phases_per_template": self.phases_per_template,
            "description": self.description,
        }


# ---------------------------------------------------------------------
# SplitPolicy
# ---------------------------------------------------------------------
class SplitPolicy:
    """Deterministic, hash-based, leakage-free train/val/test split.

    The mapping is computed once on construction. For a given
    ``(dataset_name, seed)`` it is the same on every host and
    every Python version, because SHA-256 is stable.

    For each ``scenario_id``, the policy computes::

        h = SHA256(dataset_name | seed | scenario_id)[:8]

    and bins the resulting 64-bit integer to [0, 1), then assigns
    it to the bucket whose cumulative ratio contains the bin
    point.
    """

    _SPLITS: Tuple[str, ...] = ("train", "val", "test")

    def __init__(
        self,
        spec: DatasetSpec,
        scenario_ids: Sequence[str] = (),
    ) -> None:
        self._spec = spec
        # Cumulative split boundaries in [0, 1).
        self._cum_train = float(spec.train_ratio)
        self._cum_val = float(spec.train_ratio + spec.val_ratio)
        # Build the assignment for the provided scenarios (if any)
        # up-front; later `assign()` calls reuse the same map.
        self._assignments: Dict[str, str] = {}
        for sid in scenario_ids:
            self._assignments[str(sid)] = self._bucket_for(sid)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def assign(self, scenario_id: str) -> str:
        """Return the split for ``scenario_id``.

        O(1) lookup if the id was pre-bucketed at construction;
        otherwise the bucket is computed on the fly and cached.
        """
        sid = str(scenario_id)
        if sid in self._assignments:
            return self._assignments[sid]
        bucket = self._bucket_for(sid)
        self._assignments[sid] = bucket
        return bucket

    def splits(self) -> Dict[str, List[str]]:
        """Return ``{split_name: [scenario_id, ...]}``.

        Note: only contains the ids that were pre-bucketed at
        construction (or assigned via :meth:`assign`); a fresh
        call to :meth:`splits` is stable for a given pre-bucket
        list.
        """
        out: Dict[str, List[str]] = {name: [] for name in self._SPLITS}
        for sid, bucket in self._assignments.items():
            out[bucket].append(sid)
        # Sort for determinism.
        for k in out:
            out[k] = sorted(out[k])
        return out

    def validate_no_leakage(self) -> None:
        """Sanity check: every assigned id is in exactly one split."""
        seen: Dict[str, str] = {}
        for sid, bucket in self._assignments.items():
            if sid in seen and seen[sid] != bucket:
                raise RuntimeError(
                    f"scenario_id {sid!r} appears in two splits: "
                    f"{seen[sid]!r} and {bucket!r}"
                )
            seen[sid] = bucket
        for bucket in self._SPLITS:
            if bucket not in self._assignments.values() and self._spec.train_ratio > 0:
                # Empty bucket is fine; the only required bucket
                # is train, since the user spec said train_ratio > 0.
                if bucket == "train":
                    raise RuntimeError(
                        f"no scenarios assigned to {bucket!r} (train_ratio > 0)"
                    )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _bucket_for(self, scenario_id: str) -> str:
        h = hashlib.sha256()
        h.update(self._spec.name.encode("utf-8"))
        h.update(b"|")
        h.update(self._spec.seed.to_bytes(8, "big", signed=True))
        h.update(b"|")
        h.update(scenario_id.encode("utf-8"))
        digest = h.digest()
        # Take the first 8 bytes as a uint64.
        u = int.from_bytes(digest[:8], "big", signed=False)
        # Normalise to [0, 1).
        u01 = u / float(1 << 64)
        if u01 < self._cum_train:
            return "train"
        if u01 < self._cum_val:
            return "val"
        return "test"


__all__ = ["DatasetSpec", "SplitPolicy"]
