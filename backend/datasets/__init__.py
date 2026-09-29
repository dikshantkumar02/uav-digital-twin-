"""
Datasets package — labelled fault-injection research datasets.

PHASE 21 — controlled fault-injection framework.

Public re-exports::

    from backend.datasets import (
        DatasetSpec, SplitPolicy, DatasetManifest, DatasetWriter,
    )
"""

from .manifest import DatasetManifest
from .split import DatasetSpec, SplitPolicy
from .writer import DatasetWriter

__all__ = [
    "DatasetManifest",
    "DatasetSpec",
    "DatasetWriter",
    "SplitPolicy",
]
