"""
Telemetry package — streaming bus, preprocessor, latency tracking.

Public re-exports::

    from backend.telemetry import (
        TelemetryFrame, TelemetryQueue, FrameStatus,
        Preprocessor, StreamSource, LatencyTracker,
        # PHASE 17 — streaming pipeline
        AdapterConfig, AsyncStreamingPipeline, PipelineConfig,
        PipelineResult, StageStats, TelemetryAdapter,
        compute_data_quality,
    )
"""

from .adapter import (
    AdapterConfig,
    TelemetryAdapter,
    compute_data_quality,
)
from .frame import FrameStatus, TelemetryFrame
from .latency import LatencyTracker, StageLatency
from .pipeline import (
    AsyncStreamingPipeline,
    PipelineConfig,
    PipelineResult,
    StageStats,
)
from .preprocessor import Preprocessor
from .queue import QueueFullError, TelemetryQueue
from .source import StreamSource

__all__ = [
    "AdapterConfig",
    "AsyncStreamingPipeline",
    "FrameStatus",
    "LatencyTracker",
    "PipelineConfig",
    "PipelineResult",
    "Preprocessor",
    "QueueFullError",
    "StageLatency",
    "StageStats",
    "StreamSource",
    "TelemetryAdapter",
    "TelemetryFrame",
    "TelemetryQueue",
    "compute_data_quality",
]
