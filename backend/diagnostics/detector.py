"""
Anomaly detector (PHASE 8).

Per-tick entry point: :meth:`AnomalyDetector.detect`. Takes a
:class:`~backend.digital_twin.ResidualFrame` (from the twin) and a
:class:`~backend.sensors.SensorSample` (from the bundle) and emits
an :class:`~backend.diagnostics.types.AnomalyAssessment`.

The detector is **stateful** only for the EWMA smoother; everything
else is pure. ``reset()`` clears the smoother and the last-assessment
cache.

Fusion rule, label thresholds, sensor-health table, and EWMA alpha
are all configurable so PHASE 9 can wrap or replace individual
modules without breaking the consumer interface.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from backend.digital_twin import ResidualFrame
from backend.sensors import NoiseMode, SensorSample
from backend.telemetry import FrameStatus

from .fusion import ALL_KNOWN_CHANNELS, fuse_channel, overall_score
from .residual_score import (
    EwmaSmoother,
    residual_confidences,
    residual_scores,
)
from .sensor_health import DEFAULT_SENSOR_HEALTH_TABLE
from .types import (
    AnomalyAssessment,
    AnomalyLabel,
    AnomalyThresholds,
    ChannelAnomaly,
)


class AnomalyDetector:
    """Per-tick anomaly detector."""

    def __init__(
        self,
        thresholds: Optional[AnomalyThresholds] = None,
        ewma_alpha: float = 0.20,
        sensor_health_table: Optional[Dict[NoiseMode, Tuple[float, list]]] = None,
    ) -> None:
        self._thresholds = thresholds or AnomalyThresholds()
        self._smoother = EwmaSmoother(alpha=ewma_alpha)
        self._sensor_table = sensor_health_table or DEFAULT_SENSOR_HEALTH_TABLE
        self._last: Optional[AnomalyAssessment] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @property
    def thresholds(self) -> AnomalyThresholds:
        return self._thresholds

    @property
    def smoother(self) -> EwmaSmoother:
        return self._smoother

    @property
    def last_assessment(self) -> Optional[AnomalyAssessment]:
        return self._last

    def reset(self) -> None:
        self._smoother.reset()
        self._last = None

    # ------------------------------------------------------------------
    # Per-tick detection
    # ------------------------------------------------------------------
    def detect(
        self,
        residual_frame: ResidualFrame,
        sensor_sample: Optional[SensorSample] = None,
        frame_status: FrameStatus = FrameStatus.OK,
    ) -> AnomalyAssessment:
        """Compute the anomaly assessment for this tick.

        ``frame_status`` is the validation status of the underlying
        telemetry frame (PHASE 5). If it is ``INVALID`` the
        assessment is reported as ``INSUFFICIENT_DATA`` with low
        confidence.
        """
        notes = []
        time_s = residual_frame.time_s

        # --- Special case: INVALID frame --------------------------------
        if frame_status is FrameStatus.INVALID:
            self._last = AnomalyAssessment(
                time_s=time_s,
                overall_score=0.0,
                overall_label=AnomalyLabel.INSUFFICIENT_DATA,
                confidence=0.0,
                channels={},
                notes=["frame_status=INVALID"],
                frame_status=frame_status.value,
            )
            return self._last

        # --- Per-channel residual score (smoothed) -----------------------
        raw_scores = residual_scores(residual_frame)
        confs = residual_confidences(residual_frame)
        smoothed: Dict[str, Optional[float]] = {}
        for ch, raw in raw_scores.items():
            smoothed[ch] = self._smoother.update(ch, raw)

        # --- Per-channel fusion ------------------------------------------
        channel_outputs: Dict[str, ChannelAnomaly] = {}
        # Always include channels the residual frame mentioned.
        considered = set(smoothed.keys())
        # Also fold in sensor-only channels (e.g. a STUCK reading on
        # a channel the twin happens not to be tracking right now).
        if sensor_sample is not None and sensor_sample.readings:
            for ch, reading in sensor_sample.readings.items():
                if reading.mode in (
                    NoiseMode.STUCK,
                    NoiseMode.DRIFTING,
                    NoiseMode.DROPPED,
                    NoiseMode.FAULT,
                ):
                    considered.add(ch)

        for ch in sorted(considered):
            smoothed_v = smoothed.get(ch)
            residual_conf = confs.get(ch, 0.0)
            channel_outputs[ch] = fuse_channel(
                channel=ch,
                smoothed_residual=smoothed_v,
                residual_conf=residual_conf,
                sensor_sample=sensor_sample,
                thresholds=self._thresholds,
            )

        if not channel_outputs:
            notes.append("no channels with evidence")
            self._last = AnomalyAssessment(
                time_s=time_s,
                overall_score=0.0,
                overall_label=AnomalyLabel.INSUFFICIENT_DATA,
                confidence=0.0,
                channels={},
                notes=notes,
                frame_status=frame_status.value,
            )
            return self._last

        # --- Overall score + confidence ---------------------------------
        chan_scores = {ch: ca.score for ch, ca in channel_outputs.items()}
        overall = overall_score(chan_scores)
        confs_list = [ca.confidence for ca in channel_outputs.values()]
        overall_conf = sum(confs_list) / len(confs_list) if confs_list else 0.0

        if overall >= self._thresholds.anomaly_score:
            label = AnomalyLabel.ANOMALY
        elif overall >= self._thresholds.warn_score:
            label = AnomalyLabel.WARN
        else:
            label = AnomalyLabel.NORMAL

        # Demote to INSUFFICIENT_DATA if the overall confidence is too
        # low to trust the label.
        if (
            overall_conf < self._thresholds.min_confidence
            and label is not AnomalyLabel.NORMAL
        ):
            notes.append(
                f"overall_confidence {overall_conf:.2f} < min_confidence "
                f"{self._thresholds.min_confidence:.2f}"
            )
            label = AnomalyLabel.INSUFFICIENT_DATA

        self._last = AnomalyAssessment(
            time_s=time_s,
            overall_score=overall,
            overall_label=label,
            confidence=overall_conf,
            channels=channel_outputs,
            notes=notes,
            frame_status=frame_status.value,
        )
        return self._last


__all__ = ["AnomalyDetector"]
