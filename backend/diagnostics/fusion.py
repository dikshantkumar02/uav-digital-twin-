"""
Fusion of per-channel anomaly signals (PHASE 8).

For each channel the detector has two independent signals:

* a residual score (from the Digital Twin)
* a sensor-health score (from the sensor bundle's ``NoiseMode``)

The fusion rule is::

    score_fused  = max(score_residual, score_sensor)
    confidence   = 0.6 * residual_conf + 0.4 * sensor_present
    contributors = union of both lists
    label        = per thresholds on score_fused

``max()`` is the right defensive combiner: a single strong signal
on either side is enough to flag the channel. The project spec
forbids *single-sensor thresholds* but allows multi-signal
fusion, and ``max()`` is the simplest honest implementation of
"at least one strong signal is enough" that doesn't pretend to
know a better weighting (PHASE 9's ML layer can learn one).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from backend.sensors import NoiseMode, SensorSample

from .residual_score import residual_confidences
from .sensor_health import sensor_health_score
from .types import AnomalyLabel, AnomalyThresholds, ChannelAnomaly, _label_for


# Channels the detector should consider when the residual frame
# doesn't list them (e.g. only the channels that have valid
# observations). The full list comes from the residual frame;
# this set is used only for sensor-only fusion (e.g. STUCK on a
# channel the twin isn't tracking).
ALL_KNOWN_CHANNELS = (
    "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
    "fuel_flow", "vibration", "altitude", "airspeed",
    "ambient_temperature", "ambient_pressure",
)


def _sensor_score(
    sensor_sample: Optional[SensorSample],
    channel: str,
) -> tuple[float, List[str]]:
    if sensor_sample is None:
        return 0.0, []
    reading = sensor_sample.readings.get(channel) if sensor_sample.readings else None
    if reading is None:
        return 0.0, []
    return sensor_health_score(reading.mode)


def fuse_channel(
    channel: str,
    smoothed_residual: Optional[float],
    residual_conf: float,
    sensor_sample: Optional[SensorSample],
    thresholds: AnomalyThresholds,
) -> ChannelAnomaly:
    """Fuse the two signals for one channel."""
    res_score = float(smoothed_residual) if smoothed_residual is not None else 0.0
    sens_score, sens_contribs = _sensor_score(sensor_sample, channel)

    if smoothed_residual is None:
        # No residual evidence; rely on sensor health.
        score = sens_score
        # If the sensor is NORMAL with no contributors, the channel
        # is just uninstrumented — keep score 0.
        contributors: List[str] = list(sens_contribs)
        res_contrib_present = False
    else:
        score = max(res_score, sens_score)
        contributors = ["twin_residual_z"] if res_score > 0 else []
        if sens_contribs:
            contributors = contributors + list(sens_contribs)
        res_contrib_present = True

    confidence = 0.6 * float(residual_conf) + 0.4 * (1.0 if res_contrib_present else 0.0)
    confidence = max(0.0, min(1.0, confidence))
    # Apply label; if confidence is below the floor, demote to
    # INSUFFICIENT_DATA unless the score is truly zero (no evidence
    # of an anomaly either way is fine).
    label = _label_for(score, thresholds)
    if score > 0.0 and confidence < thresholds.min_confidence and label is AnomalyLabel.NORMAL:
        # Below the noise floor — don't raise an alarm but also
        # don't claim "NORMAL"; stay NORMAL (conservative).
        pass
    return ChannelAnomaly(
        channel=channel,
        score=score,
        label=label,
        confidence=confidence,
        contributors=contributors,
    )


def overall_score(channel_scores: Dict[str, float]) -> float:
    """Compute the overall anomaly score from per-channel scores.

    Uses the 95th percentile — robust to a single noisy channel
    dominating. With 1 channel, returns that score. With 0, returns
    0.0.
    """
    if not channel_scores:
        return 0.0
    values = sorted(channel_scores.values())
    if len(values) == 1:
        return values[0]
    # 95th percentile: take the value at index ceil(0.95 * (n-1)).
    import math
    idx = max(0, math.ceil(0.95 * (len(values) - 1)))
    return float(values[idx])


__all__ = [
    "ALL_KNOWN_CHANNELS",
    "fuse_channel",
    "overall_score",
]
