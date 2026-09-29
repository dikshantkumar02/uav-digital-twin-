"""
PHASE 22 — Diagnostic Assembler.

The :class:`DiagnosticAssembler` is the single funnel that
turns the per-tick outputs of PHASES 6-12 into a unified
:class:`DiagnosticState`. It exists for two reasons:

1. **Single source of truth.** The dashboard, the API and
   the replay all read the same object. They do not, and
   must not, call the upstream components directly. This
   means any change to a wire field happens in one place.

2. **Pre-aggregation of derived fields.** Two of the 12
   user-facing fields are *derived* from a per-tick history
   that the upstream modules don't track: ``residual_trend``
   (slope of mean |z| over the last ``TREND_WINDOW`` ticks)
   and ``uncertainty`` (blended confidence gap from the
   twin, anomaly detector, classifier, health, rul, and
   risk). The assembler maintains the history and produces
   the derived values.

The assembler is **stateful** because of the residual-trend
sliding window. It is not, however, a re-implementation of
the upstream calculators — those are still the single
source of truth for their own fields. The assembler only
*reads* them and *projects* them to the diagnostic view.

Usage::

    asm = DiagnosticAssembler()
    ...
    diagnostic = asm.assemble(
        time_s=sample.time_s,
        sample=sample,
        twin_state=twin_state,
        residual=residual,
        anomaly=anomaly,
        classification=classification,
        health=health,
        rul=rul,
        risk=risk,
    )
    wire = diagnostic.to_dict()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from backend.digital_twin import ResidualFrame, TwinState
from backend.diagnostics.sensor_health import sensor_health_score
from backend.faults import FaultClass
from backend.ml import CalibrationStatus
from backend.sensors import SENSOR_CHANNELS, NoiseMode, SensorSample

from .diagnostic_state import (
    DiagnosticState,
    SensorQuality,
    TREND_EPSILON,
    TREND_WINDOW,
    TrendDirection,
    _clip01,
    _engine_state_label,
    _trend_for,
)

# Local type-only imports — these are only used in annotations and
# method bodies, never at module-load time. Avoids a circular
# import where backend.diagnostics.__init__ would otherwise need
# backend.health / backend.rul / backend.risk to be fully loaded
# before AnomalyAssessment / HealthIndex / etc. can be referenced.
if TYPE_CHECKING:
    from backend.diagnostics import AnomalyAssessment
    from backend.environment import EnvironmentState
    from backend.health import HealthIndex
    from backend.ml import FaultClassification
    from backend.risk import RiskAssessment
    from backend.rul import RulEstimate
    from backend.simulation import EngineState


# ---------------------------------------------------------------------
# Grouping FaultClass → category buckets
# ---------------------------------------------------------------------
# The user's wire format asks for per-category fault probabilities
# (engine / sensor / environment / unknown / healthy). The PHASE 7
# enum is finer-grained; we group at presentation time.
_FAULT_GROUP_BY_CLASS: Dict[FaultClass, str] = {
    FaultClass.HEALTHY: "healthy",
    FaultClass.ENGINE_DEGRADATION: "engine",
    FaultClass.OVERHEATING: "engine",
    FaultClass.LUBRICATION_PRESSURE_ANOMALY: "engine",
    FaultClass.VIBRATION_ENGINE_ANOMALY: "engine",
    FaultClass.PERFORMANCE_LOSS: "engine",
    FaultClass.SENSOR_FAULT: "sensor",
    FaultClass.ENVIRONMENTAL_DISTURBANCE: "environment",
    FaultClass.UNKNOWN_INSUFFICIENT_EVIDENCE: "unknown",
}


def _coarse_quality(score: float) -> SensorQuality:
    """Map a 0..1 sensor-health score to a coarse quality enum.

    Thresholds are aligned with the existing health-label cuts
    (PHASE 10) so the dashboard colour map stays consistent.
    """
    if score >= 0.80:
        return SensorQuality.NOMINAL
    if score >= 0.55:
        return SensorQuality.DEGRADED
    if score >= 0.20:
        return SensorQuality.UNRELIABLE
    return SensorQuality.INVALID


# ---------------------------------------------------------------------
# Per-channel residual-trend history
# ---------------------------------------------------------------------
@dataclass
class _ChannelResidualHistory:
    """Sliding window of mean |z| per channel.

    A ring buffer per channel — the ``DiagnosticAssembler``
    tracks the last ``TREND_WINDOW`` mean |z| samples so the
    trend can be computed as a finite difference between the
    first and last sample in the window.
    """

    samples: Deque[Tuple[float, float]] = field(default_factory=deque)  # (time_s, mean_abs_z)

    def append(self, time_s: float, mean_abs_z: float) -> None:
        if len(self.samples) >= TREND_WINDOW:
            self.samples.popleft()
        self.samples.append((float(time_s), float(mean_abs_z)))

    def slope(self) -> Optional[float]:
        """Return the slope of mean |z| over the window.

        ``None`` if the window has fewer than two samples.
        The slope is computed in ``(z, time)`` units; the
        diagnostic state's trend thresholds (``TREND_EPSILON``)
        are applied at the trend-decision step, not here.
        """
        if len(self.samples) < 2:
            return None
        t0, z0 = self.samples[0]
        t1, z1 = self.samples[-1]
        if t1 <= t0:
            return 0.0
        # Normalise the slope so it lands in roughly [-1, 1] for
        # typical z values: divide by the window's time span, so
        # the result is "delta z per second". The threshold check
        # below (|slope| * dt > TREND_EPSILON) is robust to the
        # exact unit choice.
        return (z1 - z0) / (t1 - t0)


# ---------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------
class DiagnosticAssembler:
    """Build a :class:`DiagnosticState` from per-tick components.

    Stateful: maintains a per-channel residual-trend history. All
    other fields are projected from the inputs without mutation.
    """

    def __init__(self, trend_window: int = TREND_WINDOW) -> None:
        self._trend_window = int(trend_window)
        self._history: Dict[str, _ChannelResidualHistory] = {}

    # ------------------------------------------------------------------
    # Trend helpers
    # ------------------------------------------------------------------
    def _update_residual_trend(
        self,
        residual: ResidualFrame,
    ) -> Dict[str, Any]:
        """Compute the residual-trend block for this tick.

        Two things go into the dict:

        * ``per_channel`` — per-channel ``direction`` + ``slope``
          + ``current_mean_abs_z`` (so consumers can plot it).
        * ``overall`` — the worst-direction channel drives the
          dashboard colour. If we have no data, we report
          ``INSUFFICIENT_DATA``.
        """
        per_channel: Dict[str, Any] = {}
        worst_direction = TrendDirection.STABLE
        worst_severity: float = 0.0

        for ch, cr in residual.residuals.items():
            hist = self._history.setdefault(ch, _ChannelResidualHistory())
            mean_abs_z = abs(float(cr.z_score))
            hist.append(residual.time_s, mean_abs_z)
            slope = hist.slope()
            # The "current vs reference" comparison is what the
            # ``_trend_for`` helper does — it maps a finite
            # difference to a direction. We use the slope per
            # unit time, scaled by the trend window length so
            # the threshold check is independent of dt.
            scaled_slope = (slope or 0.0) * float(self._trend_window) * 0.1
            direction = _trend_for(scaled_slope, 0.0)
            per_channel[ch] = {
                "direction": direction.value,
                "slope_per_s": slope,
                "current_mean_abs_z": mean_abs_z,
                "n_samples": len(hist.samples),
            }
            # Track the worst direction so the overall trend is
            # the most pessimistic channel. Worsening > Stable >
            # Improving > Insufficient. We rank by the magnitude
            # of the slope.
            if direction is TrendDirection.WORSENING:
                if abs(slope or 0.0) > worst_severity:
                    worst_direction = direction
                    worst_severity = abs(slope or 0.0)
            elif direction is TrendDirection.IMPROVING:
                if worst_direction is TrendDirection.STABLE and abs(slope or 0.0) > worst_severity:
                    worst_direction = direction
                    worst_severity = abs(slope or 0.0)

        if not per_channel:
            worst_direction = TrendDirection.INSUFFICIENT_DATA

        return {
            "overall": worst_direction.value,
            "per_channel": per_channel,
            "window": self._trend_window,
        }

    # ------------------------------------------------------------------
    # Sensor quality
    # ------------------------------------------------------------------
    @staticmethod
    def _sensor_quality(
        sample: SensorSample,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-channel sensor health view.

        For every channel in :data:`SENSOR_CHANNELS` we emit::

            { "mode": "NORMAL" | "DRIFTING" | ... | None,
              "score": 0..1,        # 0 = INVALID, 1 = perfect
              "quality": "NOMINAL" | "DEGRADED" | "UNRELIABLE" | "INVALID",
              "contributors": [str, ...] }

        Channels that did not produce a reading (dropout) are
        reported as score 0.0 / INVALID. NORMAL mode is always
        NOMINAL regardless of the raw score, because in the
        PHASE 8 sensor_health table NORMAL has score 0.0
        (``0 = no penalty``) — that does not mean the channel
        is faulty.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for ch in SENSOR_CHANNELS:
            r = sample.readings.get(ch)
            if r is None or r.value is None:
                out[ch] = {
                    "mode": None,
                    "score": 0.0,
                    "quality": SensorQuality.INVALID.value,
                    "contributors": ["no_reading"],
                }
                continue
            score, contributors = sensor_health_score(r.mode)
            if r.mode is NoiseMode.NORMAL:
                quality = SensorQuality.NOMINAL
            else:
                quality = _coarse_quality(score)
            out[ch] = {
                "mode": r.mode.value,
                "score": float(score),
                "quality": quality.value,
                "contributors": list(contributors),
            }
        return out

    # ------------------------------------------------------------------
    # Fault probabilities — group-keyed
    # ------------------------------------------------------------------
    @staticmethod
    def _fault_probabilities(
        classification: Optional[FaultClassification],
    ) -> Dict[str, float]:
        """Group-keyed fault probability vector.

        The PHASE 9 classifier emits a probability per
        :class:`FaultClass`. The wire format the user asked for
        wants them grouped into 5 categories:

        * ``engine`` — engine-truth-layer faults
        * ``sensor`` — sensor-side faults
        * ``environment`` — environment-driven faults
        * ``unknown`` — uncalibrated / unknown
        * ``healthy`` — no-fault baseline
        """
        groups: Dict[str, float] = {
            "engine": 0.0,
            "sensor": 0.0,
            "environment": 0.0,
            "unknown": 0.0,
            "healthy": 0.0,
        }
        if classification is None:
            return groups
        # If the classifier hasn't calibrated, we surface the
        # unknown bucket as 1.0 — no fake numbers.
        if classification.status is not CalibrationStatus.CALIBRATED:
            groups["unknown"] = 1.0
            return groups
        for fc, p in classification.probabilities.items():
            grp = _FAULT_GROUP_BY_CLASS.get(fc, "unknown")
            groups[grp] += float(p)
        # Renormalise so the groups sum to 1.0 (the per-class
        # vector already did, but adding across groups should
        # still leave the total at 1.0 in a calibrated model).
        total = sum(groups.values())
        if total > 0.0:
            for k in list(groups.keys()):
                groups[k] = groups[k] / total
        return groups

    # ------------------------------------------------------------------
    # RUL block
    # ------------------------------------------------------------------
    @staticmethod
    def _rul_block(
        rul: Optional[RulEstimate],
    ) -> Dict[str, Any]:
        """Compact RUL view.

        The wire format only needs ``estimate``, ``lower``,
        ``upper`` and ``confidence``; the rich view also
        surfaces ``status`` and ``trend`` so the API caller
        can drive a go/no-go display without re-querying.
        """
        if rul is None:
            return {
                "estimate": None,
                "lower": None,
                "upper": None,
                "confidence": 0.0,
                "status": "RUL_UNCERTAIN",
                "trend": TrendDirection.INSUFFICIENT_DATA.value,
            }
        return {
            "estimate": float(rul.tte_hours_central),
            "lower": float(rul.tte_hours_lower),
            "upper": float(rul.tte_hours_upper),
            "confidence": float(rul.confidence),
            "status": rul.status.value,
            "trend": rul.trend.value,
        }

    # ------------------------------------------------------------------
    # Health index block
    # ------------------------------------------------------------------
    @staticmethod
    def _health_block(
        health: Optional[HealthIndex],
    ) -> Dict[str, Any]:
        if health is None:
            return {
                "overall_score": 0.0,
                "overall_label": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "trend": TrendDirection.INSUFFICIENT_DATA.value,
                "subsystems": {},
            }
        return {
            "overall_score": float(health.overall_score),
            "overall_label": health.overall_label.value,
            "confidence": float(health.confidence),
            "trend": health.trend.value,
            "subsystems": {k: v.to_dict() for k, v in health.subsystems.items()},
        }

    # ------------------------------------------------------------------
    # Risk block
    # ------------------------------------------------------------------
    @staticmethod
    def _risk_block(
        risk: Optional[RiskAssessment],
    ) -> Dict[str, Any]:
        if risk is None:
            return {
                "risk_score": 0.0,
                "risk_level": "LOW",
                "status": "INSUFFICIENT_DATA",
                "mission_phase": "PRE_FLIGHT",
                "confidence": 0.0,
            }
        return {
            "risk_score": float(risk.risk_score),
            "risk_level": risk.risk_level.value,
            "status": risk.status.value,
            "mission_phase": risk.mission_phase.value,
            "confidence": float(risk.confidence),
            "drivers": [d.to_dict() for d in risk.drivers],
            "recommendations": list(risk.recommendations),
        }

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------
    @staticmethod
    def _uncertainty(
        residual: ResidualFrame,
        anomaly: Optional[AnomalyAssessment],
        classification: Optional[FaultClassification],
        health: Optional[HealthIndex],
        rul: Optional[RulEstimate],
        risk: Optional[RiskAssessment],
    ) -> float:
        """Single 0..1 number — the confidence gap.

        Weighted blend of the per-component confidences. If any
        component is missing, the blend is shifted toward
        INSUFFICIENT_DATA (uncertainty closer to 1.0).
        """
        # Per-component "informativeness" (the complement of
        # the uncertainty that component itself reports).
        comps: List[float] = []
        comps.append(_clip01(residual.overall_confidence))
        if anomaly is not None:
            comps.append(_clip01(anomaly.confidence))
        if classification is not None:
            if classification.status is CalibrationStatus.CALIBRATED:
                comps.append(_clip01(classification.confidence))
            else:
                comps.append(0.0)
        if health is not None:
            comps.append(_clip01(health.confidence))
        if rul is not None:
            comps.append(_clip01(rul.confidence))
        if risk is not None:
            comps.append(_clip01(risk.confidence))
        if not comps:
            return 1.0
        # Mean informativeness → flip to uncertainty.
        return _clip01(1.0 - (sum(comps) / len(comps)))

    # ------------------------------------------------------------------
    # Evidence & data quality
    # ------------------------------------------------------------------
    @staticmethod
    def _evidence(
        sample: SensorSample,
        anomaly: Optional[AnomalyAssessment],
        health: Optional[HealthIndex],
        risk: Optional[RiskAssessment],
    ) -> Tuple[str, ...]:
        """Compact, human-readable evidence trail.

        A short list of strings suitable for direct display in
        the dashboard's "Why?" panel. We deliberately cap the
        list at 8 items so the wire format stays small.
        """
        out: List[str] = []
        if anomaly is not None:
            label = anomaly.overall_label.value
            if label not in ("OK", "NORMAL", "INSUFFICIENT_DATA"):
                out.append(f"anomaly: {label} ({anomaly.overall_score:.2f})")
            for note in anomaly.notes[:3]:
                out.append(f"anomaly note: {note}")
        if health is not None:
            if health.overall_label.value not in ("HEALTHY", "INSUFFICIENT_DATA"):
                out.append(
                    f"health: {health.overall_label.value} "
                    f"(score={health.overall_score:.2f})"
                )
        if risk is not None:
            if risk.status.value not in ("GO", "INSUFFICIENT_DATA"):
                out.append(
                    f"risk: {risk.status.value} "
                    f"(score={risk.risk_score:.2f})"
                )
        # Dropout summary (capped).
        n_drop = sum(1 for ch in SENSOR_CHANNELS if sample.is_dropped(ch))
        if n_drop > 0:
            out.append(f"sensor dropouts: {n_drop} channel(s)")
        return tuple(out[:8])

    @staticmethod
    def _data_quality(sample: SensorSample) -> float:
        """Fraction of channels that produced a non-dropped reading.

        A score of 1.0 means every channel reported this tick;
        0.0 means every channel is dropped. The 0.5 / 0.85 /
        1.0 colour bands in the dashboard follow the same
        pattern as the health-label cuts.
        """
        n_total = len(SENSOR_CHANNELS)
        if n_total == 0:
            return 0.0
        n_good = sum(1 for ch in SENSOR_CHANNELS if not sample.is_dropped(ch))
        return _clip01(n_good / n_total)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def assemble(
        self,
        *,
        time_s: float,
        sample: SensorSample,
        twin_state: Optional[TwinState],
        residual: ResidualFrame,
        anomaly: Optional[AnomalyAssessment],
        classification: Optional[FaultClassification],
        health: Optional[HealthIndex],
        rul: Optional[RulEstimate],
        risk: Optional[RiskAssessment],
        engine: Optional[EngineState] = None,
        env: Optional[EnvironmentState] = None,
    ) -> DiagnosticState:
        """Build a :class:`DiagnosticState` from the per-tick chain.

        All components are *read* — the assembler never mutates
        upstream state. The only state the assembler maintains
        is the residual-trend history (sliding window).
        """
        # 1) observed_state: per-channel reading (None for dropout).
        observed: Dict[str, Any] = {
            ch: sample.readings[ch].value
            for ch in SENSOR_CHANNELS
            if ch in sample.readings
        }
        # 2) expected_state: per-channel twin prediction.
        if twin_state is not None:
            twin_d = twin_state.to_dict()
            # Project twin fields back to channel names.
            expected: Dict[str, float] = {}
            for ch in SENSOR_CHANNELS:
                # The twin's channel → field map is maintained
                # in digital_twin.CHANNEL_TO_STATE. We don't
                # import it here to keep this module
                # dependency-free; the twin's to_dict uses the
                # full engine-field naming, so we surface the
                # ones we know how to map.
                if ch == "rpm":
                    expected[ch] = float(twin_d.get("rpm", 0.0))
                elif ch == "egt":
                    expected[ch] = float(twin_d.get("egt_c", 0.0))
                elif ch == "cht":
                    expected[ch] = float(twin_d.get("cht_c", 0.0))
                elif ch == "oil_pressure":
                    expected[ch] = float(twin_d.get("oil_pressure_psi", 0.0))
                elif ch == "oil_temperature":
                    expected[ch] = float(twin_d.get("oil_temperature_c", 0.0))
                elif ch == "fuel_flow":
                    expected[ch] = float(twin_d.get("fuel_flow_lph", 0.0))
                elif ch == "vibration":
                    expected[ch] = float(twin_d.get("vibration_rms_g", 0.0))
                elif ch == "altitude":
                    expected[ch] = float(twin_d.get("altitude_m", 0.0))
                elif ch == "airspeed":
                    expected[ch] = float(twin_d.get("airspeed_mps", 0.0))
                elif ch == "ambient_temperature":
                    expected[ch] = float(twin_d.get("ambient_temperature_c", 0.0))
                elif ch == "ambient_pressure":
                    expected[ch] = float(twin_d.get("ambient_pressure_pa", 0.0))
                # imu_accel is a derived / sensor-only channel
                # — the twin doesn't model it.
        else:
            expected = {}

        # 3) residual: per-channel block.
        residual_block: Dict[str, Dict[str, Any]] = {
            ch: {
                "observed": float(cr.observed) if cr.observed is not None else None,
                "predicted": float(cr.predicted),
                "residual": float(cr.residual),
                "z_score": float(cr.z_score),
                "confidence": float(cr.confidence),
                "in_bounds": bool(cr.in_bounds),
            }
            for ch, cr in residual.residuals.items()
        }

        # 4) residual_trend: state-tracked.
        trend_block = self._update_residual_trend(residual)

        # 5) sensor_quality: per-channel.
        sensor_block = self._sensor_quality(sample)

        # 6) environment_context: compact env snapshot.
        if env is not None:
            env_d = env.to_dict()
            env_block: Dict[str, Any] = {
                "altitude_m": float(env_d.get("altitude_m", 0.0)),
                "airspeed_mps": float(env_d.get("airspeed_mps", 0.0)),
                "throttle": float(env_d.get("throttle", 0.0)),
                "ambient_temperature_c": float(env_d.get("temperature_c", 0.0)),
                "ambient_pressure_pa": float(env_d.get("pressure_pa", 0.0)),
                "wind_mps": float(env_d.get("total_w_mps", 0.0)),
                "turbulence_intensity": float(env_d.get("turbulence_w_mps", 0.0)),
            }
        else:
            env_block = {}

        # 7) ai_anomaly_score: single number.
        ai_score = float(anomaly.overall_score) if anomaly is not None else 0.0

        # 8) fault_probabilities: group-keyed.
        fault_probs = self._fault_probabilities(classification)

        # 9) health_index: from health.
        health_block = self._health_block(health)

        # 10) rul: compact block.
        rul_block = self._rul_block(rul)

        # 11) uncertainty: blended confidence gap.
        uncertainty_val = self._uncertainty(
            residual=residual,
            anomaly=anomaly,
            classification=classification,
            health=health,
            rul=rul,
            risk=risk,
        )

        # 12) mission_risk: from risk.
        risk_block = self._risk_block(risk)

        # engine_state: coarse label.
        if health is not None:
            engine_state = _engine_state_label(health)
        else:
            engine_state = "UNKNOWN"

        # evidence + data_quality.
        evidence = self._evidence(sample, anomaly, health, risk)
        data_quality = self._data_quality(sample)

        return DiagnosticState(
            time_s=float(time_s),
            observed_state=observed,
            expected_state=expected,
            residual=residual_block,
            residual_trend=trend_block,
            sensor_quality=sensor_block,
            environment_context=env_block,
            ai_anomaly_score=ai_score,
            fault_probabilities=fault_probs,
            health_index=health_block,
            rul=rul_block,
            uncertainty=uncertainty_val,
            mission_risk=risk_block,
            evidence=evidence,
            data_quality=data_quality,
            engine_state=engine_state,
            raw_engine=engine,
            raw_environment=env,
            raw_anomaly=anomaly,
            raw_classification=classification,
            raw_health=health,
            raw_rul=rul,
            raw_risk=risk,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop the residual-trend history (used between scenarios)."""
        self._history.clear()


__all__ = ["DiagnosticAssembler"]
