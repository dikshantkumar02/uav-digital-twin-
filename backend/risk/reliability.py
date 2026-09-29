"""
PHASE 23 — Mission reliability risk layer.

A **read-only, advisory** view that sits on top of the existing
:class:`~backend.risk.MissionRiskCalculator` (PHASE 12) and
ingests the discrete signals the user explicitly named:

* engine health
* fault probability (engine / sensor / environment)
* RUL uncertainty
* mission phase
* remaining mission duration
* engine load
* environmental severity
* sensor confidence

The output is a 4-band :class:`ReliabilityBand` (``LOW``,
``MEDIUM``, ``HIGH``, ``UNKNOWN``) plus a structured
:class:`MissionReliabilityExplanation` with a single headline
sentence and a tuple of named, human-readable drivers. There
is no "mission success probability" number and there are no
control commands — the layer is explainable, conservative,
and read-only.

Design notes
------------

* The layer is **additive**. The PHASE 12 ``RiskAssessment``
  is one of the inputs (``risk_assessment``), and the
  reliability band is computed independently of the go/no-go
  decision — they can disagree (e.g. PHASE 12 says ``GO`` but
  reliability is ``MEDIUM`` because RUL uncertainty is high).
* When evidence is missing, the band is ``UNKNOWN`` (mirrors
  PHASE 12's ``INSUFFICIENT_DATA`` but surfaces to the user
  as the more conservative ``UNKNOWN``).
* The aggregator (:func:`aggregate_reliability`) is pure — no
  I/O, no mutable state, no per-instance attributes. The
  orchestrator (in the dashboard runner) owns the trend buffer
  if any.
* Per-signal weights live in :class:`ReliabilityConfig` and
  are operator-tunable via ``config/reliability.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.risk.types import (
    DriverSeverity,
    MissionPhase,
    RiskAssessment,
    RiskStatus,
    _clip,
)


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
class ReliabilityBand(str, Enum):
    """Coarse 4-band mission-reliability classification.

    * ``LOW`` — risk drivers nominal; mission may continue as planned.
    * ``MEDIUM`` — one or more elevated drivers; continue with awareness.
    * ``HIGH`` — multiple elevated drivers or one critical driver;
      consider return-to-base.
    * ``UNKNOWN`` — evidence insufficient to estimate; require a
      sensor check before any operational decision.

    ``UNKNOWN`` is the conservative answer: when in doubt, the
    layer says "I don't know" rather than overstating risk.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


# Band thresholds on the underlying 0..1 score. Keep these
# conservative — the user wants bands, not fine-grained numbers.
DEFAULT_BAND_THRESHOLD_MEDIUM: float = 0.30
DEFAULT_BAND_THRESHOLD_HIGH: float = 0.60

# Weights on the per-signal deficits. Conservative defaults;
# operator-tunable via ReliabilityConfig.
DEFAULT_WEIGHT_FAULT_PROBABILITY: float = 0.30
DEFAULT_WEIGHT_RUL_UNCERTAINTY: float = 0.15
DEFAULT_WEIGHT_ENGINE_LOAD: float = 0.20
DEFAULT_WEIGHT_ENV_SEVERITY: float = 0.15
DEFAULT_WEIGHT_SENSOR_AMBIGUITY: float = 0.10
DEFAULT_WEIGHT_PHASE_RISK: float = 0.10

# A fault probability above this is treated as a single critical
# driver (forces HIGH even if everything else is nominal).
FAULT_PROB_CRITICAL: float = 0.70

# An engine load above this is treated as a critical driver
# regardless of the other signals.
ENGINE_LOAD_CRITICAL: float = 0.85

# An environmental severity above this is treated as critical.
ENV_SEVERITY_CRITICAL: float = 0.80

# Below this many hours remaining, the mission is treated as
# "significant" for the explanation bullet — not a risk score
# change, just a named driver.
REMAINING_HOURS_SIGNIFICANT: float = 2.0

# Below this overall confidence, the band is forced to UNKNOWN
# (the no-fake-outputs rule).
MIN_CONFIDENCE_FOR_BAND: float = 0.20


# ---------------------------------------------------------------------
# Per-signal driver
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ReliabilityDriver:
    """A single per-signal contribution to the reliability band.

    The ``note`` is the human-readable bullet the dashboard
    surfaces under "Drivers:". ``confidence`` is the per-driver
    confidence (a low-confidence driver is *not* erased — it
    appears with a `[low confidence]` qualifier so the operator
    knows the layer is being honest about what it doesn't know).
    """

    signal: str
    value: float
    severity: DriverSeverity
    note: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "value": float(self.value),
            "severity": self.severity.value,
            "note": self.note,
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class MissionReliabilityExplanation:
    """Human-readable explanation of the reliability assessment.

    * ``headline`` is a single sentence (e.g. ``"Mission
      reliability is MEDIUM."``).
    * ``drivers`` is the ordered tuple of named, human-readable
      bullets. Order is the order the user sees; we keep the
      most concerning drivers first.
    """

    headline: str
    drivers: Tuple[ReliabilityDriver, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "drivers": [d.to_dict() for d in self.drivers],
        }


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MissionReliabilityInputs:
    """The discrete signal bundle the reliability aggregator reads.

    The dashboard assembles this once per tick from
    :class:`~backend.diagnostics.DiagnosticState` (PHASE 22) and
    :class:`~backend.risk.RiskAssessment` (PHASE 12). The layer
    itself never queries upstream components — the assembler
    does that.

    All numeric fields are in [0, 1] unless noted. ``None``
    signals are treated as missing and the band is forced to
    ``UNKNOWN``.
    """

    # The PHASE 12 risk assessment — one of the inputs.
    risk_assessment: RiskAssessment
    # Per-category fault probabilities from the classifier.
    fault_prob_engine: float
    fault_prob_sensor: float
    fault_prob_environment: float
    # RUL uncertainty: 0 = certain, 1 = fully uncertain. Pass
    # 1.0 when rul.is_uncertain is True.
    rul_uncertainty: float
    # Mission context.
    mission_phase: MissionPhase
    remaining_hours: Optional[float]
    # Composite engine load (PHASE 20 ``EngineInputs.engine_load``).
    engine_load: float
    # Composite environmental severity (0..1, normalised).
    env_severity: float
    # Sensor confidence (0..1, 1 = all channels reporting).
    sensor_confidence: float

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_prob_engine": float(self.fault_prob_engine),
            "fault_prob_sensor": float(self.fault_prob_sensor),
            "fault_prob_environment": float(self.fault_prob_environment),
            "rul_uncertainty": float(self.rul_uncertainty),
            "mission_phase": self.mission_phase.value,
            "remaining_hours": self.remaining_hours,
            "engine_load": float(self.engine_load),
            "env_severity": float(self.env_severity),
            "sensor_confidence": float(self.sensor_confidence),
            "risk_score": float(self.risk_assessment.risk_score),
            "risk_status": self.risk_assessment.status.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissionReliabilityInputs":
        """Build an inputs bundle from a dict (round-trip)."""
        from backend.risk.types import RiskAssessment, RiskLevel
        # Minimal RiskAssessment reconstruction — enough for the
        # aggregator to read ``.status``, ``.risk_score`` and
        # ``.mission_phase``. The richer fields are not consulted
        # by the reliability view.
        rk = RiskAssessment(
            time_s=0.0,
            risk_score=float(d.get("risk_score", 0.0)),
            risk_level=RiskLevel.LOW,
            status=RiskStatus(str(d.get("risk_status", "INSUFFICIENT_DATA"))),
            confidence=1.0,
            mission_phase=MissionPhase(str(d.get("mission_phase", "CRUISE"))),
        )
        return cls(
            risk_assessment=rk,
            fault_prob_engine=float(d.get("fault_prob_engine", 0.0)),
            fault_prob_sensor=float(d.get("fault_prob_sensor", 0.0)),
            fault_prob_environment=float(d.get("fault_prob_environment", 0.0)),
            rul_uncertainty=float(d.get("rul_uncertainty", 0.0)),
            mission_phase=MissionPhase(str(d.get("mission_phase", "CRUISE"))),
            remaining_hours=(
                float(d["remaining_hours"])
                if d.get("remaining_hours") is not None else None
            ),
            engine_load=float(d.get("engine_load", 0.0)),
            env_severity=float(d.get("env_severity", 0.0)),
            sensor_confidence=float(d.get("sensor_confidence", 1.0)),
        )


# ---------------------------------------------------------------------
# Operator-tunable config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ReliabilityConfig:
    """Operator-tunable weights and thresholds for the reliability
    aggregator.

    Defaults are conservative. The schema mirrors
    :class:`~backend.config.RiskConfig` so the layer slots into
    the existing ``config/reliability.yaml`` pattern.
    """

    band_threshold_medium: float = DEFAULT_BAND_THRESHOLD_MEDIUM
    band_threshold_high: float = DEFAULT_BAND_THRESHOLD_HIGH
    weight_fault_probability: float = DEFAULT_WEIGHT_FAULT_PROBABILITY
    weight_rul_uncertainty: float = DEFAULT_WEIGHT_RUL_UNCERTAINTY
    weight_engine_load: float = DEFAULT_WEIGHT_ENGINE_LOAD
    weight_env_severity: float = DEFAULT_WEIGHT_ENV_SEVERITY
    weight_sensor_ambiguity: float = DEFAULT_WEIGHT_SENSOR_AMBIGUITY
    weight_phase_risk: float = DEFAULT_WEIGHT_PHASE_RISK
    fault_prob_critical: float = FAULT_PROB_CRITICAL
    engine_load_critical: float = ENGINE_LOAD_CRITICAL
    env_severity_critical: float = ENV_SEVERITY_CRITICAL
    min_confidence_for_band: float = MIN_CONFIDENCE_FOR_BAND

    @classmethod
    def defaults(cls) -> "ReliabilityConfig":
        return cls()


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MissionReliabilityAssessment:
    """The per-tick output of the reliability layer.

    ``band`` is the 4-band classification. ``explanation`` is
    the human-readable breakdown. ``contributing_inputs`` is
    a per-signal summary (raw input values) so the dashboard
    can plot it. ``confidence`` is the overall confidence in
    the assessment.
    """

    time_s: float
    band: ReliabilityBand
    confidence: float
    explanation: MissionReliabilityExplanation
    contributing_inputs: Dict[str, float] = field(default_factory=dict)
    inputs_meta: Dict[str, str] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------
    @property
    def is_unknown(self) -> bool:
        return self.band is ReliabilityBand.UNKNOWN

    @property
    def is_high(self) -> bool:
        return self.band is ReliabilityBand.HIGH

    @property
    def is_low(self) -> bool:
        return self.band is ReliabilityBand.LOW

    @property
    def is_medium(self) -> bool:
        return self.band is ReliabilityBand.MEDIUM

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """The user-facing wire format.

        The output deliberately contains no mission-success
        probability number and no control commands. Only the
        band, the explanation, and the per-input raw values.
        """
        return {
            "time_s": float(self.time_s),
            "mission_risk": self.band.value,
            "confidence": float(self.confidence),
            "explanation": self.explanation.to_dict(),
            "contributing_inputs": dict(self.contributing_inputs),
            "inputs_meta": dict(self.inputs_meta),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------
def _band_for(
    score: float,
    *,
    threshold_medium: float,
    threshold_high: float,
) -> ReliabilityBand:
    """Map a 0..1 reliability score to a :class:`ReliabilityBand`."""
    s = _clip(score)
    if s >= threshold_high:
        return ReliabilityBand.HIGH
    if s >= threshold_medium:
        return ReliabilityBand.MEDIUM
    return ReliabilityBand.LOW


# ---------------------------------------------------------------------
# Per-signal scoring
# ---------------------------------------------------------------------
def _fault_probability_deficit(
    p_engine: float, p_sensor: float, p_environment: float,
) -> float:
    """Max of the three category probabilities, in [0, 1]."""
    return _clip(max(p_engine, p_sensor, p_environment))


def _sensor_ambiguity_deficit(sensor_confidence: float) -> float:
    """1 - sensor_confidence, clipped. We treat low confidence as
    a risk *amplifier* — the unknown unknown — not as a separate
    risk source."""
    return _clip(1.0 - float(sensor_confidence))


def _phase_risk_deficit(phase: MissionPhase) -> float:
    """Coarse mission-phase risk in [0, 1].

    We use the same intuition as the PHASE 12 PHASE_MODIFIERS
    (TAKEOFF / CLIMB / LANDING are higher-risk phases) but on
    a [0, 1] scale, not as a [0, 0.1] additive offset.
    """
    table = {
        MissionPhase.PRE_FLIGHT: 0.05,
        MissionPhase.TAKEOFF: 0.80,
        MissionPhase.CLIMB: 0.50,
        MissionPhase.CRUISE: 0.10,
        MissionPhase.DESCENT: 0.40,
        MissionPhase.LANDING: 0.70,
        MissionPhase.POST_MISSION: 0.05,
    }
    return _clip(table.get(phase, 0.10))


# ---------------------------------------------------------------------
# Driver builders (explanation bullets)
# ---------------------------------------------------------------------
def _driver_for_fault_probability(
    p_engine: float, p_sensor: float, p_environment: float,
    confidence: float,
) -> Optional[ReliabilityDriver]:
    """Build a named driver when fault probability is elevated."""
    p = _fault_probability_deficit(p_engine, p_sensor, p_environment)
    if p < 0.20:
        return None
    # Pick the largest contributor for the bullet's note.
    parts = []
    if p_engine >= 0.20:
        parts.append(f"engine {p_engine:.2f}")
    if p_sensor >= 0.20:
        parts.append(f"sensor {p_sensor:.2f}")
    if p_environment >= 0.20:
        parts.append(f"environment {p_environment:.2f}")
    note = f"fault probability elevated ({', '.join(parts)})"
    return ReliabilityDriver(
        signal="fault_probability",
        value=float(p),
        severity=(
            DriverSeverity.CRITICAL
            if p >= FAULT_PROB_CRITICAL else DriverSeverity.WARNING
        ),
        note=note,
        confidence=float(confidence),
    )


def _driver_for_rul_uncertainty(
    rul_uncertainty: float, confidence: float,
) -> Optional[ReliabilityDriver]:
    if rul_uncertainty < 0.30:
        return None
    return ReliabilityDriver(
        signal="rul_uncertainty",
        value=float(rul_uncertainty),
        severity=(
            DriverSeverity.CRITICAL
            if rul_uncertainty >= 0.70 else DriverSeverity.WARNING
        ),
        note="RUL uncertainty elevated",
        confidence=float(confidence),
    )


def _driver_for_health(
    risk_assessment: RiskAssessment,
) -> Optional[ReliabilityDriver]:
    """Surface the PHASE 12 health signal as a reliability driver."""
    label = risk_assessment.inputs_meta.get("health_label", "MISSING")
    if label == "HEALTHY" or label == "MISSING":
        return None
    if label == "CRITICAL":
        note = "health index CRITICAL — engine reliability degraded"
    else:
        note = "health index declining"
    return ReliabilityDriver(
        signal="health.label",
        value=float(risk_assessment.risk_score),
        severity=(
            DriverSeverity.CRITICAL
            if label == "CRITICAL" else DriverSeverity.WARNING
        ),
        note=note,
        confidence=float(risk_assessment.confidence),
    )


def _driver_for_residual(
    risk_assessment: RiskAssessment,
) -> Optional[ReliabilityDriver]:
    """Surface the PHASE 12 anomaly / residual signal as a driver."""
    label = risk_assessment.inputs_meta.get("anomaly_label", "MISSING")
    if label not in ("ANOMALY", "WARN"):
        return None
    note = "persistent engine performance residual"
    return ReliabilityDriver(
        signal="anomaly.label",
        value=float(risk_assessment.risk_score),
        severity=(
            DriverSeverity.CRITICAL
            if label == "ANOMALY" else DriverSeverity.WARNING
        ),
        note=note,
        confidence=float(risk_assessment.confidence),
    )


def _driver_for_engine_load(
    engine_load: float,
) -> Optional[ReliabilityDriver]:
    if engine_load < 0.40:
        return None
    return ReliabilityDriver(
        signal="engine_load",
        value=float(engine_load),
        severity=(
            DriverSeverity.CRITICAL
            if engine_load >= ENGINE_LOAD_CRITICAL else DriverSeverity.WARNING
        ),
        note=f"engine load elevated ({engine_load:.2f})",
        confidence=1.0,
    )


def _driver_for_env_severity(
    env_severity: float,
) -> Optional[ReliabilityDriver]:
    if env_severity < 0.30:
        return None
    return ReliabilityDriver(
        signal="env_severity",
        value=float(env_severity),
        severity=(
            DriverSeverity.CRITICAL
            if env_severity >= ENV_SEVERITY_CRITICAL else DriverSeverity.WARNING
        ),
        note=f"environmental severity elevated ({env_severity:.2f})",
        confidence=1.0,
    )


def _driver_for_remaining_hours(
    remaining_hours: Optional[float],
) -> Optional[ReliabilityDriver]:
    if remaining_hours is None:
        return None
    if remaining_hours > REMAINING_HOURS_SIGNIFICANT:
        return None
    return ReliabilityDriver(
        signal="remaining_hours",
        value=float(remaining_hours),
        severity=DriverSeverity.WARNING,
        note="remaining mission duration significant",
        confidence=1.0,
    )


def _driver_for_sensor_ambiguity(
    sensor_confidence: float,
) -> Optional[ReliabilityDriver]:
    ambiguity = _sensor_ambiguity_deficit(sensor_confidence)
    if ambiguity < 0.20:
        return None
    return ReliabilityDriver(
        signal="sensor_confidence",
        value=float(sensor_confidence),
        severity=DriverSeverity.WARNING,
        note=f"sensor confidence low ({sensor_confidence:.2f})",
        confidence=1.0,
    )


# ---------------------------------------------------------------------
# Pure aggregator
# ---------------------------------------------------------------------
def aggregate_reliability(
    inputs: MissionReliabilityInputs,
    *,
    config: Optional[ReliabilityConfig] = None,
    time_s: Optional[float] = None,
) -> MissionReliabilityAssessment:
    """Compute the per-tick :class:`MissionReliabilityAssessment`.

    Pure function: no I/O, no mutable state, no per-instance
    attributes. Two calls with the same inputs return equal
    assessments.

    Parameters
    ----------
    inputs:
        The :class:`MissionReliabilityInputs` bundle assembled
        by the dashboard.
    config:
        Optional operator-tunable :class:`ReliabilityConfig`.
        Defaults to :meth:`ReliabilityConfig.defaults`.
    time_s:
        Optional time stamp. Defaults to the
        :class:`~backend.risk.RiskAssessment` time.
    """
    cfg = config or ReliabilityConfig.defaults()
    ts = float(time_s if time_s is not None else inputs.risk_assessment.time_s)

    # --- Per-signal deficits (each in [0, 1]) ------------------------
    d_fault = _fault_probability_deficit(
        inputs.fault_prob_engine,
        inputs.fault_prob_sensor,
        inputs.fault_prob_environment,
    )
    d_rul_unc = _clip(float(inputs.rul_uncertainty))
    d_engine_load = _clip(float(inputs.engine_load))
    d_env_sev = _clip(float(inputs.env_severity))
    d_sensor_amb = _sensor_ambiguity_deficit(inputs.sensor_confidence)
    d_phase = _phase_risk_deficit(inputs.mission_phase)

    # --- Weighted score ----------------------------------------------
    raw = (
        cfg.weight_fault_probability * d_fault
        + cfg.weight_rul_uncertainty * d_rul_unc
        + cfg.weight_engine_load * d_engine_load
        + cfg.weight_env_severity * d_env_sev
        + cfg.weight_sensor_ambiguity * d_sensor_amb
        + cfg.weight_phase_risk * d_phase
    )
    score = _clip(raw)

    # --- Overall confidence ------------------------------------------
    # The minimum of: the PHASE 12 risk assessment confidence,
    # the sensor confidence, and a per-signal floor driven by
    # how many inputs are populated. We never fabricate.
    confidences: List[float] = [
        float(inputs.risk_assessment.confidence),
        float(inputs.sensor_confidence),
    ]
    overall_conf = _clip(min(confidences))

    # --- Critical-driver overrides -----------------------------------
    # A single critical driver (e.g. very high fault probability)
    # forces HIGH regardless of the composite score.
    forced_high = (
        d_fault >= cfg.fault_prob_critical
        or d_engine_load >= cfg.engine_load_critical
        or d_env_sev >= cfg.env_severity_critical
        or inputs.risk_assessment.status is RiskStatus.ABORT
    )

    # --- Build the driver list (explanation bullets) ----------------
    drivers: List[ReliabilityDriver] = []
    d_residual = _driver_for_residual(inputs.risk_assessment)
    if d_residual is not None:
        drivers.append(d_residual)
    d_health = _driver_for_health(inputs.risk_assessment)
    if d_health is not None:
        drivers.append(d_health)
    d_remaining = _driver_for_remaining_hours(inputs.remaining_hours)
    if d_remaining is not None:
        drivers.append(d_remaining)
    d_rul = _driver_for_rul_uncertainty(
        d_rul_unc, confidence=inputs.risk_assessment.confidence,
    )
    if d_rul is not None:
        drivers.append(d_rul)
    d_fault_drv = _driver_for_fault_probability(
        inputs.fault_prob_engine,
        inputs.fault_prob_sensor,
        inputs.fault_prob_environment,
        confidence=inputs.risk_assessment.confidence,
    )
    if d_fault_drv is not None:
        drivers.append(d_fault_drv)
    d_load = _driver_for_engine_load(inputs.engine_load)
    if d_load is not None:
        drivers.append(d_load)
    d_env = _driver_for_env_severity(inputs.env_severity)
    if d_env is not None:
        drivers.append(d_env)
    d_sensor = _driver_for_sensor_ambiguity(inputs.sensor_confidence)
    if d_sensor is not None:
        drivers.append(d_sensor)

    # Order: most concerning first. CRITICAL before WARNING, then
    # by signal value descending.
    _severity_rank = {
        DriverSeverity.CRITICAL: 0,
        DriverSeverity.WARNING: 1,
        DriverSeverity.INFO: 2,
    }
    drivers.sort(key=lambda d: (_severity_rank[d.severity], -d.value))

    # --- Decide the band --------------------------------------------
    if overall_conf < cfg.min_confidence_for_band:
        band = ReliabilityBand.UNKNOWN
    elif forced_high:
        band = ReliabilityBand.HIGH
    else:
        band = _band_for(
            score,
            threshold_medium=cfg.band_threshold_medium,
            threshold_high=cfg.band_threshold_high,
        )

    # --- Build the explanation --------------------------------------
    if band is ReliabilityBand.UNKNOWN:
        headline = (
            "Mission reliability is UNKNOWN — insufficient validated evidence."
        )
    else:
        headline = f"Mission reliability is {band.value}."

    explanation = MissionReliabilityExplanation(
        headline=headline,
        drivers=tuple(drivers),
    )

    # --- Per-input summary ------------------------------------------
    contributing = {
        "fault_probability": float(d_fault),
        "rul_uncertainty": float(d_rul_unc),
        "engine_load": float(d_engine_load),
        "env_severity": float(d_env_sev),
        "sensor_ambiguity": float(d_sensor_amb),
        "phase_risk": float(d_phase),
        "score": float(score),
    }

    # --- Audit / provenance -----------------------------------------
    inputs_meta = {
        "band": band.value,
        "score": f"{score:.3f}",
        "overall_confidence": f"{overall_conf:.3f}",
        "mission_phase": inputs.mission_phase.value,
        "risk_status": inputs.risk_assessment.status.value,
        "health_label": inputs.risk_assessment.inputs_meta.get(
            "health_label", "MISSING"
        ),
        "rul_status": inputs.risk_assessment.inputs_meta.get(
            "rul_status", "MISSING"
        ),
        "anomaly_label": inputs.risk_assessment.inputs_meta.get(
            "anomaly_label", "MISSING"
        ),
    }

    notes: List[str] = []
    if forced_high:
        notes.append("forced HIGH by a critical driver")
    if band is ReliabilityBand.UNKNOWN:
        notes.append(
            f"overall confidence {overall_conf:.2f} below floor "
            f"{cfg.min_confidence_for_band:.2f}"
        )

    return MissionReliabilityAssessment(
        time_s=ts,
        band=band,
        confidence=float(overall_conf),
        explanation=explanation,
        contributing_inputs=contributing,
        inputs_meta=inputs_meta,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_BAND_THRESHOLD_HIGH",
    "DEFAULT_BAND_THRESHOLD_MEDIUM",
    "DEFAULT_WEIGHT_ENGINE_LOAD",
    "DEFAULT_WEIGHT_ENV_SEVERITY",
    "DEFAULT_WEIGHT_FAULT_PROBABILITY",
    "DEFAULT_WEIGHT_PHASE_RISK",
    "DEFAULT_WEIGHT_RUL_UNCERTAINTY",
    "DEFAULT_WEIGHT_SENSOR_AMBIGUITY",
    "ENGINE_LOAD_CRITICAL",
    "ENV_SEVERITY_CRITICAL",
    "FAULT_PROB_CRITICAL",
    "MIN_CONFIDENCE_FOR_BAND",
    "REMAINING_HOURS_SIGNIFICANT",
    "MissionReliabilityAssessment",
    "MissionReliabilityExplanation",
    "MissionReliabilityInputs",
    "ReliabilityBand",
    "ReliabilityConfig",
    "ReliabilityDriver",
    "aggregate_reliability",
]
