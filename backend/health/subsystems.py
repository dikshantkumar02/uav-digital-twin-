"""
Per-subsystem health scorers (PHASE 10).

Each scorer is a **pure function** that takes the engine state (and,
for the ``SENSORS`` subsystem, the PHASE 8 anomaly assessment) and
returns a :class:`~backend.health.types.SubsystemHealth`. None of the
scorers mutate state — the orchestrator (:class:`HealthIndexCalculator`)
owns the trend and aggregation.

Scoring conventions
-------------------

* Channels with a ``[min, max]`` envelope score
  ``1 - max(0, (x - max) / (max - min))`` and similarly below the
  min. Clamped to [0, 1].
* Channels with a single ``nominal`` value (oil pressure) score
  ``1 - |x - nominal| / max(|nominal - min|, |max - nominal|)``.
* ``wear`` is ``1 - wear`` directly.
* Missing channels (no observation) return 0.0 with confidence 0.0
  (conservative — the calculator interprets this as
  INSUFFICIENT_DATA when *all* channels in the subsystem are
  missing).
* The ``SENSORS`` subsystem uses the per-channel anomaly scores
  from :class:`~backend.diagnostics.AnomalyAssessment` directly.
"""

from __future__ import annotations

from typing import Optional

from backend.config import EngineConfig
from backend.diagnostics import AnomalyAssessment, AnomalyLabel
from backend.simulation import EngineState

from .types import HealthLabel, Subsystem, SubsystemHealth


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _envelope_score(
    value: Optional[float],
    *,
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Score a value against a [lo, hi] envelope.

    Convention: ``[lo, hi]`` is the **permitted operating range**.
    Values **strictly inside** the range score 1.0; values **at**
    the limits already score 0.0 (at the redline the engine is
    critical, not healthy). The engine.yaml ``limits`` block
    contains redline values — being at the limit is therefore a
    critical reading.

    Returns ``(score, confidence)``. ``confidence`` is 0.0 if the
    value is missing, 1.0 if the envelope is well-defined
    (``hi > lo``), and 0.5 if the envelope is degenerate.
    """
    if value is None:
        return 0.0, 0.0
    if hi <= lo:
        # Degenerate envelope — return a neutral score with low confidence.
        return 0.5, 0.5
    v = float(value)
    if lo < v < hi:
        return 1.0, 1.0
    return 0.0, 1.0


def _nominal_score(
    value: Optional[float],
    *,
    nominal: float,
    lo: float,
    hi: float,
) -> tuple[float, float]:
    """Score a value against a single nominal target in [lo, hi].

    The score drops linearly as the value moves away from
    ``nominal``; at ``lo`` or ``hi`` the score is 0.0. Values
    outside [lo, hi] also score 0.0.
    """
    if value is None:
        return 0.0, 0.0
    v = float(value)
    span_lo = max(1e-9, abs(nominal - lo))
    span_hi = max(1e-9, abs(hi - nominal))
    if v < lo or v > hi:
        # Outside the envelope — return 0.0 with full confidence.
        return 0.0, 1.0
    if v == nominal:
        return 1.0, 1.0
    if v < nominal:
        # Score linearly from 1.0 at nominal to 0.0 at lo (exclusive).
        return _clip01(1.0 - (nominal - v) / span_lo), 1.0
    # Score linearly from 1.0 at nominal to 0.0 at hi (exclusive).
    return _clip01(1.0 - (v - nominal) / span_hi), 1.0


def _aggregate(
    scored: list[tuple[float, float]],
    contributors_idx: list[int],
    names: list[str],
) -> tuple[float, float, list[str]]:
    """Combine per-channel (score, confidence) into a subsystem score.

    The subsystem score is the **mean** of its per-channel scores,
    weighted by per-channel confidence. Channels with zero
    confidence are excluded. If every channel has zero confidence,
    the subsystem reports ``(0.0, 0.0)`` which the orchestrator
    interprets as INSUFFICIENT_DATA.
    """
    if not scored:
        return 0.0, 0.0, []
    total_w = sum(max(0.0, conf) for _, conf in scored)
    if total_w <= 0.0:
        return 0.0, 0.0, []
    total = sum(score * max(0.0, conf) for score, conf in scored)
    score = _clip01(total / total_w)
    conf = _clip01(total_w / float(len(scored)))
    contributors = [
        names[i] for i in contributors_idx
        if i < len(names)
    ]
    return float(score), float(conf), contributors


# ---------------------------------------------------------------------
# THERMAL — EGT, CHT, oil temperature
# ---------------------------------------------------------------------
def score_thermal(
    state: Optional[EngineState],
    cfg: EngineConfig,
) -> SubsystemHealth:
    """Score the thermal subsystem.

    Channels:
      * ``egt_c`` vs ``limits.egt_max_c`` (max-only envelope).
      * ``cht_c`` vs ``limits.cht_max_c`` (max-only envelope).
      * ``oil_temperature_c`` vs ``[oil_temp_min_c, oil_temp_max_c]``.
    """
    names = ["egt_c", "cht_c", "oil_temperature_c"]
    if state is None:
        return SubsystemHealth(
            subsystem=Subsystem.THERMAL,
            score=0.0, confidence=0.0,
            notes=["no engine state provided"],
        )

    egt = getattr(state, "egt_c", None)
    cht = getattr(state, "cht_c", None)
    oil_t = getattr(state, "oil_temperature_c", None)

    # EGT/CHT use a generous lower bound of 0C; the floor rarely
    # matters in flight but it makes the score well-defined.
    egt_lo, egt_hi = 0.0, float(cfg.limits.egt_max_c)
    cht_lo, cht_hi = 0.0, float(cfg.limits.cht_max_c)
    oil_lo = float(cfg.limits.oil_temp_min_c)
    oil_hi = float(cfg.limits.oil_temp_max_c)

    s_egt, c_egt = _envelope_score(egt, lo=egt_lo, hi=egt_hi)
    s_cht, c_cht = _envelope_score(cht, lo=cht_lo, hi=cht_hi)
    s_oil, c_oil = _envelope_score(oil_t, lo=oil_lo, hi=oil_hi)

    scored = [(s_egt, c_egt), (s_cht, c_cht), (s_oil, c_oil)]
    contributors_idx: list[int] = []
    # A channel is a contributor if its score is < 1.0 and it has
    # some confidence.
    for i, (s, c) in enumerate(scored):
        if c > 0.0 and s < 1.0:
            contributors_idx.append(i)

    score, conf, contributors = _aggregate(scored, contributors_idx, names)
    notes: list[str] = []
    if score < 0.55 and conf > 0.0:
        notes.append("thermal subsystem out of envelope")
    return SubsystemHealth(
        subsystem=Subsystem.THERMAL,
        score=score, confidence=conf,
        contributors=contributors, notes=notes,
    )


# ---------------------------------------------------------------------
# LUBRICATION — oil pressure, oil temperature
# ---------------------------------------------------------------------
def score_lubrication(
    state: Optional[EngineState],
    cfg: EngineConfig,
) -> SubsystemHealth:
    """Score the lubrication subsystem.

    Channels:
      * ``oil_pressure_psi`` vs ``nominal_pressure_psi`` (and the
        configured [min, max] envelope).
      * ``oil_temperature_c`` vs the configured oil-temp envelope.
    """
    names = ["oil_pressure_psi", "oil_temperature_c"]
    if state is None:
        return SubsystemHealth(
            subsystem=Subsystem.LUBRICATION,
            score=0.0, confidence=0.0,
            notes=["no engine state provided"],
        )

    oil_p = getattr(state, "oil_pressure_psi", None)
    oil_t = getattr(state, "oil_temperature_c", None)
    nominal = float(cfg.lubrication.nominal_pressure_psi)
    p_lo = float(cfg.limits.oil_pressure_min_psi)
    p_hi = float(cfg.limits.oil_pressure_max_psi)
    t_lo = float(cfg.limits.oil_temp_min_c)
    t_hi = float(cfg.limits.oil_temp_max_c)

    s_p, c_p = _nominal_score(oil_p, nominal=nominal, lo=p_lo, hi=p_hi)
    s_t, c_t = _envelope_score(oil_t, lo=t_lo, hi=t_hi)

    scored = [(s_p, c_p), (s_t, c_t)]
    contributors_idx = [
        i for i, (s, c) in enumerate(scored)
        if c > 0.0 and s < 1.0
    ]
    score, conf, contributors = _aggregate(scored, contributors_idx, names)
    notes: list[str] = []
    if score < 0.55 and conf > 0.0:
        notes.append("lubrication subsystem out of envelope")
    return SubsystemHealth(
        subsystem=Subsystem.LUBRICATION,
        score=score, confidence=conf,
        contributors=contributors, notes=notes,
    )


# ---------------------------------------------------------------------
# PERFORMANCE — RPM, MAP, BSFC, fuel flow, brake power
# ---------------------------------------------------------------------
def score_performance(
    state: Optional[EngineState],
    cfg: EngineConfig,
) -> SubsystemHealth:
    """Score the performance subsystem.

    Channels:
      * ``rpm`` vs ``[idle_rpm, redline_rpm]``. ``rated_rpm`` is the
        nominal target.
      * ``manifold_pressure_inhg`` vs ``[idle_map_inhg, max_map_inhg]``.
      * ``bsfc_g_per_kwh`` — used as a *drift* indicator: scores
        poorly if it deviates from the expected map value at the
        current (RPM, MAP). We do not have the map evaluator at hand
        here, so we score BSFC conservatively as ``1 - 0.5 * |bsfc - 400| / 200``
        clamped to [0, 1] (400 g/kWh is the centre of the synthetic
        BSFC envelope; 200 g/kWh is a generous half-width).
      * ``fuel_flow_lph`` — drift indicator with centre ~30 L/h
        (a rough cruise value for this 180-hp class), half-width 25
        L/h. Clamped.
      * ``brake_power_kw`` vs ``rated_power_kw`` (and idle = 0).
    """
    names = ["rpm", "manifold_pressure_inhg", "bsfc_g_per_kwh",
             "fuel_flow_lph", "brake_power_kw"]
    if state is None:
        return SubsystemHealth(
            subsystem=Subsystem.PERFORMANCE,
            score=0.0, confidence=0.0,
            notes=["no engine state provided"],
        )

    rpm = getattr(state, "rpm", None)
    map_inhg = getattr(state, "manifold_pressure_inhg", None)
    bsfc = getattr(state, "bsfc_g_per_kwh", None)
    ff = getattr(state, "fuel_flow_lph", None)
    bp = getattr(state, "brake_power_kw", None)

    idle_rpm = float(cfg.geometry.idle_rpm)
    redline_rpm = float(cfg.geometry.redline_rpm)
    rated_rpm = float(cfg.geometry.rated_rpm)
    rated_pwr = float(cfg.geometry.rated_power_kw)

    # RPM: nominal is rated_rpm. Use a symmetric envelope.
    s_rpm, c_rpm = _nominal_score(
        rpm, nominal=rated_rpm, lo=idle_rpm, hi=redline_rpm,
    )

    # MAP: nominal is full throttle. Use a linear envelope.
    map_idle = float(cfg.throttle_to_map.idle_map_inhg)
    map_max = float(cfg.throttle_to_map.max_map_inhg)
    s_map, c_map = _envelope_score(map_inhg, lo=map_idle, hi=map_max)

    # BSFC and fuel flow: drift indicators. Conservative centre
    # values; the half-widths are large so we only flag a real
    # performance loss.
    if bsfc is None:
        s_bsfc, c_bsfc = 0.0, 0.0
    else:
        s_bsfc = _clip01(1.0 - 0.5 * abs(float(bsfc) - 400.0) / 200.0)
        c_bsfc = 1.0

    if ff is None:
        s_ff, c_ff = 0.0, 0.0
    else:
        s_ff = _clip01(1.0 - 0.5 * abs(float(ff) - 30.0) / 25.0)
        c_ff = 1.0

    # Brake power: nominal is rated, envelope is [0, rated_pwr * 1.2]
    s_bp, c_bp = _nominal_score(
        bp, nominal=rated_pwr, lo=0.0, hi=1.2 * rated_pwr,
    )

    scored = [(s_rpm, c_rpm), (s_map, c_map), (s_bsfc, c_bsfc),
              (s_ff, c_ff), (s_bp, c_bp)]
    contributors_idx = [
        i for i, (s, c) in enumerate(scored)
        if c > 0.0 and s < 1.0
    ]
    score, conf, contributors = _aggregate(scored, contributors_idx, names)
    notes: list[str] = []
    if score < 0.55 and conf > 0.0:
        notes.append("performance subsystem degraded")
    return SubsystemHealth(
        subsystem=Subsystem.PERFORMANCE,
        score=score, confidence=conf,
        contributors=contributors, notes=notes,
    )


# ---------------------------------------------------------------------
# MECHANICAL — vibration RMS, wear
# ---------------------------------------------------------------------
def score_mechanical(
    state: Optional[EngineState],
    cfg: EngineConfig,
) -> SubsystemHealth:
    """Score the mechanical subsystem.

    Channels:
      * ``vibration_rms_g`` vs ``limits.vibration_rms_max_g``
        (max-only envelope).
      * ``wear`` vs ``degradation.max_wear`` (linear, 1 - wear/max_wear).
    """
    names = ["vibration_rms_g", "wear"]
    if state is None:
        return SubsystemHealth(
            subsystem=Subsystem.MECHANICAL,
            score=0.0, confidence=0.0,
            notes=["no engine state provided"],
        )

    vib = getattr(state, "vibration_rms_g", None)
    wear = getattr(state, "wear", None)
    vib_max = float(cfg.limits.vibration_rms_max_g)
    wear_max = float(cfg.degradation.max_wear)

    s_vib, c_vib = _envelope_score(vib, lo=0.0, hi=vib_max)
    if wear is None or wear_max <= 0:
        s_wear, c_wear = 0.0, 0.0
    else:
        s_wear = _clip01(1.0 - float(wear) / wear_max)
        c_wear = 1.0

    scored = [(s_vib, c_vib), (s_wear, c_wear)]
    contributors_idx = [
        i for i, (s, c) in enumerate(scored)
        if c > 0.0 and s < 1.0
    ]
    score, conf, contributors = _aggregate(scored, contributors_idx, names)
    notes: list[str] = []
    if score < 0.55 and conf > 0.0:
        notes.append("mechanical subsystem degraded")
    return SubsystemHealth(
        subsystem=Subsystem.MECHANICAL,
        score=score, confidence=conf,
        contributors=contributors, notes=notes,
    )


# ---------------------------------------------------------------------
# SENSORS — per-channel anomaly scores from PHASE 8
# ---------------------------------------------------------------------
def score_sensors(
    anomaly: Optional[AnomalyAssessment],
) -> SubsystemHealth:
    """Score the sensors subsystem from the per-channel anomaly scores.

    Each channel's score in the anomaly assessment is a [0, 1]
    "how anomalous is this reading" value. A healthy reading scores
    0.0; a saturated anomaly scores 1.0. The sensors subsystem
    **inverts** that into a health score:

        health = 1 - anomaly_score

    The subsystem's overall score is the mean of the inverted
    per-channel scores (no confidence weighting — anomaly scores
    are already a [0, 1] signal). Contributors are the channels
    whose anomaly label is ``WARN`` or ``ANOMALY``.
    """
    if anomaly is None or not anomaly.channels:
        return SubsystemHealth(
            subsystem=Subsystem.SENSORS,
            score=0.0, confidence=0.0,
            notes=["no anomaly assessment provided"],
        )

    scored: list[tuple[float, float]] = []
    contributors: list[str] = []
    for ch_name, ca in anomaly.channels.items():
        # ``ca.score`` is the [0, 1] anomaly score; invert it.
        health = _clip01(1.0 - float(ca.score))
        scored.append((health, float(ca.confidence)))
        if ca.label in (AnomalyLabel.ANOMALY, AnomalyLabel.WARN):
            contributors.append(ch_name)

    total = sum(s for s, _ in scored)
    score = _clip01(total / float(len(scored)))
    # Confidence: mean of the per-channel confidences.
    conf = _clip01(sum(c for _, c in scored) / float(len(scored)))

    notes: list[str] = []
    if score < 0.55 and conf > 0.0:
        notes.append("sensors subsystem: anomalous channels present")
    return SubsystemHealth(
        subsystem=Subsystem.SENSORS,
        score=score, confidence=conf,
        contributors=contributors, notes=notes,
    )


# ---------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------
__all__ = [
    "score_lubrication",
    "score_mechanical",
    "score_performance",
    "score_sensors",
    "score_thermal",
]
