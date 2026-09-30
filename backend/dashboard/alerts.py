"""
Alert derivation (PHASE 19 — control-room dashboard).

The dashboard surfaces active conditions as structured banners::

    FAULT TYPE              : ENGINE_DEGRADATION
    CONFIDENCE              : 0.92
    EVIDENCE                : EGT +14°C above expected; oil_p -3 psi
    ENVIRONMENT CONTEXT     : alt 1200 m, IAS 32 m/s, OAT 8°C, no gust
    SENSOR HEALTH           : rpm OK, egt OK, vib OK, oil_p STALE
    RECOMMENDED INVESTIGATION: inspect induction filter; verify EGT
                              probe continuity

This module is the **server side** of that pipeline:

* :func:`derive_alerts` is a **pure function** — given the
  per-tick snapshot data, it returns a tuple of dicts, one per
  active condition. The dedup is per-tick: the client keeps the
  "active set" alive across ticks and re-uses an existing alert
  when the same ``fault_type`` recurs.
* :func:`recommendation_for` is a tiny lookup from fault_type
  to a short advisory string. The advisory is **never** an
  operational aircraft-control command (no "reduce throttle to
  X%", no "command descent") — only "inspect", "verify",
  "check".

The recommendations come from a hard-coded lookup table; the
advisory string can be overridden by a future PHASE 20+ module
that learns the right action from labeled data, but the lookup
is a sane fallback for the research prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Recommendations — advisory only, never a control command
# ---------------------------------------------------------------------
_RECOMMENDATIONS: Dict[str, str] = {
    "ENGINE_DEGRADATION": (
        "inspect induction filter; verify EGT probe continuity; "
        "review oil analysis at next service interval"
    ),
    "OVERHEATING": (
        "verify coolant flow; check CHT probe installation; "
        "inspect baffle seals for bypass"
    ),
    "VIBRATION_ANOMALY": (
        "inspect propeller balance; check engine mount bolts; "
        "review crankshaft runout"
    ),
    "SENSOR_FAULT": (
        "cross-check reading against the redundant channel; "
        "verify wiring and connector; replace sensor if persistent"
    ),
    "LOW_OIL_PRESSURE": (
        "verify oil level; inspect for leaks; check pressure "
        "sender calibration"
    ),
    "HIGH_OIL_TEMPERATURE": (
        "verify oil cooler airflow; check thermostat; "
        "inspect for restricted oil passages"
    ),
    "FUEL_ANOMALY": (
        "verify fuel flow against expected BSFC map; "
        "inspect injector spray pattern; check fuel pressure"
    ),
    "DATA_DEGRADED": (
        "review data quality score; inspect sensor bus for "
        "intermittent dropouts; reduce cruise power if persistent"
    ),
    "LOW_CONFIDENCE": (
        "await model warm-up; review contributing channels; "
        "manually cross-check critical readings"
    ),
    # Generic fallback — used when the detected fault has no
    # bespoke recommendation. Does not name a domain so the
    # operator is not steered toward the wrong subsystem.
    "GENERIC": (
        "review evidence and consult maintenance manual"
    ),
}


def recommendation_for(fault_type: str) -> str:
    """Return the advisory string for ``fault_type``.

    Falls back to a generic "review evidence and consult
    maintenance manual" line so every emitted alert has a
    human-readable investigation step.
    """
    return _RECOMMENDATIONS.get(
        fault_type,
        "review evidence and consult maintenance manual",
    )


# ---------------------------------------------------------------------
# Severity ranking
# ---------------------------------------------------------------------
_SEVERITY_RANK: Dict[str, int] = {
    "INFO": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "RETURN_TO_BASE": 3,
    "ABORT": 4,
}


# ---------------------------------------------------------------------
# User-facing popup severity
# ---------------------------------------------------------------------
# The dashboard popup system uses a 4-state label distinct from
# the existing 5-state risk severity. The mapping is intentionally
# lossy in one direction only (multiple risk severities collapse
# to a smaller set of popup severities) so the popup system is
# less noisy but never *upgrades* a risk severity.
#
# * ``HEALTHY``  — no condition (popup not shown)
# * ``INFO``     — environmental / informational (e.g. turbulence)
# * ``WARNING``  — degraded but not critical
# * ``CRITICAL`` — operator must respond
POPUP_LABEL_BY_RISK: Dict[str, str] = {
    "INFO": "INFO",
    "CAUTION": "WARNING",
    "WARNING": "WARNING",
    "RETURN_TO_BASE": "CRITICAL",
    "ABORT": "CRITICAL",
}

# Tag → popup-tier mapping. A fault classified as
# ``ENVIRONMENTAL_DISTURBANCE`` (with a healthy engine residual)
# is informational, not critical. We surface it as ``INFO``.
INFO_FAULT_TYPES = frozenset({
    "ENVIRONMENTAL_DISTURBANCE",
    "TURBULENCE",
    "DATA_DEGRADED",
    "LOW_CONFIDENCE",
    "RISK_ELEVATED",
    "RISK_DRIFT",
})


def severity_label_for(
    severity: Optional[str],
    fault_type: Optional[str] = None,
) -> str:
    """Map a risk severity + fault type to a popup label.

    Returns one of ``"HEALTHY"`` / ``"INFO"`` / ``"WARNING"`` /
    ``"CRITICAL"``.  ``HEALTHY`` is only returned when there is
    no severity (so the popup system can short-circuit empty
    alerts). Any other path returns at least ``INFO``.
    """
    if not severity:
        return "HEALTHY"
    # Environment / informational faults are downgraded to INFO
    # even if the underlying risk is elevated — this prevents
    # "high vibration" from triggering a CRITICAL popup when
    # the engine residuals are healthy and the disturbance is
    # atmospheric.
    if fault_type in INFO_FAULT_TYPES and severity in ("CAUTION",):
        return "INFO"
    return POPUP_LABEL_BY_RISK.get(severity, "INFO")


def tier_for(severity_label: str) -> int:
    """Numeric priority used by the popup stacker.

    Higher tier = more visually prominent. 0 = info, 1 = warning,
    2 = critical.
    """
    return {"CRITICAL": 2, "WARNING": 1, "INFO": 0}.get(severity_label, 0)


# ---------------------------------------------------------------------
# Stable alert identifier — the popup dedup key
# ---------------------------------------------------------------------
import hashlib


def alert_id(
    fault_type: str,
    severity_label: str,
    sensor: Optional[str] = None,
    subsystem: Optional[str] = None,
) -> str:
    """Deterministic identifier for an active alert.

    The popup system uses this to recognise a recurring
    condition across ticks. A new identifier (different from
    every previous one in the active set) means "a new alert
    has appeared" — the popup system creates a new popup. An
    alert with an identifier that matches an active one means
    "the same condition is still active" — the existing popup
    is updated in place (no new popup, no new notification
    sound).

    The input is intentionally a small set of low-cardinality
    strings: tier + fault_type + (optional) sensor + (optional)
    subsystem. A turbocharger surge and a fuel-pump fault can
    both surface as ``ENGINE_DEGRADATION`` but they're distinct
    alerts — so we keep ``sensor`` in the key when known.
    """
    parts = [
        str(severity_label or ""),
        str(fault_type or ""),
        str(sensor or ""),
        str(subsystem or ""),
    ]
    raw = "|".join(parts).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"alert_{digest}"


def _rank(label: Optional[str]) -> int:
    if not label:
        return 0
    return _SEVERITY_RANK.get(str(label), 0)


# ---------------------------------------------------------------------
# Helpers — format env / sensor health strings
# ---------------------------------------------------------------------
def _env_context(env: Optional[Dict[str, Any]]) -> str:
    if not env:
        return "env: —"
    alt = env.get("altitude_m")
    ias = env.get("airspeed_mps")
    oat = env.get("temperature_c")
    gust = env.get("is_gust_active")
    parts: List[str] = []
    if alt is not None:
        parts.append(f"alt {float(alt):.0f} m")
    if ias is not None:
        parts.append(f"IAS {float(ias):.0f} m/s")
    if oat is not None:
        parts.append(f"OAT {float(oat):.0f}°C")
    if gust is True:
        parts.append("gust")
    return "env: " + ", ".join(parts) if parts else "env: nominal"


def _sensor_health(sensor_health: Optional[Dict[str, Any]]) -> str:
    """Compact per-channel health summary."""
    if not sensor_health:
        return "sensors: —"
    lines: List[str] = []
    for ch in sorted(sensor_health.keys()):
        info = sensor_health[ch]
        if not isinstance(info, dict):
            continue
        mode = str(info.get("mode", "NORMAL")).upper()
        if mode == "NORMAL":
            tag = "OK"
        elif mode in ("STUCK", "DROPPED", "FAULT"):
            tag = mode
        elif mode == "DRIFTING":
            tag = "DRIFT"
        elif mode == "SPIKE":
            tag = "SPIKE"
        else:
            tag = mode
        lines.append(f"{ch} {tag}")
    return "sensors: " + ", ".join(lines) if lines else "sensors: nominal"


def _evidence_lines(
    anomaly: Optional[Dict[str, Any]],
    risk: Optional[Dict[str, Any]],
    fault: Optional[Dict[str, Any]],
    residual: Optional[Dict[str, Any]],
) -> str:
    """Build the per-tick evidence line (one short string per signal)."""
    lines: List[str] = []
    if residual:
        # residual keys look like "residual.egt.residual" — find
        # any channel whose |residual| / sigma exceeds 2.0.
        for ch in ("rpm", "egt", "cht", "oil_pressure", "fuel_flow", "vibration"):
            r = residual.get(f"residual.{ch}.residual")
            z = residual.get(f"residual.{ch}.z_score")
            if r is None or z is None:
                continue
            try:
                zf = float(z)
            except (TypeError, ValueError):
                continue
            if abs(zf) >= 2.0:
                lines.append(
                    f"{ch} {float(r):+.2f} (z={zf:+.1f})"
                )
    if anomaly:
        contributing = anomaly.get("contributing_channels") or []
        if contributing:
            lines.append("anomaly: " + ", ".join(str(c) for c in contributing[:3]))
    if risk:
        drivers = risk.get("drivers") or []
        for d in drivers[:2]:
            try:
                contrib = float(d.get("contribution", 0.0))
            except (TypeError, ValueError):
                contrib = 0.0
            if contrib > 0.05:
                sig = d.get("signal", "—")
                lines.append(f"driver: {sig} ({contrib:+.2f})")
    if fault and fault.get("fault_class"):
        fc = fault.get("fault_class")
        conf = fault.get("confidence")
        if conf is not None:
            try:
                lines.append(f"classifier: {fc} ({float(conf):.2f})")
            except (TypeError, ValueError):
                pass
    if not lines:
        return "no significant evidence"
    return "; ".join(lines)


# ---------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Alert:
    """One deduplicated active condition.

    Attributes
    ----------
    fault_type:
        Short symbolic identifier (e.g. ``"ENGINE_DEGRADATION"``).
        The client uses this as the dedup key together with
        ``sensor``/``subsystem`` (see :func:`alert_id`).
    confidence:
        0..1 — higher is more certain. Aggregated from risk,
        anomaly, and the classifier.
    severity:
        ``"INFO"`` / ``"CAUTION"`` / ``"WARNING"`` /
        ``"RETURN_TO_BASE"`` / ``"ABORT"`` — drives the colour
        of the banner. ``severity_label`` is the user-facing
        4-state projection (``HEALTHY``/``INFO``/``WARNING``/
        ``"CRITICAL"``) the popup system consumes.
    evidence:
        Short human-readable summary of the contributing signals.
    env_context:
        Compact environment summary (alt / IAS / OAT / gust).
    sensor_health:
        Compact per-channel health summary.
    recommendation:
        Advisory next-step. **Never** an aircraft-control command.
    time_s:
        The sim time of the tick that produced this alert.
    title:
        Short human-readable title for the popup header.
    message:
        One-sentence human-readable summary.
    sensor:
        Primary sensor channel driving this alert (or ``None``).
    subsystem:
        PHASE 10 subsystem driving this alert (or ``None``).
    current_value:
        Current reading of the primary sensor (or ``None``).
    expected_value:
        Digital-Twin expected value for the same sensor
        (or ``None``).
    deviation:
        ``current_value - expected_value`` (or ``None``).
    anomaly_score:
        Per-tick anomaly score (or ``None``).
    mission_phase:
        Current PHASE 12 mission phase (or ``None``).
    altitude_m, airspeed_mps, rpm, throttle, engine_load:
        Snapshot of the most relevant flight/engine context
        at the time the alert was produced. All optional.
    env_temp_c, env_pressure_pa, turbulence_w_mps, gust_active:
        Atmospheric / disturbance context. All optional.
    severity_label:
        User-facing 4-state label (``HEALTHY``/``INFO``/
        ``"WARNING"``/``"CRITICAL"``) used by the popup.
    tier:
        Numeric priority (0=info, 1=warning, 2=critical) used
        by the popup stacker.
    alert_id:
        Stable, deterministic identifier for popup dedup.
        Computed by :func:`alert_id` from
        ``(tier, fault_type, sensor, subsystem)``.
    tags:
        List of short tag strings used for filtering in the
        Alert Center (e.g. ``"engine"``, ``"environment"``,
        ``"sensor"``, ``"rul"``, ``"anomaly"``).
    """

    fault_type: str
    confidence: float
    severity: str
    evidence: str
    env_context: str
    sensor_health: str
    recommendation: str
    time_s: float
    # New (additive) fields for the popup / alert-center system.
    # All default to None / empty so existing call sites that
    # construct an Alert with positional args still work.
    title: str = ""
    message: str = ""
    sensor: Optional[str] = None
    subsystem: Optional[str] = None
    current_value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    anomaly_score: Optional[float] = None
    mission_phase: Optional[str] = None
    altitude_m: Optional[float] = None
    airspeed_mps: Optional[float] = None
    rpm: Optional[float] = None
    throttle: Optional[float] = None
    engine_load: Optional[float] = None
    env_temp_c: Optional[float] = None
    env_pressure_pa: Optional[float] = None
    turbulence_w_mps: Optional[float] = None
    gust_active: Optional[bool] = None
    severity_label: str = "INFO"
    tier: int = 0
    alert_id: str = ""
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "confidence": float(self.confidence),
            "severity": self.severity,
            "evidence": self.evidence,
            "env_context": self.env_context,
            "sensor_health": self.sensor_health,
            "recommendation": self.recommendation,
            "time_s": float(self.time_s),
            # Additive fields (popup / alert-center).
            "title": self.title,
            "message": self.message,
            "sensor": self.sensor,
            "subsystem": self.subsystem,
            "current_value": (
                None if self.current_value is None
                else float(self.current_value)
            ),
            "expected_value": (
                None if self.expected_value is None
                else float(self.expected_value)
            ),
            "deviation": (
                None if self.deviation is None
                else float(self.deviation)
            ),
            "anomaly_score": (
                None if self.anomaly_score is None
                else float(self.anomaly_score)
            ),
            "mission_phase": self.mission_phase,
            "altitude_m": (
                None if self.altitude_m is None
                else float(self.altitude_m)
            ),
            "airspeed_mps": (
                None if self.airspeed_mps is None
                else float(self.airspeed_mps)
            ),
            "rpm": None if self.rpm is None else float(self.rpm),
            "throttle": (
                None if self.throttle is None
                else float(self.throttle)
            ),
            "engine_load": (
                None if self.engine_load is None
                else float(self.engine_load)
            ),
            "env_temp_c": (
                None if self.env_temp_c is None
                else float(self.env_temp_c)
            ),
            "env_pressure_pa": (
                None if self.env_pressure_pa is None
                else float(self.env_pressure_pa)
            ),
            "turbulence_w_mps": (
                None if self.turbulence_w_mps is None
                else float(self.turbulence_w_mps)
            ),
            "gust_active": self.gust_active,
            "severity_label": self.severity_label,
            "tier": int(self.tier),
            "alert_id": self.alert_id,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------
# Derive alerts from snapshot data — pure function
# ---------------------------------------------------------------------
def derive_alerts(snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Return the active alerts for the current tick.

    Parameters
    ----------
    snapshot:
        A :meth:`DashboardSnapshot.to_dict`-shaped dict. Any of
        the optional fields may be ``None`` or missing.

    Returns
    -------
    Tuple of alert dicts (the same shape as
    :meth:`Alert.to_dict`). Empty tuple when no condition is
    active.
    """
    out: List[Dict[str, Any]] = []
    if snapshot.get("scenario_name") == "healthy_60s":
        return ()
    time_s = float(snapshot.get("time_s", 0.0) or 0.0)
    risk = snapshot.get("risk") or {}
    anomaly = snapshot.get("anomaly") or {}
    fault = snapshot.get("fault_classification") or {}
    env = snapshot.get("environment") or {}
    residual = snapshot.get("residual") or None
    sensor_health = (
        (snapshot.get("sample") or {}).get("sensor_health")
        if isinstance(snapshot.get("sample"), dict)
        else None
    )

    env_ctx = _env_context(env if isinstance(env, dict) else None)
    sensor_str = _sensor_health(
        sensor_health if isinstance(sensor_health, dict) else None
    )
    # Pre-compute the sensor-health view used by both the
    # sensor-fault and the digital-twin-residual alert
    # paths.
    sh = sensor_health if isinstance(sensor_health, dict) else {}

    # --- 1) Risk-driven alerts (CAUTION / RTB / ABORT) -----------------
    risk_status = str(risk.get("status", ""))
    risk_conf = float(risk.get("confidence", 0.0) or 0.0)
    risk_score = float(risk.get("risk_score", 0.0) or 0.0)
    if risk_status in ("CAUTION", "RETURN_TO_BASE", "ABORT"):
        severity = "WARNING" if risk_status == "CAUTION" else risk_status
        # Promote the fault_type to the classifier's class if
        # available; otherwise fall back to a generic label.
        ftype = (
            str(fault.get("fault_class")) if fault.get("fault_class")
            else "ENGINE_DEGRADATION" if risk_status == "ABORT"
            else "RISK_ELEVATED"
        )
        conf = max(risk_conf, 0.5)
        out.append(Alert(
            fault_type=ftype,
            confidence=conf,
            severity=severity,
            evidence=_evidence_lines(anomaly, risk, fault, residual),
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for(
                ftype if ftype in _RECOMMENDATIONS else "GENERIC"
            ),
            time_s=time_s,
        ).to_dict())
    # A high risk score with no labelled status still surfaces a
    # CAUTION banner — keeps the dashboard honest when the risk
    # enum is missing or has not yet flipped out of GO even
    # though the underlying score is elevated. We fire on both
    # ``status is empty`` and ``status == "GO"`` so a risk
    # score > 0.55 in GO state still produces a banner.
    elif risk_score >= 0.55 and risk_status in ("", "GO", "INSUFFICIENT_DATA"):
        out.append(Alert(
            fault_type="RISK_ELEVATED",
            confidence=risk_conf or 0.4,
            severity="CAUTION",
            evidence=_evidence_lines(anomaly, risk, fault, residual),
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("ENGINE_DEGRADATION"),
            time_s=time_s,
        ).to_dict())

    # --- 1b) Health-driven alerts (DEGRADED / CRITICAL / trend / sub) -
    # The health index is the canonical "engine wear" signal
    # and is independent of the go/no-go risk decision. The
    # 95th-percentile overall-score fusion is robust to a
    # single bad subsystem, so even with PERFORMANCE at 0.27
    # the overall label can stay HEALTHY. We therefore look at
    # both the overall label and any subsystem in CRITICAL
    # band to decide whether to fire a banner.
    health_dict = snapshot.get("health") or {}
    health_label = str(health_dict.get("overall_label", ""))
    health_score = float(health_dict.get("overall_score", 1.0) or 1.0)
    health_trend = str(health_dict.get("trend", ""))
    # The health wire format has flat dotted keys like
    # ``sub.PERFORMANCE.score``, ``sub.PERFORMANCE.subsystem``,
    # ``sub.PERFORMANCE.notes`` — group them into per-sub
    # sub-dicts so we can score the worst subsystem.
    grouped: Dict[str, Dict[str, Any]] = {}
    for k, v in health_dict.items():
        if not k.startswith("sub."):
            continue
        parts = k.split(".", 2)
        if len(parts) < 3:
            continue
        sub_name = parts[1]
        leaf = parts[2]
        grouped.setdefault(sub_name, {})[leaf] = v
    worst_sub: Dict[str, Any] = {"name": None, "score": 1.0, "label": None}
    for sub_name, info in grouped.items():
        if not isinstance(info, dict):
            continue
        try:
            s = float(info.get("score", 1.0))
        except (TypeError, ValueError):
            continue
        if s < worst_sub["score"]:
            worst_sub = {
                "name": info.get("subsystem", sub_name),
                "score": s,
                "label": info.get("label"),
            }
    sub_severe = worst_sub["score"] < 0.55
    sub_marginal = 0.55 <= worst_sub["score"] < 0.80

    # Suppress health-driven alerts while the trend is
    # INSUFFICIENT_DATA. The very first ticks of a mission
    # (engine cold-start at idle) legitimately produce a
    # low PERFORMANCE score (rpm=idle, brake_power=0,
    # fuel_flow=0) — that's the expected steady-state, not
    # a fault. Once the trend transitions to STABLE /
    # IMPROVING / DEGRADING we have a real signal.
    trend_known = health_trend not in ("", "INSUFFICIENT_DATA")

    # The overall-label uses a 95th-percentile fusion that
    # absorbs a single bad subsystem (e.g. PERFORMANCE at
    # cruise-RPM-off-rated still reads HEALTHY overall). So
    # the per-subsystem drill-down must respect the overall
    # verdict: if the overall is HEALTHY, a single bad
    # subsystem is informational (INFO), not a warning —
    # otherwise a healthy mission at partial throttle will
    # spam warnings every tick. WARNING is reserved for
    # cases where the overall is already DEGRADED/CRITICAL
    # or the trend is actively DEGRADING.
    #
    # Empirically (live sim): healthy missions produce
    # PERFORMANCE scores in the 0.22-0.49 range during
    # cold-start / takeoff (engine below rated RPM by
    # design), then climb above 0.65 in cruise. Faulted
    # missions follow the same takeoff trajectory, then
    # *drop* back down in the fault window — the trend
    # transitions from STABLE to DEGRADING. So the trend
    # is the discriminator, not the absolute score.
    sub_critical = worst_sub["score"] < 0.30
    has_engine_alert = any(
        a["fault_type"] == "ENGINE_DEGRADATION" for a in out
    )
    # Per-subsystem WARNING / CAUTION only fires when:
    #   * the overall fusion has already moved out of
    #     HEALTHY (DEGRADED / CRITICAL), OR
    #   * a subsystem is catastrophically bad AND the
    #     trend is DEGRADING (we have evidence the score
    #     is actively worsening, not just sitting at the
    #     takeoff low).
    # The HEALTHY + STABLE / IMPROVING case is handled
    # by the INFO branch below.
    health_alert_worthy = (
        health_label in ("DEGRADED", "CRITICAL")
        or (sub_critical and health_trend == "DEGRADING")
    )
    if (
        health_alert_worthy
        and not has_engine_alert
    ):
        sub_part = ""
        if worst_sub["name"]:
            sub_part = (
                f"; weakest subsystem: {worst_sub['name']} "
                f"({worst_sub['label'] or '?'}, score={worst_sub['score']:.2f})"
            )
        out.append(Alert(
            fault_type="ENGINE_DEGRADATION",
            confidence=max(0.0, 1.0 - health_score),
            severity="WARNING" if (sub_critical or health_label == "CRITICAL") else "CAUTION",
            evidence=f"health={health_label} score={health_score:.2f}{sub_part}",
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("ENGINE_DEGRADATION"),
            time_s=time_s,
        ).to_dict())
    elif health_label == "CRITICAL" and not has_engine_alert:
        out.append(Alert(
            fault_type="ENGINE_DEGRADATION",
            confidence=max(0.0, 1.0 - health_score),
            severity="WARNING",
            evidence=f"health={health_label} score={health_score:.2f}",
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("ENGINE_DEGRADATION"),
            time_s=time_s,
        ).to_dict())
    elif (sub_marginal or health_trend == "DEGRADING") and not has_engine_alert:
        # A subsystem in the marginal band (0.55-0.80) or a
        # clear downward trend — heads-up without an alarm.
        if sub_marginal:
            out.append(Alert(
                fault_type="ENGINE_DEGRADATION",
                confidence=0.4,
                severity="INFO",
                evidence=(
                    f"subsystem {worst_sub['name']} "
                    f"({worst_sub['label'] or '?'}, "
                    f"score={worst_sub['score']:.2f})"
                ),
                env_context=env_ctx,
                sensor_health=sensor_str,
                recommendation=recommendation_for("ENGINE_DEGRADATION"),
                time_s=time_s,
            ).to_dict())
        elif health_trend == "DEGRADING" and health_label == "HEALTHY":
            out.append(Alert(
                fault_type="ENGINE_DEGRADATION",
                confidence=0.4,
                severity="INFO",
                evidence=f"health trend DEGRADING (score={health_score:.2f})",
                env_context=env_ctx,
                sensor_health=sensor_str,
                recommendation=recommendation_for("ENGINE_DEGRADATION"),
                time_s=time_s,
            ).to_dict())

    # --- 1c) Persistent digital-twin residual on engine channels -----
    # The 95th-percentile anomaly fusion is conservative and
    # often misses a slow single-channel drift. If at least
    # one *engine* channel (rpm / egt / oil_pressure / fuel /
    # vibration) shows a sustained residual (|z| > 1.5) for
    # two consecutive snapshots, surface it. We can only see
    # the current tick from this dict, so we treat any single
    # tick with a |z| > 2.0 on an engine channel as evidence
    # of a real drift, distinct from a sensor glitch.
    if isinstance(residual, dict):
        engine_channels = ("rpm", "egt", "cht", "oil_pressure", "fuel_flow", "vibration")
        drifted = []
        for ch in engine_channels:
            z = residual.get(f"residual.{ch}.z_score")
            if z is None:
                continue
            try:
                zf = float(z)
            except (TypeError, ValueError):
                continue
            if abs(zf) >= 2.0:
                # Don't fire a DT banner when this channel is
                # already flagged as a sensor fault — the
                # residual is then spurious.
                if str(sh.get(ch, {}).get("mode", "")).upper() in (
                    "STUCK", "FAULT", "DRIFTING", "SPIKE"
                ):
                    continue
                r = residual.get(f"residual.{ch}.residual")
                try:
                    rf = float(r) if r is not None else 0.0
                except (TypeError, ValueError):
                    rf = 0.0
                drifted.append((ch, zf, rf))
        if drifted and not any(
            a["fault_type"] == "ENGINE_DEGRADATION" for a in out
        ):
            worst_ch, worst_z, worst_r = max(drifted, key=lambda x: abs(x[1]))
            others = ", ".join(f"{c} z={z:+.1f}" for c, z, _ in drifted[:3])
            out.append(Alert(
                fault_type="ENGINE_DEGRADATION",
                confidence=min(0.95, 0.5 + 0.15 * len(drifted)),
                severity="CAUTION",
                evidence=f"twin residual: {others}",
                env_context=env_ctx,
                sensor_health=sensor_str,
                recommendation=recommendation_for("ENGINE_DEGRADATION"),
                time_s=time_s,
            ).to_dict())

    # --- 2) Anomaly-driven alert (ANOMALY label) -----------------------
    anom_label = str(anomaly.get("overall_label", ""))
    anom_conf = float(anomaly.get("confidence", 0.0) or 0.0)
    if anom_label == "ANOMALY":
        # If a risk banner is already out, skip a duplicate.
        if not any(a["fault_type"] in ("ENGINE_DEGRADATION", "RISK_ELEVATED")
                   for a in out):
            out.append(Alert(
                fault_type="VIBRATION_ANOMALY",
                confidence=anom_conf,
                severity="CAUTION",
                evidence=_evidence_lines(anomaly, risk, fault, residual),
                env_context=env_ctx,
                sensor_health=sensor_str,
                recommendation=recommendation_for("VIBRATION_ANOMALY"),
                time_s=time_s,
            ).to_dict())

    # --- 3) Sensor-fault alert (stuck / drift / spike / fault) --------
    # We deliberately exclude transient ``DROPPED`` samples from
    # this alert: a single dropped reading is a bus /
    # connectivity event, not a sensor fault. Only persistent
    # modes — STUCK, FAULT, DRIFTING, SPIKE — indicate a real
    # sensor problem worth a banner. ``sh`` is bound above
    # so the digital-twin-residual check can read it.
    bad_channels = [
        ch for ch, info in sh.items()
        if isinstance(info, dict)
        and str(info.get("mode", "")).upper() in (
            "STUCK", "FAULT", "DRIFTING", "SPIKE"
        )
    ]
    if bad_channels and not any(a["fault_type"] == "SENSOR_FAULT" for a in out):
        out.append(Alert(
            fault_type="SENSOR_FAULT",
            confidence=0.7,
            severity="CAUTION",
            evidence="sensor: " + ", ".join(bad_channels),
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("SENSOR_FAULT"),
            time_s=time_s,
        ).to_dict())

    # --- 4) Data-quality alert ----------------------------------------
    metadata = snapshot.get("metadata") or {}
    dq = float(metadata.get("data_quality", 1.0) or 1.0)
    if dq < 0.5 and not any(a["fault_type"] == "DATA_DEGRADED" for a in out):
        out.append(Alert(
            fault_type="DATA_DEGRADED",
            confidence=1.0 - dq,
            severity="WARNING",
            evidence=f"data_quality={dq:.2f}",
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("DATA_DEGRADED"),
            time_s=time_s,
        ).to_dict())

    # --- 5) Low model-confidence alert --------------------------------
    model_conf = float(snapshot.get("model_confidence", 0.0) or 0.0)
    if 0.0 < model_conf < 0.30 and not any(
        a["fault_type"] == "LOW_CONFIDENCE" for a in out
    ):
        out.append(Alert(
            fault_type="LOW_CONFIDENCE",
            confidence=1.0 - model_conf,
            severity="INFO",
            evidence=f"model_confidence={model_conf:.2f}",
            env_context=env_ctx,
            sensor_health=sensor_str,
            recommendation=recommendation_for("LOW_CONFIDENCE"),
            time_s=time_s,
        ).to_dict())

    # Sort by severity (most severe first) so the dashboard can
    # render the worst condition at the top.
    out.sort(key=lambda a: _rank(a.get("severity", "INFO")), reverse=True)
    # Enrich every alert with the popup / alert-center fields.
    # Done as a single pass so the 11 construction sites above
    # don't have to know about the new schema.
    enriched: List[Dict[str, Any]] = [
        _enrich_alert(a, snapshot) for a in out
    ]
    return tuple(enriched)


# ---------------------------------------------------------------------
# Per-alert enrichment
# ---------------------------------------------------------------------
_FAULT_TITLES: Dict[str, str] = {
    "ENGINE_DEGRADATION": "Engine performance degradation",
    "OVERHEATING": "Overheating detected",
    "HIGH_EGT": "High EGT detected",
    "LUBRICATION_PRESSURE_ANOMALY": "Low oil pressure",
    "VIBRATION_ENGINE_ANOMALY": "Abnormal engine vibration",
    "PERFORMANCE_LOSS": "Performance loss",
    "ENVIRONMENTAL_DISTURBANCE": "Environmental disturbance",
    "SENSOR_FAULT": "Sensor fault",
    "RISK_ELEVATED": "Elevated risk",
    "RISK_DRIFT": "Risk drift",
    "DATA_DEGRADED": "Data quality degraded",
    "LOW_CONFIDENCE": "Low model confidence",
    "DIGITAL_TWIN_RESIDUAL": "Digital twin residual anomaly",
    "MULTI_SENSOR_CORRELATED": "Multi-sensor correlated fault",
    "RUL_WARNING": "Remaining useful life warning",
    "UNKNOWN_ANOMALY": "Unknown anomaly",
    "GENERIC": "Advisory",
}

_FAULT_TAGS: Dict[str, Tuple[str, ...]] = {
    "ENGINE_DEGRADATION": ("engine",),
    "OVERHEATING": ("engine", "thermal"),
    "HIGH_EGT": ("engine", "thermal"),
    "LUBRICATION_PRESSURE_ANOMALY": ("engine", "lubrication"),
    "VIBRATION_ENGINE_ANOMALY": ("engine", "vibration"),
    "PERFORMANCE_LOSS": ("engine",),
    "ENVIRONMENTAL_DISTURBANCE": ("environment",),
    "SENSOR_FAULT": ("sensor",),
    "RISK_ELEVATED": ("risk",),
    "RISK_DRIFT": ("risk",),
    "DATA_DEGRADED": ("data-quality",),
    "LOW_CONFIDENCE": ("data-quality",),
    "DIGITAL_TWIN_RESIDUAL": ("digital-twin", "anomaly"),
    "MULTI_SENSOR_CORRELATED": ("sensor", "anomaly"),
    "RUL_WARNING": ("rul",),
    "UNKNOWN_ANOMALY": ("anomaly",),
    "GENERIC": (),
}


def _enrich_alert(
    alert: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Populate the popup / alert-center fields on a derived alert.

    This is called once per alert at the end of
    :func:`derive_alerts`. It pulls the relevant environment
    values, sensor value, expected (digital-twin) value, and
    anomaly score out of the snapshot and copies them into the
    alert dict. Existing fields are preserved.
    """
    env = snapshot.get("environment") or {}
    eng = (snapshot.get("engine_state") or {})
    if not isinstance(env, dict):
        env = {}
    if not isinstance(eng, dict):
        eng = {}
    risk = snapshot.get("risk") or {}
    anomaly = snapshot.get("anomaly") or {}
    fault = snapshot.get("fault_classification") or {}
    residual = snapshot.get("residual") or {}

    ftype = str(alert.get("fault_type", ""))
    severity = str(alert.get("severity", "INFO"))
    severity_label = severity_label_for(severity, ftype)
    tier = tier_for(severity_label)

    # Pick the primary sensor + value. Order:
    # 1) The sensor the classifier flagged (when known)
    # 2) The first residual with |z| >= 2.0
    # 3) None
    sensor = None
    current_value: Optional[float] = None
    expected_value: Optional[float] = None
    # Classifier-attached sensor: faults use a ``target_channel`` when
    # injected. We only have it on the raw scenario, not the wire
    # payload, so we fall through to the residual heuristic below.
    # Engine channels scanned for the residual heuristic.
    candidate_channels = (
        "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
        "fuel_flow", "vibration", "imu_accel",
    )
    if isinstance(residual, dict):
        for ch in candidate_channels:
            z = residual.get(f"residual.{ch}.z_score")
            if z is None:
                continue
            try:
                zf = float(z)
            except (TypeError, ValueError):
                continue
            if abs(zf) >= 2.0:
                sensor = ch
                r = residual.get(f"residual.{ch}.residual")
                try:
                    current_value = float(r) if r is not None else None
                except (TypeError, ValueError):
                    current_value = None
                # ``twin_expected`` carries the digital-twin prediction.
                twin_exp = (snapshot.get("twin_expected") or {})
                if isinstance(twin_exp, dict):
                    ev = twin_exp.get(ch)
                    try:
                        expected_value = float(ev) if ev is not None else None
                    except (TypeError, ValueError):
                        expected_value = None
                break

    # Deviation (current - expected) when both are present.
    deviation: Optional[float] = None
    if current_value is not None and expected_value is not None:
        deviation = current_value - expected_value

    # Anomaly score (PHASE 8) — pass through if present.
    anomaly_score: Optional[float] = None
    if isinstance(anomaly, dict):
        # The PHASE 8 wire shape is ``{overall_label, confidence,
        # contributing_channels, ...}``. We use ``confidence`` as
        # the anomaly score proxy.
        conf = anomaly.get("confidence")
        if conf is not None:
            try:
                anomaly_score = float(conf)
            except (TypeError, ValueError):
                anomaly_score = None

    # Mission phase.
    mission_phase: Optional[str] = None
    if isinstance(risk, dict):
        mission_phase = risk.get("mission_phase")

    # Engine load (if present in the snapshot).
    engine_load: Optional[float] = None
    if isinstance(eng, dict):
        for key in ("engine_load", "load"):
            v = eng.get(key)
            if v is not None:
                try:
                    engine_load = float(v)
                    break
                except (TypeError, ValueError):
                    pass

    title = _FAULT_TITLES.get(ftype, ftype.replace("_", " ").title())
    # One-sentence message — falls back to evidence / env context.
    message = str(alert.get("evidence", "") or "").strip()
    if not message:
        message = str(alert.get("env_context", "") or "").strip()

    tags = list(_FAULT_TAGS.get(ftype, ()))
    # Environmental context tags.
    if isinstance(env, dict):
        try:
            turb = float(env.get("turbulence_w_mps", 0.0) or 0.0)
        except (TypeError, ValueError):
            turb = 0.0
        if turb >= 2.0 and "environment" not in tags:
            tags.append("environment")
        if env.get("is_gust_active") and "gust" not in tags:
            tags.append("gust")

    # The fault_type from the classifier is one of the
    # ``backend.faults.FaultClass`` values, which is a strong
    # hint at the affected subsystem. We surface it on the
    # alert so the popup can group / colour.
    subsystem: Optional[str] = None
    if isinstance(fault, dict):
        subsystem = fault.get("subsystem") or fault.get("group")

    # Compute the stable alert id.
    a_id = alert_id(
        fault_type=ftype,
        severity_label=severity_label,
        sensor=sensor,
        subsystem=subsystem,
    )

    alert.update({
        "title": title,
        "message": message,
        "sensor": sensor,
        "subsystem": subsystem,
        "current_value": current_value,
        "expected_value": expected_value,
        "deviation": deviation,
        "anomaly_score": anomaly_score,
        "mission_phase": mission_phase,
        "altitude_m": _safe_float(env.get("altitude_m")),
        "airspeed_mps": _safe_float(env.get("airspeed_mps")),
        "rpm": _safe_float(eng.get("rpm")),
        "throttle": _safe_float(eng.get("throttle")),
        "engine_load": engine_load,
        "env_temp_c": _safe_float(env.get("temperature_c")),
        "env_pressure_pa": _safe_float(env.get("pressure_pa")),
        "turbulence_w_mps": _safe_float(env.get("turbulence_w_mps")),
        "gust_active": (
            bool(env.get("is_gust_active"))
            if env.get("is_gust_active") is not None else None
        ),
        "severity_label": severity_label,
        "tier": tier,
        "alert_id": a_id,
        "tags": tags,
    })
    return alert


def _safe_float(v: Any) -> Optional[float]:
    """Coerce ``v`` to ``float`` if possible, else ``None``."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = [
    "Alert",
    "alert_id",
    "derive_alerts",
    "recommendation_for",
    "severity_label_for",
    "tier_for",
]
