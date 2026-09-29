"""
Severity progression — how a fault's severity evolves over time.

A pure function so it's easy to unit-test and reuse from the injector
and the diagnostic engine. The supported shapes are:

* ``"linear"`` — ramps from 0 at ``onset`` to ``peak`` at ``onset+duration``
  and **stays at** ``peak`` until the caller stops asking.
* ``"step"`` — jumps from 0 to ``peak`` at ``onset`` and stays there.

In both cases the active window is ``[onset, onset+duration)``. A
``duration`` of ``None`` means the fault never decays — it stays at
``peak`` after the (1 s) ramp-up. The injector (PHASE 7) is
responsible for "ending" the fault at the right moment (e.g. by
calling :meth:`clear_fault` on the sensor bundle).
"""

from __future__ import annotations

from typing import Optional


_VALID_MODELS = ("linear", "step", "exponential", "pulse")


def severity_at(
    t_s: float,
    onset_s: float,
    duration_s: Optional[float],
    peak: float,
    model: str = "linear",
) -> float:
    """Return the severity at time ``t_s`` for the given envelope.

    Parameters
    ----------
    t_s
        Current mission time in seconds.
    onset_s
        When the fault starts. Before this, severity is 0.
    duration_s
        Length of the active window. ``None`` means the fault persists
        indefinitely (after a 1 s linear ramp-up if ``model='linear'``).
    peak
        Maximum severity the envelope reaches, in [0, 1].
    model
        ``"linear"`` (default), ``"step"``, ``"exponential"`` or
        ``"pulse"`` (PHASE 21). See module docstring for the
        shape of each.
    """
    if model not in _VALID_MODELS:
        raise ValueError(
            f"progression model must be one of {_VALID_MODELS}, got {model!r}"
        )
    peak = max(0.0, min(1.0, float(peak)))
    if t_s < onset_s:
        return 0.0
    if duration_s is None:
        # No decay.
        if model == "linear":
            ramp = max(0.0, min(1.0, (t_s - onset_s) / max(1e-9, 1.0)))
            return peak * ramp
        if model == "exponential":
            # 1s ramp up to peak, then stays at peak.
            return peak if (t_s - onset_s) >= 1.0 else peak * (t_s - onset_s)
        # "step" and "pulse" both jump to peak.
        return peak
    if t_s >= onset_s + duration_s:
        # Past the active window.
        return 0.0
    # Within the active window.
    if model == "step":
        return peak
    if model == "exponential":
        # 1s ramp up, then exponential decay with tau = duration/3.
        elapsed = t_s - onset_s
        if elapsed < 1.0:
            return peak * elapsed
        tau = max(1e-9, duration_s / 3.0)
        from math import exp
        return peak * exp(-(elapsed - 1.0) / tau)
    if model == "pulse":
        # 0..20% : ramp 0 -> 0.6*peak
        # 20..80%: hold at 0.6*peak
        # 80..100%: ramp 0.6*peak -> 0.4*peak
        elapsed = t_s - onset_s
        frac = elapsed / max(1e-9, duration_s)
        if frac < 0.2:
            return peak * 0.6 * (frac / 0.2)
        if frac < 0.8:
            return peak * 0.6
        return peak * (0.6 - 0.2 * ((frac - 0.8) / 0.2))
    # linear
    ramp = (t_s - onset_s) / max(1e-9, duration_s)
    return peak * max(0.0, min(1.0, ramp))


__all__ = ["severity_at"]
