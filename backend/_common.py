"""
Cross-cutting constants shared by the diagnostic / RUL / risk / health
modules.

The :data:`DEFAULT_TREND_WINDOW` is the default size of the rolling
window the per-domain trend detectors (RUL TTE, risk score, residual
|z|, health score) keep for slope calculation. Each domain may still
override this value locally if its semantics differ; this is the
*default* and the *documented* shared value, so future drift is
visible in code review (a domain that diverges must justify it).
"""

from __future__ import annotations

# Default number of recent samples kept by the trend detectors.
# See ``backend/rul/types.py``, ``backend/risk/types.py``,
# ``backend/diagnostics/diagnostic_state.py`` and
# ``backend/health/aggregator.py`` for the per-domain re-exports.
DEFAULT_TREND_WINDOW: int = 10

__all__ = ["DEFAULT_TREND_WINDOW"]
