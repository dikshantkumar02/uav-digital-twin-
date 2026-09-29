"""
Tests for the shared constants module.

Verifies that the per-domain ``TREND_WINDOW`` re-exports cannot
drift apart. Each domain (RUL, risk, diagnostics, health) keeps
its own ``TREND_WINDOW`` symbol so existing imports/tests work,
but each one must equal the central
:data:`backend._common.DEFAULT_TREND_WINDOW`. Without this guard
someone could bump one domain's window without realising the
others still point at the original value.
"""

from __future__ import annotations

import pytest

from backend._common import DEFAULT_TREND_WINDOW
from backend.diagnostics.diagnostic_state import TREND_WINDOW as D_TW
from backend.health.aggregator import TREND_WINDOW as H_TW
from backend.risk.types import TREND_WINDOW as K_TW
from backend.rul.types import TREND_WINDOW as R_TW


def test_default_trend_window_is_10() -> None:
    """The default value is the historical 10-tick rolling window.

    Any change here is a project-wide policy change and must be
    accompanied by a test review of the four trend detectors.
    """
    assert DEFAULT_TREND_WINDOW == 10


def test_all_domain_trend_windows_agree() -> None:
    """Every domain's TREND_WINDOW must equal the central default.

    A drift here means a module was edited to override the
    re-export — which is allowed but must be explicit and
    accompanied by this test passing (the divergent module
    would be the one to fail).
    """
    assert R_TW == DEFAULT_TREND_WINDOW
    assert K_TW == DEFAULT_TREND_WINDOW
    assert D_TW == DEFAULT_TREND_WINDOW
    assert H_TW == DEFAULT_TREND_WINDOW


def test_default_trend_window_is_positive_int() -> None:
    """Sanity: a non-positive window would break all four trend
    detectors (the deque would never accumulate, so the trend
    would always be INSUFFICIENT_DATA)."""
    assert isinstance(DEFAULT_TREND_WINDOW, int)
    assert DEFAULT_TREND_WINDOW > 0
    assert DEFAULT_TREND_WINDOW <= 1000


@pytest.mark.parametrize("name,value", [
    ("rul", R_TW),
    ("risk", K_TW),
    ("diagnostics", D_TW),
    ("health", H_TW),
])
def test_each_domain_window_is_usable(name: str, value: int) -> None:
    """Each per-domain symbol is the same int that the domain
    uses to size its trend buffer. Re-importing the symbol in
    the test guarantees the constant is exported publicly."""
    assert value == DEFAULT_TREND_WINDOW, (
        f"{name}.TREND_WINDOW drifted to {value} "
        f"(central default is {DEFAULT_TREND_WINDOW})"
    )
