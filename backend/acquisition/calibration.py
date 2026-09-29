"""
Per-channel calibration (PHASE 18 — embedded telemetry acquisition).

A :class:`CalibrationTable` describes the transfer function the
edge applies to every raw ADC value before it leaves the bus::

    y = apply(raw)
        = (raw * gain + offset) then optional polynomial

A :class:`CalibrationRegistry` is the per-channel bundle, keyed
by the channel name. It is intentionally simple — no ML, no
online learning, no fancy curve fitting. The MCU firmware can
store the same tables on its end so the conversion is bit-exact
both ways.

The :meth:`CalibrationTable.is_due` helper drives the
``CALIBRATION_DUE`` bit in :class:`SensorHealth` — when the
elapsed time since the last calibration exceeds the configured
``valid_for_ms``, the edge tags every sample with that flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


# ---------------------------------------------------------------------
# Per-channel table
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationTable:
    """Per-channel gain, offset, optional polynomial correction.

    Attributes
    ----------
    channel:
        Sensor channel name (matches :data:`TELEMETRY_FIELDS`).
    gain:
        Multiplicative correction applied to ``raw`` first. Default
        1.0 (no change).
    offset:
        Additive correction applied after the gain. Default 0.0.
    poly_coeffs:
        Optional polynomial correction evaluated on the linear
        result via Horner's method. Empty tuple disables it.
        E.g. ``(a, b, c)`` evaluates ``a*x**2 + b*x + c``.
    calibrated_at_ms:
        MCU wall-time the calibration was performed.
    valid_for_ms:
        Calibration validity window in milliseconds. After
        ``calibrated_at_ms + valid_for_ms`` the
        :class:`SensorHealth` ``CALIBRATION_DUE`` flag should
        be set. Default 30 days.
    """

    channel: str
    gain: float = 1.0
    offset: float = 0.0
    poly_coeffs: Tuple[float, ...] = ()
    calibrated_at_ms: int = 0
    valid_for_ms: int = 30 * 24 * 3600 * 1000   # 30 days

    def apply(self, raw: float) -> float:
        """Apply gain, offset, and optional polynomial to ``raw``."""
        y = float(raw) * float(self.gain) + float(self.offset)
        if self.poly_coeffs:
            # Horner's method. The first coefficient is the
            # highest-order term. E.g. (a, b, c) evaluates
            # a*x*x + b*x + c as ((((a)*x)+b)*x)+c.
            result = 0.0
            for c in self.poly_coeffs:
                result = result * y + float(c)
            y = result
        return y

    def is_due(self, now_ms: int) -> bool:
        """Return True if the calibration is past its due date."""
        return (int(now_ms) - int(self.calibrated_at_ms)) > int(
            self.valid_for_ms
        )


# ---------------------------------------------------------------------
# Per-bundle registry
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationRegistry:
    """Per-channel :class:`CalibrationTable` registry.

    ``tables`` is a dict keyed by channel name. Channels not
    present in the dict use a passthrough (gain=1, offset=0,
    no polynomial).
    """

    tables: Dict[str, CalibrationTable] = field(default_factory=dict)

    def apply(self, channel: str, raw: float) -> float:
        """Apply the calibration for ``channel`` (passthrough if unknown)."""
        table = self.tables.get(channel)
        if table is None:
            return float(raw)
        return table.apply(raw)

    def is_due(self, channel: str, now_ms: int) -> bool:
        """Return True if the channel's calibration is past due."""
        table = self.tables.get(channel)
        if table is None:
            return False
        return table.is_due(now_ms)

    def due_channels(self, now_ms: int) -> Iterable[str]:
        """Yield channel names whose calibration is past due."""
        for name, table in self.tables.items():
            if table.is_due(now_ms):
                yield name

    def __contains__(self, channel: str) -> bool:
        return channel in self.tables

    def __getitem__(self, channel: str) -> CalibrationTable:
        return self.tables[channel]

    def __len__(self) -> int:
        return len(self.tables)

    @classmethod
    def identity(
        cls,
        channels: Optional[Iterable[str]] = None,
        *,
        calibrated_at_ms: int = 0,
        valid_for_ms: int = 30 * 24 * 3600 * 1000,
    ) -> "CalibrationRegistry":
        """Build a passthrough registry for the given channels.

        Useful as a default: the abstraction layer never has to
        special-case "no calibration configured".
        """
        if channels is None:
            tables: Dict[str, CalibrationTable] = {}
        else:
            tables = {
                ch: CalibrationTable(
                    channel=ch,
                    gain=1.0,
                    offset=0.0,
                    poly_coeffs=(),
                    calibrated_at_ms=calibrated_at_ms,
                    valid_for_ms=valid_for_ms,
                )
                for ch in channels
            }
        return cls(tables=tables)


__all__ = ["CalibrationTable", "CalibrationRegistry"]
