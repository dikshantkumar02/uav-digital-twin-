"""
Feature extraction (PHASE 9).

A pure function that turns a :class:`~backend.ml.window.WindowBuffer`
into a fixed-length numpy array matching :data:`FEATURE_NAMES`. The
contract is:

* Output is always shape ``(FEATURE_DIM,)`` (1-D).
* Missing channels (no residuals in the window) get zero-filled.
* Deterministic given the input window.
"""

from __future__ import annotations

import numpy as np

from backend.diagnostics.fusion import ALL_KNOWN_CHANNELS

from .types import FEATURE_DIM, FEATURE_NAMES
from .window import WindowBuffer


# Channels whose per-channel stats we compute. Matches the order in
# FEATURE_NAMES — keep these two lists in sync.
_RESIDUAL_CHANNELS = (
    "rpm", "egt", "cht", "oil_pressure", "oil_temperature",
    "fuel_flow", "vibration", "altitude", "airspeed",
    "ambient_temperature", "ambient_pressure",
)

assert len(_RESIDUAL_CHANNELS) == 11  # 11 channels x 4 stats = 44


def _safe_stats(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return (mean_abs, max_abs, std, outlier_count) for an array of z-scores.

    Outliers are |z| > 3. Empty input returns zeros.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0
    abs_v = np.abs(values)
    return (
        float(abs_v.mean()),
        float(abs_v.max()),
        float(values.std()),
        int((abs_v > 3.0).sum()),
    )


class FeatureExtractor:
    """Turns a WindowBuffer into a feature vector."""

    def extract(self, window: WindowBuffer) -> np.ndarray:
        """Compute the feature vector for a (possibly partial) window."""
        out = np.zeros(FEATURE_DIM, dtype=np.float64)
        ticks = window.ticks()
        if not ticks:
            return out

        # ----- Per-channel residual stats ----------------------------
        # Collect z-scores per channel across the window.
        per_ch_z: dict[str, list[float]] = {ch: [] for ch in _RESIDUAL_CHANNELS}
        for tick in ticks:
            for ch, r in tick.residual.residuals.items():
                if ch in per_ch_z and r.z_score is not None:
                    per_ch_z[ch].append(float(r.z_score))
        for ch in _RESIDUAL_CHANNELS:
            arr = np.array(per_ch_z[ch], dtype=np.float64)
            mean_abs, max_abs, std, out_count = _safe_stats(arr)
            base = FEATURE_NAMES.index(f"res.{ch}.mean_abs_z")
            out[base + 0] = mean_abs
            out[base + 1] = max_abs
            out[base + 2] = std
            out[base + 3] = out_count

        # ----- Per-window sensor-health aggregates --------------------
        counts = {"DROPPED": 0, "STUCK": 0, "DRIFTING": 0, "FAULT": 0, "SPIKE": 0}
        for tick in ticks:
            if tick.sample is None or not tick.sample.readings:
                continue
            for reading in tick.sample.readings.values():
                m = str(reading.mode.value).split(".")[-1]
                if m in counts:
                    counts[m] += 1
        out[FEATURE_NAMES.index("sensors.dropped_count")] = counts["DROPPED"]
        out[FEATURE_NAMES.index("sensors.stuck_count")] = counts["STUCK"]
        out[FEATURE_NAMES.index("sensors.drift_count")] = counts["DRIFTING"]
        out[FEATURE_NAMES.index("sensors.fault_count")] = counts["FAULT"]
        out[FEATURE_NAMES.index("sensors.spike_count")] = counts["SPIKE"]

        # ----- Overall residual stats --------------------------------
        all_z = np.concatenate([
            np.array(per_ch_z[ch], dtype=np.float64) for ch in _RESIDUAL_CHANNELS
        ]) if any(per_ch_z.values()) else np.array([], dtype=np.float64)
        mean_abs, max_abs, _, out_count = _safe_stats(all_z)
        out[FEATURE_NAMES.index("residual.overall_mean_abs_z")] = mean_abs
        out[FEATURE_NAMES.index("residual.overall_max_abs_z")] = max_abs
        out[FEATURE_NAMES.index("residual.overall_outlier_count")] = out_count

        # ----- Flight state (means) ----------------------------------
        rpms, maps, alts, airspds, throttles = [], [], [], [], []
        for tick in ticks:
            if tick.env is None:
                continue
            throttles.append(float(tick.env.throttle))
            alts.append(float(tick.env.altitude_m))
            airspds.append(float(tick.env.airspeed_mps))
            if tick.sample is not None and "rpm" in tick.sample.readings:
                v = tick.sample.readings["rpm"].value
                if v is not None:
                    rpms.append(float(v))
        if rpms:
            out[FEATURE_NAMES.index("flight.mean_rpm")] = float(np.mean(rpms))
        # We don't get manifold_pressure directly on the env, so this
        # feature is best-effort: a constant if unavailable. We leave
        # it 0 in the current pipeline.
        if alts:
            out[FEATURE_NAMES.index("flight.mean_altitude_m")] = float(np.mean(alts))
        if airspds:
            out[FEATURE_NAMES.index("flight.mean_airspeed_mps")] = float(np.mean(airspds))
        if throttles:
            out[FEATURE_NAMES.index("flight.mean_throttle")] = float(np.mean(throttles))

        # ----- Environment (means) -----------------------------------
        pressures, temps, winds = [], [], []
        for tick in ticks:
            if tick.env is None:
                continue
            pressures.append(float(tick.env.atmosphere.pressure_pa))
            temps.append(float(tick.env.atmosphere.temperature_c))
            winds.append(float(tick.env.wind.total_w_mps))
        if pressures:
            out[FEATURE_NAMES.index("env.mean_pressure_pa")] = float(np.mean(pressures))
        if temps:
            out[FEATURE_NAMES.index("env.mean_temperature_c")] = float(np.mean(temps))
        if winds:
            out[FEATURE_NAMES.index("env.mean_wind_mps")] = float(np.mean(winds))

        return out


__all__ = ["FeatureExtractor"]
