"""
Acquisition configuration (PHASE 18 — embedded telemetry acquisition).

A :class:`AcquisitionConfig` is the operator-facing knob bag for
the acquisition layer. The on-disk YAML is loaded by
:func:`load_acquisition_config`; the layer is runnable from a
fresh checkout with no YAML at all (defaults to a fully
simulated stack with the default sampling table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .sampling import DEFAULT_SAMPLING, ChannelSampling


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AcquisitionConfig:
    """All knobs the acquisition layer needs."""

    mission_id: str = "mission-001"
    vehicle_id: str = "vehicle-001"
    engine_id: str = "engine-001"

    # "simulated" / "serial" / "can"
    source: str = "simulated"

    # Serial settings — only used when source == "serial".
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 115200
    serial_crc: str = "ccitt"            # "ccitt" / "none"

    # CAN settings — only used when source == "can".
    can_interface: str = "socketcan"
    can_channel: str = "can0"
    can_bitrate: int = 500_000

    # Pin map — purely informational, the edge only logs it.
    # The MCU firmware reads its own pin map.
    pins: Dict[str, str] = field(default_factory=dict)

    # Sampling — keyed by channel name, with native + bus rates.
    sampling: Dict[str, ChannelSampling] = field(
        default_factory=lambda: dict(DEFAULT_SAMPLING)
    )

    # Calibration validity window (ms).
    default_valid_for_ms: int = 30 * 24 * 3600 * 1000

    # Plausibility limits — populated from the spec ranges; the
    # operator can tighten them in YAML.
    plausible_ranges: Dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def defaults(cls) -> "AcquisitionConfig":
        return cls()


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_acquisition_config(
    config_dir: Optional[Path | str] = None,
) -> AcquisitionConfig:
    """Load ``config/acquisition.yaml`` and return an :class:`AcquisitionConfig`.

    Forgiving: a missing file returns the defaults. A present
    but invalid file raises the same :class:`ConfigError` the
    rest of the project uses.
    """
    from backend.config import ConfigError  # late import to avoid cycles

    base = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    path = base / "acquisition.yaml"
    if not path.exists():
        return AcquisitionConfig.defaults()

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to load YAML configuration. "
            "Install it with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("acquisition.yaml must be a mapping")
    inner = raw.get("acquisition", raw)
    if not isinstance(inner, dict):
        return AcquisitionConfig.defaults()
    try:
        return _parse(inner)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"acquisition configuration validation failed: {exc}") from exc


def _parse(d: Dict[str, Any]) -> AcquisitionConfig:
    sampling: Dict[str, ChannelSampling] = dict(DEFAULT_SAMPLING)
    raw_sampling = d.get("sampling", {})
    if isinstance(raw_sampling, dict):
        for ch, cfg in raw_sampling.items():
            if not isinstance(cfg, dict):
                continue
            native = float(cfg.get("native_rate_hz", 1.0))
            bus = float(cfg.get("bus_rate_hz", 1.0))
            if bus <= 0.0:
                bus = 1.0
            if native < bus:
                native = bus
            sampling[str(ch)] = ChannelSampling(
                channel=str(ch),
                native_rate_hz=native,
                bus_rate_hz=bus,
            )

    serial = d.get("serial", {}) or {}
    can = d.get("can", {}) or {}

    pins = d.get("pins", {}) or {}
    if not isinstance(pins, dict):
        pins = {}

    plausible = d.get("plausible_ranges", {}) or {}
    parsed_plausible: Dict[str, tuple[float, float]] = {}
    if isinstance(plausible, dict):
        for ch, lim in plausible.items():
            if isinstance(lim, (list, tuple)) and len(lim) == 2:
                parsed_plausible[str(ch)] = (float(lim[0]), float(lim[1]))

    return AcquisitionConfig(
        mission_id=str(d.get("mission_id", "mission-001")),
        vehicle_id=str(d.get("vehicle_id", "vehicle-001")),
        engine_id=str(d.get("engine_id", "engine-001")),
        source=str(d.get("source", "simulated")),
        serial_port=str(serial.get("port", "/dev/ttyUSB0")),
        serial_baudrate=int(serial.get("baudrate", 115200)),
        serial_crc=str(serial.get("crc", "ccitt")),
        can_interface=str(can.get("interface", "socketcan")),
        can_channel=str(can.get("channel", "can0")),
        can_bitrate=int(can.get("bitrate", 500_000)),
        pins={str(k): str(v) for k, v in pins.items()},
        sampling=sampling,
        default_valid_for_ms=int(
            d.get("default_valid_for_ms", 30 * 24 * 3600 * 1000)
        ),
        plausible_ranges=parsed_plausible,
    )


__all__ = ["AcquisitionConfig", "load_acquisition_config"]
