"""
Embedded telemetry acquisition layer (PHASE 18).

This package is the "sensors → MCU → bus → edge" layer the user
spec called for. It is **monitoring and telemetry acquisition
only** — no aircraft control functionality is implemented here
and none will be.

The package is organised so a digital-twin / AI consumer sees the
**same wire format** whether the source is the in-process
simulator (default) or a real MCU connected over serial or CAN.

Public re-exports::

    from backend.acquisition import (
        # schema
        TELEMETRY_FIELDS, TELEMETRY_SCHEMA_VERSION,
        TelemetrySample, SensorHealth,
        # calibration
        CalibrationTable, CalibrationRegistry,
        # sampling
        ChannelSampling, DEFAULT_SAMPLING,
        # serial protocol
        McuFrame, COMMANDS, McuCrcError, McuFormatError,
        # CAN
        CanPort, CanMessage, LoopbackCanBus, NullCanBus, CAN_FRAME_MAP,
        # sensor abstraction
        SensorSource, SensorAbstraction,
        # simulation fallback
        SimulatedAcquisition,
        # config
        AcquisitionConfig, load_acquisition_config,
        # pipeline integration
        AcquisitionPipelineAdapter,
    )
"""

from .calibration import CalibrationRegistry, CalibrationTable
from .can_adapter import (
    CAN_FRAME_MAP,
    CanMessage,
    CanPort,
    LoopbackCanBus,
    NullCanBus,
    decode_can_messages_to_dict,
    encode_sample_to_can_messages,
)
from .pipeline_adapter import AcquisitionPipelineAdapter
from .sampling import DEFAULT_SAMPLING, ChannelSampling
from .schema import (
    TELEMETRY_FIELDS,
    TELEMETRY_SCHEMA_VERSION,
    SensorHealth,
    TelemetrySample,
)
from .sensor import SensorAbstraction, SensorSource
from .serial_protocol import (
    COMMANDS,
    McuCrcError,
    McuFormatError,
    McuFrame,
)
from .simulated import SimulatedAcquisition
from .config_loader import AcquisitionConfig, load_acquisition_config

__all__ = [
    "CAN_FRAME_MAP",
    "COMMANDS",
    "CalibrationRegistry",
    "CalibrationTable",
    "CanMessage",
    "CanPort",
    "ChannelSampling",
    "DEFAULT_SAMPLING",
    "LoopbackCanBus",
    "McuCrcError",
    "McuFormatError",
    "McuFrame",
    "NullCanBus",
    "SensorAbstraction",
    "SensorHealth",
    "SensorSource",
    "SimulatedAcquisition",
    "TELEMETRY_FIELDS",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetrySample",
    "AcquisitionPipelineAdapter",
    "AcquisitionConfig",
    "load_acquisition_config",
    "decode_can_messages_to_dict",
    "encode_sample_to_can_messages",
]
