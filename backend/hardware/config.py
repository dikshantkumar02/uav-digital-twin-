"""
Re-export :class:`HardwareConfig` from the central config package
so the transport code can do::

    from backend.hardware.config import HardwareConfig

without depending on the public config surface.
"""

from backend.config import HardwareConfig

__all__ = ["HardwareConfig"]
