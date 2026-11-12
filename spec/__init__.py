"""SPEC: microscope devices and plain-Python measurements. See docs/DESIGN.md."""

from importlib.metadata import version as _version

__version__ = _version("spec")  # from the git tag at install time (hatch-vcs)

from . import devices  # noqa: E402,F401  (registers the drivers)
from .device import Camera, Device, DeviceConfig, DeviceError  # noqa: E402
from .run import Run  # noqa: E402
from .setup import Setup  # noqa: E402

__all__ = ["Camera", "Device", "DeviceConfig", "DeviceError", "Run", "Setup"]
