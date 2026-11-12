"""Basler ace camera via pypylon. Frames are returned raw (e.g. Bayer mosaic)."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..device import Camera, DeviceConfig

SHAPE_MULTIPLE = 256  # width and height must be multiples of this (tiles in the data processing)


def _writable(node) -> bool:
    # GenICam access modes: 2 = write-only, 4 = read-write
    return node.GetAccessMode() in (2, 4)


@dataclass(frozen=True, kw_only=True)
class BaslerCameraConfig(DeviceConfig):
    serial_number: str
    pixel_format: str = "BayerRG12"
    width: int | None = None  # None: keep the camera's value
    height: int | None = None
    offset_x: int | None = None
    offset_y: int | None = None
    pfs: Path | None = None  # Basler feature file, loaded before the values above
    timeout_s: float = 10.0  # grab timeout on top of the exposure time

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value % SHAPE_MULTIPLE:
                raise ValueError(f"camera {name} {value} is not a multiple of {SHAPE_MULTIPLE}")


@dataclass(frozen=True)
class BaslerCameraState:
    exposure_ms: float
    gain: float  # dB


class BaslerCamera(Camera):
    driver = "basler_camera"
    Config = BaslerCameraConfig
    State = BaslerCameraState

    def __init__(self, name, config=None, /, *, sdk=None, **config_values):
        super().__init__(name, config, **config_values)
        self._sdk = sdk  # injectable pypylon.pylon for tests
        self._camera = None

    def resource(self):
        return ("basler", self.config.serial_number)

    @property
    def full_scale(self) -> int:
        """Largest pixel value for the pixel format (255, 1023, 4095, ...)."""
        bits = re.search(r"(\d+)", self.config.pixel_format)
        return 2 ** int(bits.group(1)) - 1 if bits else 255

    # ---- driver -----------------------------------------------------------

    async def _connect(self):
        await self._blocking(self._open)

    def _open(self):
        if self._sdk is None:
            from pypylon import pylon

            self._sdk = pylon
        pylon, c = self._sdk, self.config
        info = pylon.DeviceInfo()
        info.SetSerialNumber(c.serial_number)
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
        camera.Open()
        self._camera = camera
        if c.pfs is not None:
            pylon.FeaturePersistence.LoadFromString(c.pfs.read_text(), camera.GetNodeMap(), True)
        for node, value in (("ExposureAuto", "Off"), ("GainAuto", "Off"), ("TriggerMode", "Off")):
            if hasattr(camera, node):
                getattr(camera, node).SetValue(value)
        camera.PixelFormat.SetValue(c.pixel_format)
        # ROI. With CenterX/CenterY on, the camera centres the ROI itself and the
        # offsets are read-only; explicit offsets switch centring off.
        for axis, size, offset in (("X", c.width, c.offset_x), ("Y", c.height, c.offset_y)):
            offset_node = getattr(camera, f"Offset{axis}")
            if offset is not None and hasattr(camera, f"Center{axis}"):
                getattr(camera, f"Center{axis}").SetValue(False)
            if (offset is not None or size is not None) and _writable(offset_node):
                offset_node.SetValue(0)  # so that any width/height fits
            if size is not None:
                getattr(camera, "Width" if axis == "X" else "Height").SetValue(size)
            if offset is not None:
                offset_node.SetValue(offset)
        if camera.PixelFormat.GetValue() != c.pixel_format:
            raise RuntimeError(f"camera did not accept pixel format {c.pixel_format}")
        size = (camera.Width.GetValue(), camera.Height.GetValue())
        if size[0] % SHAPE_MULTIPLE or size[1] % SHAPE_MULTIPLE:
            raise RuntimeError(
                f"camera image is {size[0]}x{size[1]}; width and height must be multiples of "
                f"{SHAPE_MULTIPLE} (set width/height in the camera toml)"
            )
        device = camera.GetDeviceInfo()
        self.info = {
            "vendor": device.GetVendorName(),
            "model": device.GetModelName(),
            "serial_number": device.GetSerialNumber(),
            "device_version": device.GetDeviceVersion(),
            "pixel_format": camera.PixelFormat.GetValue(),
            "width": camera.Width.GetValue(),
            "height": camera.Height.GetValue(),
            "offset_x": camera.OffsetX.GetValue(),
            "offset_y": camera.OffsetY.GetValue(),
            "center_x": camera.CenterX.GetValue() if hasattr(camera, "CenterX") else None,
            "center_y": camera.CenterY.GetValue() if hasattr(camera, "CenterY") else None,
            "exposure_range_ms": [
                camera.ExposureTime.GetMin() / 1000,
                camera.ExposureTime.GetMax() / 1000,
            ],
            "gain_range": [camera.Gain.GetMin(), camera.Gain.GetMax()],
        }
        if hasattr(camera, "DeviceFirmwareVersion"):
            self.info["firmware_version"] = camera.DeviceFirmwareVersion.GetValue()

    async def _read(self):
        return await self._blocking(self._read_state)

    def _read_state(self):
        return BaslerCameraState(
            exposure_ms=self._camera.ExposureTime.GetValue() / 1000,
            gain=self._camera.Gain.GetValue(),
        )

    async def _apply(self, state, changed):
        await self._blocking(self._write, state, changed)

    def _write(self, state, changed):
        if "exposure_ms" in changed:
            self._camera.ExposureTime.SetValue(state.exposure_ms * 1000)
        if "gain" in changed:
            self._camera.Gain.SetValue(state.gain)

    def _grab(self) -> np.ndarray:
        camera = self._camera
        timeout_ms = round(self.config.timeout_s * 1000 + self.state.exposure_ms)
        camera.StartGrabbingMax(1)
        try:
            with camera.RetrieveResult(
                timeout_ms, self._sdk.TimeoutHandling_ThrowException
            ) as result:
                if not result.GrabSucceeded():
                    raise RuntimeError(f"grab failed: {result.ErrorDescription}")
                frame = np.array(result.Array, copy=True, order="C")
        finally:
            camera.StopGrabbing()
        if frame.shape[0] % SHAPE_MULTIPLE or frame.shape[1] % SHAPE_MULTIPLE:
            raise RuntimeError(f"frame shape {frame.shape} is not a multiple of {SHAPE_MULTIPLE}")
        return frame

    async def _close(self):
        if self._camera is not None:
            await self._blocking(self._shutdown)

    def _shutdown(self):
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            finally:
                camera.Close()
