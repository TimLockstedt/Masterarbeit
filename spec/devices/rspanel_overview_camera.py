"""Still camera on the rspanel Raspberry Pi (HTTP, PNG). Returns RGB uint8."""

import socket
from dataclasses import dataclass

import numpy as np

from ..device import Camera, DeviceConfig


@dataclass(frozen=True, kw_only=True)
class RspanelOverviewCameraConfig(DeviceConfig):
    host: str
    port: int = 9235
    resolution: str = "2304x1296"  # 1536x864, 2304x1296, 4608x2592, max, 1080p
    timeout_s: float = 45.0


class RspanelOverviewCamera(Camera):
    driver = "rspanel_overview_camera"
    Config = RspanelOverviewCameraConfig

    def __init__(self, name, config=None, /, *, client=None, **config_values):
        super().__init__(name, config, **config_values)
        self._client = client  # injectable for tests

    def resource(self):
        return ("rspanel_camera", self.config.host, self.config.port)

    async def _connect(self):
        await self._blocking(self._open)

    def _open(self):
        c = self.config
        if self._client is None:
            from rspanel import CameraClient

            self._client = CameraClient(c.host, c.port, timeout=c.timeout_s)
            # The camera API is plain HTTP; check that the listener is reachable.
            socket.create_connection((c.host, c.port), timeout=c.timeout_s).close()
        self.info = {"host": c.host, "port": c.port, "resolution": c.resolution}

    def _grab(self) -> np.ndarray:
        import cv2

        png = self._client.capture_png(self.config.resolution)
        frame = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("overview PNG could not be decoded")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    async def _close(self):
        self._client = None
