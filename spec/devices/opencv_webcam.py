"""Any USB/V4L2 webcam via OpenCV. Returns RGB uint8."""

from dataclasses import dataclass

import numpy as np

from ..device import Camera, DeviceConfig


@dataclass(frozen=True, kw_only=True)
class OpenCVWebcamConfig(DeviceConfig):
    device: str | int = "/dev/video0"
    width: int | None = None  # None: camera default
    height: int | None = None
    fps: float | None = None
    fourcc: str | None = None  # e.g. "MJPG"
    discard_frames: int = 3  # webcams buffer frames; drop these to get a fresh one


class OpenCVWebcam(Camera):
    driver = "opencv_webcam"
    Config = OpenCVWebcamConfig

    def __init__(self, name, config=None, /, *, cv2_module=None, **config_values):
        super().__init__(name, config, **config_values)
        self._cv2 = cv2_module  # injectable for tests
        self._capture = None

    def resource(self):
        return ("webcam", str(self.config.device))

    async def _connect(self):
        await self._blocking(self._open)

    def _open(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        cv2, c = self._cv2, self.config
        capture = cv2.VideoCapture(c.device, cv2.CAP_V4L2)
        self._capture = capture
        if not capture.isOpened():
            raise RuntimeError(f"cannot open webcam {c.device}")
        if c.fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, c.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, c.height),
            (cv2.CAP_PROP_FPS, c.fps),
        ):
            if value is not None:
                capture.set(prop, value)
        self.info = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "backend": capture.getBackendName(),
        }

    def _grab(self) -> np.ndarray:
        for _ in range(self.config.discard_frames):
            self._capture.grab()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"webcam {self.config.device} returned no frame")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    async def _close(self):
        if self._capture is not None:
            await self._blocking(self._release)

    def _release(self):
        capture, self._capture = self._capture, None
        capture.release()
