"""rspanel LED display (own project, ../rspanel). The pattern is uint8 RGB.

The display cannot be read back: after connect it is cleared to black, and the
state is what was last shown. Closing clears it to black again.

``preload(stack)`` uploads a sequence once; afterwards ``set(pattern=...)`` with
one of its patterns only switches the shown index (fast, nothing uploaded).
Any other pattern is uploaded as before and ends the preloaded sequence.
"""

import asyncio
import dataclasses
import functools
import hashlib
from dataclasses import dataclass

import numpy as np

from ..device import Device, DeviceConfig


@dataclass(frozen=True, kw_only=True)
class RspanelDisplayConfig(DeviceConfig):
    host: str
    port: int = 9234
    width: int
    height: int
    x_offset: int = 0
    y_offset: int = 0
    timeout_s: float = 5.0
    load_timeout_s: float = 60.0
    settle_ms: float = 0.0  # wait after the display reports READY


@dataclass(frozen=True)
class RspanelDisplayState:
    pattern: np.ndarray  # uint8, shape (height, width, 3)


class RspanelDisplay(Device):
    driver = "rspanel_display"
    Config = RspanelDisplayConfig
    State = RspanelDisplayState

    def __init__(self, name, config=None, /, *, client=None, **config_values):
        super().__init__(name, config, **config_values)
        self._client = client  # injectable for tests
        self._preloaded: dict[bytes, int] = {}  # pattern hash -> index on the display

    def resource(self):
        return ("rspanel_display", self.config.host, self.config.port)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.config.height, self.config.width, 3)

    def black(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)

    def off(self):
        return self.set(pattern=self.black())

    def preload(self, patterns: np.ndarray):
        """Upload a sequence (N, H, W, 3) once and wait; then ``set(pattern=patterns[i])``
        only switches the index. The shown pattern does not change."""
        if not self.connected:
            raise RuntimeError(f"{self.name} is not connected")
        stack = np.array(patterns, dtype=np.uint8, copy=True)
        if stack.ndim != 4:
            raise ValueError(f"{self.name}: preload needs a stack (N, H, W, 3), got {stack.shape}")
        for pattern in stack:
            self._check_state({"pattern": pattern})
        self._run(functools.partial(self._preload, stack))
        self.wait()
        return self

    def _check_state(self, values):
        pattern = values.get("pattern")
        if pattern is not None and (pattern.dtype != np.uint8 or pattern.shape != self.shape):
            raise ValueError(
                f"{self.name}: pattern must be uint8 with shape {self.shape}, "
                f"got {pattern.dtype} {pattern.shape}"
            )
        return values

    # ---- driver -----------------------------------------------------------

    async def _connect(self):
        c = self.config
        if self._client is None:
            from rspanel import AsyncDisplayClient, SessionViewport

            self._client = AsyncDisplayClient(
                c.host,
                c.port,
                timeout=c.timeout_s,
                load_timeout=c.load_timeout_s,
                expected_width=c.width,
                expected_height=c.height,
                viewport=SessionViewport(
                    width=c.width, height=c.height, x_offset=c.x_offset, y_offset=c.y_offset
                ),
            )
        await self._client.connect()
        await self._client.clear_patterns()
        self._preloaded = {}
        identity = self._client.identity
        if dataclasses.is_dataclass(identity):
            self.info = dataclasses.asdict(identity)
        self._shown = self.black()

    async def _read(self):
        return RspanelDisplayState(pattern=self._shown)

    async def _preload(self, stack):
        self._preloaded = {}
        keys = [_key(pattern) for pattern in stack]
        shown = _key(self._shown)
        if shown not in keys:  # keep showing it: upload it with the sequence
            stack = np.concatenate([stack, self._shown[np.newaxis]])
            keys.append(shown)
        await self._client.load_patterns(stack)
        await self._client.show_pattern(keys.index(shown))
        self._preloaded = {key: i for i, key in reversed(list(enumerate(keys)))}  # first index wins

    async def _apply(self, state, changed):
        index = self._preloaded.get(_key(state.pattern))
        if index is None:
            self._preloaded = {}
            await self._client.load_patterns(state.pattern[np.newaxis])
            index = 0
        await self._client.show_pattern(index)
        self._shown = state.pattern
        if self.config.settle_ms:
            await asyncio.sleep(self.config.settle_ms / 1000)

    async def _close(self):
        client, self._client = self._client, None
        if client is None:
            return
        try:
            if client.connected:
                await client.clear_patterns()
        finally:
            await client.close()


def _key(pattern: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(pattern).tobytes()).digest()
