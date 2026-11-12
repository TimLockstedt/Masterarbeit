"""One axis of a Zaber controller (zaber-motion ASCII).

Axes on the same serial port share one connection; it is opened by the first
axis that connects and closed by the last one that closes.
"""

import asyncio
from dataclasses import dataclass

from ..device import Device, DeviceConfig


@dataclass(frozen=True, kw_only=True)
class ZaberAxisConfig(DeviceConfig):
    port: str  # prefer /dev/serial/by-id/... (stable across reboots)
    device_address: int
    axis: int
    timeout_s: float = 30.0  # per move
    settle_ms: float = 0.0  # wait after a move
    home: bool = False  # True: connect homes an unreferenced axis (moves!); False: connect fails


@dataclass(frozen=True)
class ZaberAxisState:
    position_mm: float


class _Shared:
    def __init__(self, connection, devices):
        self.connection, self.devices, self.users = connection, devices, 0


_connections: dict[str, _Shared] = {}
_lock: asyncio.Lock | None = None


def _connection_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _open_serial(port):
    from zaber_motion.ascii import Connection

    return await Connection.open_serial_port_async(port)


class ZaberAxis(Device):
    driver = "zaber_axis"
    Config = ZaberAxisConfig
    State = ZaberAxisState
    required_limits = ("position_mm",)

    def __init__(self, name, config=None, /, *, open_serial=None, **config_values):
        super().__init__(name, config, **config_values)
        self._open_serial = open_serial or _open_serial  # injectable for tests
        self._axis = None

    def resource(self):
        c = self.config
        return ("zaber", c.port, c.device_address, c.axis)

    def home(self):
        """Home the axis (blocks), then read the position."""
        if not self.connected:
            raise RuntimeError(f"{self.name} is not connected")
        self._run(self._home)
        return self.read()

    # ---- driver -----------------------------------------------------------

    async def _connect(self):
        c = self.config
        async with _connection_lock():
            shared = _connections.get(c.port)
            if shared is None:
                connection = await self._open_serial(c.port)
                try:
                    found = await connection.detect_devices_async()
                except BaseException:
                    await connection.close_async()
                    raise
                shared = _connections[c.port] = _Shared(
                    connection, {d.device_address: d for d in found}
                )
            shared.users += 1
        self._shared = shared
        device = shared.devices.get(c.device_address)
        if device is None:
            raise RuntimeError(
                f"no Zaber device at address {c.device_address} on {c.port}; "
                f"found {sorted(shared.devices)}"
            )
        if not 1 <= c.axis <= device.axis_count:
            raise RuntimeError(f"Zaber device {c.device_address} has {device.axis_count} axes")
        self._axis = device.get_axis(c.axis)
        self.info = {
            "device": device.name,
            "serial_number": device.serial_number,
            "firmware": "{0.major}.{0.minor}.{0.build}".format(device.firmware_version),
            "peripheral": self._axis.peripheral_name,
        }
        # Ready to work: unparked and referenced.
        if await self._axis.is_parked_async():
            await self._axis.unpark_async()
        if not await self._axis.is_homed_async():
            if not c.home:
                raise RuntimeError(
                    f"{self.name} is not homed (no reference position). Connect never moves an axis "
                    f"by default: home it explicitly (controller, Zaber Launcher), or force it for one "
                    f"measurement with [devices.{self.name}] config = {{home = true}}"
                )
            await self._home()

    async def _read(self):
        position = await asyncio.wait_for(
            self._axis.get_position_async("mm"), self.config.timeout_s
        )
        return ZaberAxisState(position_mm=position)

    async def _apply(self, state, changed):
        await self._motion(
            self._axis.move_absolute_async(state.position_mm, "mm", wait_until_idle=True)
        )

    async def _home(self):
        await self._motion(self._axis.home_async(wait_until_idle=True))

    async def _motion(self, operation):
        try:
            await asyncio.wait_for(operation, self.config.timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(self._axis.stop_async(), self.config.timeout_s)
            raise
        if self.config.settle_ms:
            await asyncio.sleep(self.config.settle_ms / 1000)

    async def _close(self):
        shared = getattr(self, "_shared", None)
        if shared is None:
            return
        self._shared, self._axis = None, None
        async with _connection_lock():
            shared.users -= 1
            if shared.users == 0:
                _connections.pop(self.config.port, None)
                await shared.connection.close_async()
