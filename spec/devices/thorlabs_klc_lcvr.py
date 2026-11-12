"""Thorlabs KLC101 controller driving one liquid-crystal variable retarder.

Uses KLCserial 0.27 with stricter checks: that library silently turns a missing
reply into 0 V and prints instead of raising.
"""

import asyncio
import math
import time
from dataclasses import dataclass

from ..device import Device, DeviceConfig

VOLTAGE_REPLY_HEADER = bytes.fromhex("03 20 06 00")
REPLY_QUIET_S = 0.02  # a reply is complete after this long without new bytes


@dataclass(frozen=True, kw_only=True)
class ThorlabsKlcLcvrConfig(DeviceConfig):
    serial_number: str
    timeout_s: float = 2.0
    settle_ms: float = 0.0  # wait after a voltage change (LC response time)
    readback_tolerance_v: float = 0.002


@dataclass(frozen=True)
class ThorlabsKlcLcvrState:
    voltage_v: float


def _open_klc(config):
    """Open a KLC101 by serial number; returns a checked KLCserial controller."""
    from KLCserial import KLC

    class CheckedKLC(KLC):
        last_reply = ""
        last_decoded = None

        def __init__(self):
            try:
                super().__init__(SN=config.serial_number)
            except BaseException:
                if getattr(self, "ser", None) is not None:
                    self.ser.close()
                raise

        def sendcmd(self, cmd):
            if not self.port_is_open():
                raise ConnectionError("KLC port is closed")
            self.ser.reset_input_buffer()  # drop stale bytes from earlier replies
            payload = bytes.fromhex(cmd)
            if self.ser.write(payload) != len(payload):
                raise OSError("incomplete KLC serial write")

        def recvreply(self):
            # The original sleeps a fixed 40 ms; wait for the reply instead.
            deadline = time.monotonic() + config.timeout_s
            reply, last = b"", None
            while time.monotonic() < deadline:
                waiting = self.ser.in_waiting
                if waiting:
                    reply += self.ser.read(waiting)
                    last = time.monotonic()
                elif last is not None and time.monotonic() - last > REPLY_QUIET_S:
                    break
                time.sleep(0.002)
            return "".join(f"{byte:02x} " for byte in reply)

        def openKLC(self, *args, **kwargs):
            super().openKLC(*args, **kwargs)
            if self.ser is None or not self.ser.is_open:
                raise ConnectionError(
                    f"KLC {config.serial_number} not found under /dev/serial/by-id"
                )
            self.ser.timeout = config.timeout_s
            self.ser.write_timeout = config.timeout_s

        def decodereply(self, reply):
            self.last_reply = reply
            self.last_decoded = super().decodereply(reply)
            return self.last_decoded

        def output_enabled(self) -> bool:  # hardware channel enable (front panel)
            self.sendcmd(self.cmds["req_HWchan_status"])
            return self.decodereply(self.recvreply())[1] == "1"

        def enable_output(self):
            """Switch the output on and select mode V1; verify both."""
            if not self.output_enabled():
                self.sendcmd(self.cmds["enable_HWchan"])
                if not self.output_enabled():
                    raise OSError("KLC did not enable its output")
            self.last_decoded = None
            self.set_chan_mode(1)
            if self.last_decoded != ("channel mode", "1"):
                raise OSError("KLC did not confirm output mode V1")

        def get_chan_mode(self):  # the original only prints the mode
            self.sendcmd(self.cmds["req_chan_mode"])
            return self.decodereply(self.recvreply())

        def closeKLC(self):
            if self.ser is not None and self.ser.is_open:
                self.ser.close()

    return CheckedKLC()


class ThorlabsKlcLcvr(Device):
    driver = "thorlabs_klc_lcvr"
    Config = ThorlabsKlcLcvrConfig
    State = ThorlabsKlcLcvrState
    required_limits = ("voltage_v",)

    def __init__(self, name, config=None, /, *, open_klc=None, **config_values):
        super().__init__(name, config, **config_values)
        self._open_klc = open_klc or _open_klc  # injectable for tests
        self._klc = None

    def resource(self):
        return ("klc", self.config.serial_number)

    # ---- driver -----------------------------------------------------------

    async def _connect(self):
        await self._blocking(self._open)

    def _open(self):
        self._klc = self._open_klc(self.config)
        self._klc.enable_output()  # ready to work: output on, mode V1
        self.info = {"serial_number": self.config.serial_number, "port": self._klc.ser.port}

    async def _read(self):
        return ThorlabsKlcLcvrState(voltage_v=await self._blocking(self._voltage))

    def _voltage(self) -> float:
        klc = self._klc
        klc.last_reply = ""
        value = klc.get_voltage(channel=1)
        raw = bytes.fromhex(klc.last_reply)
        if len(raw) != 12 or raw[:4] != VOLTAGE_REPLY_HEADER:
            raise OSError("missing, incomplete or unexpected KLC voltage reply")
        if value is None or not math.isfinite(value) or not 0 <= value <= 25:
            raise OSError(f"invalid KLC voltage readback {value!r}")
        return float(value)

    async def _apply(self, state, changed):
        await self._blocking(self._set_voltage, state.voltage_v)
        if self.config.settle_ms:
            await asyncio.sleep(self.config.settle_ms / 1000)

    def _set_voltage(self, value):
        klc = self._klc
        klc.set_voltage(value, channel=1)
        klc.last_decoded = None
        klc.set_chan_mode(1)
        if klc.last_decoded != ("channel mode", "1"):
            raise OSError("KLC did not confirm output mode V1")
        actual = self._voltage()
        if abs(actual - value) > self.config.readback_tolerance_v:
            raise OSError(f"KLC voltage mismatch: requested {value} V, read {actual} V")

    async def _close(self):
        if self._klc is not None:
            await self._blocking(self._shutdown)

    def _shutdown(self):
        klc, self._klc = self._klc, None
        klc.closeKLC()  # the output keeps its voltage
