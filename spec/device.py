"""Device base classes.

A device has a frozen ``Config`` (connection, ``limits``, ``defaults``) and a
frozen ``State`` (the values a measurement changes). Users call plain,
synchronous methods; the hardware work runs on one shared background event
loop so that different devices work in parallel. See docs/DESIGN.md.
"""

import asyncio
import concurrent.futures
import dataclasses
import functools
import hashlib
import math
import threading
import tomllib
import types
import typing
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


class DeviceError(RuntimeError):
    """A device operation failed (as opposed to invalid input)."""


# --------------------------------------------------------------------------
# Background event loop shared by all devices.

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _event_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="spec-devices", daemon=True
            ).start()
        return _loop


def _submit(coroutine) -> concurrent.futures.Future:
    loop = _event_loop()
    if threading.current_thread().name == "spec-devices":
        coroutine.close()
        raise RuntimeError("device methods must not be called from driver code")
    return asyncio.run_coroutine_threadsafe(coroutine, loop)


# --------------------------------------------------------------------------
# Validation.


def _check(name: str, value: Any, hint: Any, limits: Mapping) -> Any:
    """Return ``value`` converted to ``hint``'s type, or raise TypeError/ValueError."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(hint)
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _check(name, value, option, limits)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
        raise TypeError("; ".join(errors))

    if hint is float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    elif hint is int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        value = int(value)
    elif hint is bool:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be true or false, got {value!r}")
        value = bool(value)
    elif hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {value!r}")
    elif hint is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be a path, got {value!r}")
        value = Path(value)
    elif hint is np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(value).__name__}")
        value = np.array(value, copy=True, order="C")
        value.flags.writeable = False
    elif hint is Any:
        pass
    elif not isinstance(value, origin or hint):
        raise TypeError(f"{name} must be {hint}, got {value!r}")

    if name in limits:
        low, high = limits[name]
        if not low <= value <= high:
            raise ValueError(f"{name}={value} is outside the limits [{low}, {high}]")
    return value


def check_fields(cls, values: Mapping, limits: Mapping = {}) -> dict:
    """Check a partial set of fields of dataclass ``cls``; return converted values."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise TypeError(
            f"{cls.__name__}: unknown field(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(names) or 'none'}"
        )
    return {key: _check(key, value, hints[key], limits) for key, value in values.items()}


def validated(cls, values: Mapping, limits: Mapping = {}):
    """Build dataclass ``cls`` from ``values`` with type and limit checks."""
    checked = check_fields(cls, values, limits)
    missing = [
        f.name
        for f in dataclasses.fields(cls)
        if f.name not in checked
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    if missing:
        raise TypeError(f"{cls.__name__}: missing field(s) {', '.join(missing)}")
    return cls(**checked)


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    return a == b


def jsonable(value):
    """Convert config/state values for JSON; arrays become sha1 + shape + dtype."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return {
            "sha1": hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _merge(base: dict, override: Mapping) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


# --------------------------------------------------------------------------
# Config and device.


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeviceConfig:
    """Base config. ``limits``: {state field: [min, max]}; ``defaults``: start state."""

    limits: dict = dataclasses.field(default_factory=dict)
    defaults: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class NoState:
    """State of a device with nothing to set (e.g. a webcam)."""


DRIVERS: dict[str, type["Device"]] = {}


class Device:
    """Template for all devices. Subclasses implement the ``_``-prefixed coroutines."""

    driver: ClassVar[str] = ""
    Config: ClassVar[type] = DeviceConfig
    State: ClassVar[type] = NoState
    required_limits: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.driver:
            DRIVERS[cls.driver] = cls

    def __init__(self, name: str, config=None, /, **config_values):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"device name must be a Python identifier, got {name!r}")
        if config is None:
            config = validated(self.Config, config_values)
        elif config_values:
            raise TypeError("pass either a config object or keyword values, not both")
        elif not isinstance(config, self.Config):
            raise TypeError(f"{name}: config must be {self.Config.__name__}")
        self.name = name
        self.config = config
        self.limits = self._checked_limits(config.limits)
        self.defaults = check_fields(self.State, config.defaults, self.limits)
        self.defaults = self._check_state(self.defaults)
        self.state = None  # requested state; None while not connected
        self.info: dict = {}  # device identity etc., filled by _connect
        self.source: dict = {}  # toml file, filled by from_toml
        self.setup = None  # set by Setup
        self._last: concurrent.futures.Future | None = None
        self._error: BaseException | None = None
        self._worker: concurrent.futures.ThreadPoolExecutor | None = None

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"

    def _checked_limits(self, limits: Mapping) -> dict:
        fields = {f.name for f in dataclasses.fields(self.State)}
        result = {}
        for key, value in limits.items():
            if key not in fields:
                raise ValueError(
                    f"{self.name}: limit for unknown state field {key!r}; "
                    f"state fields: {', '.join(sorted(fields)) or 'none'}"
                )
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
                )
                or not value[0] <= value[1]
            ):
                raise ValueError(f"{self.name}: limit {key} must be [min, max], got {value!r}")
            result[key] = (float(value[0]), float(value[1]))
        missing = [key for key in self.required_limits if key not in result]
        if missing:
            raise ValueError(f"{self.name}: limits required for {', '.join(missing)}")
        return result

    # ---- construction from toml ------------------------------------------

    @classmethod
    def from_toml(cls, path, name=None, *, config: Mapping = {}, defaults: Mapping = {}):
        """Load a device toml. ``config`` overrides config values, ``defaults``
        overrides the start state (both used by measurement tomls)."""
        path = Path(path)
        text = path.read_text()
        data = tomllib.loads(text)
        driver = data.pop("driver", None)
        if cls is Device:
            if driver not in DRIVERS:
                raise ValueError(f"{path}: unknown driver {driver!r}; known: {sorted(DRIVERS)}")
            cls = DRIVERS[driver]
        elif driver != cls.driver:
            raise ValueError(f"{path}: driver is {driver!r}, expected {cls.driver!r}")
        data = _merge(data, config)
        data["defaults"] = {**data.get("defaults", {}), **defaults}
        hints = typing.get_type_hints(cls.Config)
        for key, value in data.items():
            if Path in typing.get_args(hints.get(key)) or hints.get(key) is Path:
                if isinstance(value, str) and not Path(value).is_absolute():
                    data[key] = path.parent / value
        try:
            device = cls(name or path.stem, validated(cls.Config, data))
        except (TypeError, ValueError) as error:
            raise type(error)(f"{path}: {error}") from None
        device.source = {
            "path": str(path.resolve()),
            "toml": text,
            "config_overrides": dict(config),
            "default_overrides": dict(defaults),
        }
        return device

    # ---- user API -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.state is not None

    def connect(self, wait: bool = True):
        """Connect, read the hardware state and apply ``defaults``."""
        if self.connected:
            return self
        self._error = None
        self._worker = concurrent.futures.ThreadPoolExecutor(1, f"spec-{self.name}")

        async def connecting():
            await self._connect()
            current = await self._read()
            target = dataclasses.replace(current, **self.defaults)
            changed = {k for k in self.defaults if not _same(getattr(current, k), self.defaults[k])}
            if changed:
                await self._apply(target, changed)
            self.state = target

        self._run(connecting)
        if wait:
            try:
                self.wait()
            except BaseException:
                try:
                    self.close()
                except Exception:
                    pass
                raise
        return self

    def set(self, state=None, /, **values):
        """Validate now; apply the changed fields in the background; return immediately."""
        if not self.connected:
            raise DeviceError(f"{self.name} is not connected")
        self._raise_if_failed()
        if state is None:
            new = {}
        elif isinstance(state, self.State):
            new = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
        elif isinstance(state, Mapping):
            new = dict(state)
        else:
            raise TypeError(
                f"{self.name}: set() takes a {self.State.__name__}, a dict or keywords"
            )
        new.update(values)
        checked = self._check_state(check_fields(self.State, new, self.limits))
        changed = {k for k, v in checked.items() if not _same(getattr(self.state, k), v)}
        if changed:
            target = dataclasses.replace(self.state, **checked)
            self.state = target
            self._run(functools.partial(self._apply, target, changed))
        return self

    @property
    def busy(self) -> bool:
        """An operation started on this device has not finished yet."""
        return self._last is not None and not self._last.done()

    def wait(self, timeout: float | None = None):
        """Block until all operations started on this device have finished."""
        last = self._last
        if last is not None:
            done, _ = concurrent.futures.wait([last], timeout)
            if not done:
                raise TimeoutError(f"{self.name} still busy after {timeout} s")
        self._raise_if_failed()

    def read(self):
        """Read the state from the hardware (waits for pending operations)."""
        self.wait()
        future = self._run(self._read)
        self.wait()
        self.state = future.result()
        return self.state

    def close(self):
        self._close_future().result()

    def describe(self) -> dict:
        """Everything about this device for metadata.json."""
        return {
            "driver": self.driver,
            "config": jsonable(self.config),
            "state": jsonable(self.state),
            "info": jsonable(self.info),
            "source": self.source,
        }

    def resource(self):
        """Hashable identity of the physical device; Setup refuses duplicates."""
        return None

    # ---- driver API -------------------------------------------------------

    async def _connect(self):
        pass

    async def _read(self):
        return self.State()

    async def _apply(self, state, changed: set[str]):
        pass

    async def _close(self):
        pass

    def _check_state(self, values: dict) -> dict:
        """Extra checks of (partial) state values, e.g. array shapes."""
        return values

    async def _blocking(self, fn, *args):
        """Run a blocking SDK call on this device's own thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._worker, functools.partial(fn, *args))

    # ---- internals --------------------------------------------------------

    def _raise_if_failed(self):
        if self._error is not None:
            error = self._error
            raise DeviceError(f"{self.name}: {type(error).__name__}: {error}") from error

    def _run(self, make_coroutine) -> concurrent.futures.Future:
        """Queue an operation after the previous one on this device."""
        previous = self._last

        async def step():
            if previous is not None:
                await asyncio.gather(asyncio.wrap_future(previous), return_exceptions=True)
            if self._error is not None:
                return None
            try:
                return await make_coroutine()
            except BaseException as error:
                self._error = error
                raise

        self._last = _submit(step())
        return self._last

    def _close_future(self) -> concurrent.futures.Future:
        previous, worker = self._last, self._worker

        async def closing():
            if previous is not None:
                await asyncio.gather(asyncio.wrap_future(previous), return_exceptions=True)
            try:
                await self._close()
            finally:
                if worker is not None:
                    worker.shutdown(wait=False)
                if self._worker is worker:
                    self._worker = None

        self.state = None
        self._last = None
        return _submit(closing())


class Camera(Device):
    """A device that captures images. ``capture()`` blocks and is never async."""

    last_capture_time: str | None = None

    def capture(self) -> np.ndarray:
        """Wait until the whole setup is ready, then grab one image."""
        if not self.connected:
            raise DeviceError(f"{self.name} is not connected")
        (self.setup if self.setup is not None else self).wait()
        self.last_capture_time = datetime.now().astimezone().isoformat()
        try:
            return self._worker.submit(self._grab).result()
        except Exception as error:
            raise DeviceError(f"{self.name}: capture failed: {error}") from error

    def _grab(self) -> np.ndarray:
        """Blocking; runs on the device thread."""
        raise NotImplementedError
