"""A group of devices used together by one measurement."""

import concurrent.futures
import tomllib
from pathlib import Path

from .device import Device, DeviceError

MEASUREMENT_KEYS = {"setup", "output", "devices", "measurement"}


def _load(folder: Path, names, config: dict, defaults: dict) -> list[Device]:
    if not folder.is_dir():
        raise FileNotFoundError(f"instrument folder {folder} does not exist")
    unknown = (set(config) | set(defaults)) - set(names)
    if unknown:
        raise ValueError(f"overrides for devices not in the setup: {', '.join(sorted(unknown))}")
    devices = []
    for name in names:
        file = folder / f"{name}.toml"
        if not file.is_file():
            available = sorted(p.stem for p in folder.glob("*.toml"))
            raise FileNotFoundError(
                f"no device toml for {name!r} in {folder}; available: {', '.join(available)}"
            )
        devices.append(
            Device.from_toml(
                file, name, config=config.get(name, {}), defaults=defaults.get(name, {})
            )
        )
    return devices


def _calibration(folder: Path) -> dict[str, dict]:
    """``<instrument>/calibration/<name>.toml`` -> {name: values}."""
    result = {}
    for path in sorted((folder / "calibration").glob("*.toml")):
        text = path.read_text()
        values = tomllib.loads(text)
        values["_source"] = {"path": str(path), "toml": text}
        result[path.stem] = values
    return result


def _raise(errors: list[BaseException], message: str):
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(message, errors)


class Setup:
    """``Setup(dev1, dev2, ...)`` or ``Setup.from_toml("measurement.toml")``.

    Use as a context manager: entering connects every device in parallel and
    applies its start state; the block runs only if all of that succeeded.
    Leaving closes every device, also after errors.
    """

    def __init__(self, *devices: Device, measurement: dict | None = None, source: dict | None = None):
        if not devices:
            raise ValueError("a setup needs at least one device")
        by_name: dict[str, Device] = {}
        resources: dict = {}
        for device in devices:
            if not isinstance(device, Device):
                raise TypeError(f"not a device: {device!r}")
            if device.name in by_name:
                raise ValueError(f"two devices are named {device.name!r}")
            if (
                hasattr(Setup, device.name)
                or device.name in ("measurement", "source", "calibration")
                or device.name.startswith("_")
            ):
                raise ValueError(f"device name {device.name!r} is reserved")
            if device.setup is not None:
                raise ValueError(f"{device.name} already belongs to another setup")
            resource = device.resource()
            if resource is not None:
                if resource in resources:
                    raise ValueError(
                        f"{device.name} and {resources[resource]} are the same "
                        f"physical device {resource}"
                    )
                resources[resource] = device.name
            by_name[device.name] = device
        for device in devices:
            device.setup = self
        self._devices = by_name
        self.measurement = measurement or {}  # [measurement] table of the toml
        self.source = source or {}  # measurement toml path/text, output folder
        folder = self.source.get("setup")
        self.calibration = _calibration(Path(folder)) if folder else {}
        self.source["calibration"] = {
            name: values.pop("_source") for name, values in self.calibration.items()
        }

    @classmethod
    def from_toml(cls, path):
        """Create the devices listed under ``[devices]`` of a measurement toml."""
        path = Path(path)
        text = path.read_text()
        data = tomllib.loads(text)
        unknown = set(data) - MEASUREMENT_KEYS
        if unknown:
            raise ValueError(
                f"{path}: unknown key(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(MEASUREMENT_KEYS))}"
            )
        if "setup" not in data:
            raise ValueError(f"{path}: missing 'setup' (folder with the device tomls)")
        folder = (path.parent / data["setup"]).resolve()
        tables = {name: dict(values) for name, values in data.get("devices", {}).items()}
        if not tables:
            raise ValueError(f"{path}: no devices listed under [devices]")
        try:
            devices = _load(
                folder,
                list(tables),
                config={name: t.pop("config", {}) for name, t in tables.items()},
                defaults=tables,
            )
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise type(error)(f"{path}: {error}") from None
        output = data.get("output")
        source = {
            "path": str(path.resolve()),
            "name": path.stem,
            "toml": text,
            "setup": str(folder),
            "output": str((path.parent / output).resolve()) if output else None,
        }
        return cls(*devices, measurement=data.get("measurement", {}), source=source)

    @classmethod
    def from_folder(cls, folder, *names, config: dict | None = None, defaults: dict | None = None):
        """Devices ``names`` from an instrument folder, e.g. for calibration scripts.

        ``config`` / ``defaults``: {device name: overrides}."""
        folder = Path(folder).resolve()
        devices = _load(folder, names, config or {}, defaults or {})
        return cls(*devices, source={"setup": str(folder), "name": None, "output": None})

    # ---- access -------------------------------------------------------------

    def __getattr__(self, name):
        devices = self.__dict__.get("_devices", {})
        if name in devices:
            return devices[name]
        raise AttributeError(
            f"setup has no device {name!r}; devices: {', '.join(devices) or 'none'}"
        )

    def __getitem__(self, name) -> Device:
        return self._devices[name]

    def __iter__(self):
        return iter(self._devices.values())

    def __len__(self):
        return len(self._devices)

    def __contains__(self, name):
        return name in self._devices

    def __repr__(self):
        return f"<Setup {', '.join(self._devices)}>"

    @property
    def names(self) -> list[str]:
        return list(self._devices)

    # ---- operations ---------------------------------------------------------

    def connect(self):
        """Connect all devices in parallel; on any failure close all and raise."""
        for device in self:
            device.connect(wait=False)
        errors = self._wait_all()
        if errors:
            self.close(errors)
            _raise(errors, "connecting the setup failed")
        return self

    def wait(self, timeout: float | None = None):
        """Block until every device is ready; raise device errors."""
        _raise(self._wait_all(timeout), "devices failed")

    def _wait_all(self, timeout=None) -> list[BaseException]:
        futures = [d._last for d in self if d._last is not None]
        done, pending = concurrent.futures.wait(futures, timeout)
        if pending:
            busy = [d.name for d in self if d._last in pending]
            raise TimeoutError(f"still busy after {timeout} s: {', '.join(busy)}")
        errors = []
        for device in self:
            try:
                device._raise_if_failed()
            except DeviceError as error:
                errors.append(error)
        return errors

    def close(self, errors: list | None = None):
        """Close all devices in parallel."""
        futures = {d.name: d._close_future() for d in self}
        close_errors = []
        for name, future in futures.items():
            try:
                future.result()
            except Exception as error:
                close_errors.append(DeviceError(f"{name}: close failed: {error}"))
        if errors is not None:
            for error in close_errors:
                errors[0].add_note(str(error))
            return
        _raise(close_errors, "closing the setup failed")

    def describe(self) -> dict:
        return {d.name: d.describe() for d in self}

    def states(self) -> dict:
        return {d.name: d.state for d in self}

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            try:
                self.wait()
            except BaseException as error:
                self.close([error])
                raise
            self.close()
        else:
            self.close([exc])
        return False
