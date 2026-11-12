"""One measurement output folder: metadata.json, TIFF images and the LED patterns (PNG)."""

import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from . import __version__
from .device import Camera, jsonable

FINISHED_COLOR = (0, 255, 0)
FINISHED_S = 5

PACKAGES = [
    "numpy",
    "tifffile",
    "pypylon",
    "zaber-motion",
    "KLCserial",
    "rspanel",
    "opencv-python-headless",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _versions() -> dict:
    result = {"python": sys.version, "spec": __version__}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _git(folder: Path) -> dict | None:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(folder), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def _script() -> dict | None:
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path or not Path(path).is_file():
        return None
    path = Path(path).resolve()
    return {"path": str(path), "argv": sys.argv, "source": path.read_text(), "git": _git(path.parent)}


class Run:
    """Create ``<output>/<timestamp>_<name>/`` and write ``metadata.json``.

    Create it after the setup is connected (``with setup, Run(setup) as run:``),
    so the metadata contains the initial device states.
    """

    def __init__(self, setup, output=None, name=None):
        if not all(device.connected for device in setup):
            raise RuntimeError("connect the setup before creating a Run")
        output = output or setup.source.get("output")
        if output is None:
            raise ValueError("no output folder: pass output= or set 'output' in the toml")
        name = name or setup.source.get("name") or "run"
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.setup = setup
        self.path = Path(output) / f"{stamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.images = 0
        self.metadata = {
            "started": _now(),
            "finished": None,
            "status": "running",
            "images": 0,
            "measurement": setup.source,
            "parameters": jsonable(setup.measurement),
            "script": _script(),
            "versions": _versions(),
            "devices": setup.describe(),
        }
        self._write_metadata()

    def record(self, key: str, value):
        """Add ``key`` to metadata.json (e.g. a drawn ROI, chosen parameters)."""
        if key in self.metadata:
            raise KeyError(f"metadata already has {key!r}")
        self.metadata[key] = jsonable(value)
        self._write_metadata()

    def led_patterns(self, measurement_type: str, patterns, description: str = "") -> list[str]:
        """Save the panel patterns of one measurement type as PNGs.

        ``patterns``: one (H, W, 3) uint8 pattern or a stack (N, H, W, 3). Files:
        ``led_pattern/<measurement_type>/000.png``, ...; metadata.json gets
        ``led_patterns[<measurement_type>] = {"folder", "files", "shape", "description"}``.
        Returns the paths relative to the run folder, in order: pass them on with
        ``capture(..., led_pattern=path)`` so every image names its pattern.
        """
        name = str(measurement_type)
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"measurement type {name!r} must be a plain folder name")
        known = self.metadata.setdefault("led_patterns", {})
        if name in known:
            raise KeyError(f"LED patterns for {name!r} are already saved")
        stack = np.asarray(patterns)
        if stack.ndim == 3:
            stack = stack[np.newaxis]
        if stack.ndim != 4 or stack.shape[-1] != 3 or stack.dtype != np.uint8:
            raise ValueError(f"patterns must be uint8 (H, W, 3) or (N, H, W, 3), got {stack.dtype} {stack.shape}")
        from PIL import Image

        folder = self.path / "led_pattern" / name
        folder.mkdir(parents=True, exist_ok=False)
        files = []
        for index, pattern in enumerate(stack):
            file = folder / f"{index:03d}.png"
            with open(file, "xb") as handle:  # never overwrite
                Image.fromarray(pattern, "RGB").save(handle, format="PNG")
            files.append(file.name)
        known[name] = {
            "folder": f"led_pattern/{name}",
            "files": files,
            "shape": list(stack.shape[1:]),
            "description": description,
        }
        self._write_metadata()
        return [f"led_pattern/{name}/{file}" for file in files]

    def _write_metadata(self):
        self.metadata["images"] = self.images
        text = json.dumps(self.metadata, indent=2, default=str)
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(text)
        temporary.replace(self.path / "metadata.json")

    def _target(self, relative) -> Path:
        target = (self.path / relative).resolve()
        if not target.is_relative_to(self.path.resolve()):
            raise ValueError(f"{relative} points outside the run folder")
        if target.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"{relative}: images are saved as .tif/.tiff")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save(self, image: np.ndarray, path, **metadata) -> Path:
        """Save ``image`` to ``path`` (relative to the run folder); never overwrites."""
        target = self._target(path)
        description = {
            "measurement_time": metadata.pop("measurement_time", _now()),
            **jsonable(metadata),
            "devices": {
                name: jsonable(state) for name, state in self.setup.states().items()
            },
        }
        with open(target, "xb") as file:  # "x": fail if the file exists
            tifffile.imwrite(
                file, np.asarray(image), description=json.dumps(description), metadata=None
            )
        self.images += 1
        return target

    def capture(self, camera: Camera, path, **metadata) -> np.ndarray:
        """Capture with ``camera`` (waits for the setup) and save; returns the image."""
        self._target(path)  # check the path before taking the image
        image = camera.capture()
        self.save(
            image,
            path,
            measurement_time=camera.last_capture_time,
            camera=camera.name,
            **metadata,
        )
        return image

    def close(self, error: BaseException | None = None):
        self.metadata["finished"] = _now()
        if error is None:
            self.metadata["status"] = "completed"
        else:
            self.metadata["status"] = "failed"
            self.metadata["error"] = "".join(traceback.format_exception(error))
        self._write_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc)
        return False


def add_finished_option(parser):
    """``-t/--finished-s``: seconds of ``show_finished`` (measurement scripts)."""
    parser.add_argument("-t", "--finished-s", type=float, default=FINISHED_S,
                        help=f"whole panel green for this long when finished without error "
                             f"(default {FINISHED_S:g}, 0: off)")


def show_finished(panel, seconds: float, color=FINISHED_COLOR):
    """Light the whole panel in ``color`` for ``seconds`` (0: nothing).

    Call it after the ``with Run(...)`` block, so it only happens when the run
    ended without error; disconnecting the panel then turns it black."""
    if seconds <= 0:
        return
    panel.set(pattern=np.broadcast_to(np.asarray(color, np.uint8), panel.shape).copy())
    panel.wait()
    print(f"finished: panel {list(color)} for {seconds:g} s")
    time.sleep(seconds)
