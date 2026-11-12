#!/usr/bin/env -S uv run
"""Illumination: show light patterns on the panel, with an optional live camera window.

A helper tool, not a measurement: to see the setup, to look through the camera
(also with pylon), to try patterns, and to save an image or a pattern sequence
on the spot.

    scripts/illumination.py                      # window: live camera + pattern controls
    scripts/illumination.py --no-camera          # window without camera (pylon can use it)
    scripts/illumination.py full white           # no window: constant pattern until Ctrl-C
    scripts/illumination.py spot red -r 20
    scripts/illumination.py hadamard 3 --segments 32
    scripts/illumination.py hadamard --fps 5     # no window: the sequence in a loop
    scripts/illumination.py file --file PATH     # an image file, or a folder as sequence

Pattern types: full (whole panel), spot (disc, -r), ring (--inner..--outer),
canonical / hadamard (the SLI codings of the ring, --segments), file (--file).
After the type: a pattern index and/or a colour (green, white, red, blue or
R,G,B), in any order. Defaults: green, centre from
config/<instrument>/calibration/panel_camera_center.toml.

The panel switches off after --max-s seconds (default 600, 0: never); in the
window the time restarts with every change. Camera settings: the instrument's
camera.toml; change them in the window.

Window: type, index, geometry, colour, brightness and fps on the right; the
overview camera's image instead of the histogram on request (--overview, --no-overview);
Left/Right: index, Shift+arrows: move the centre by 1 panel px, Up/Down: move
the z stage by the step (or type z in its field; --no-stage-z: none), Space: start/stop
the sequence at the fps (preloaded on the panel; the live image is not
synchronised with it), "Save image" (s): the current image, "Save sequence":
like a measurement, every pattern of the current type, one image each.
Saved to <OUTPUT>/<timestamp>_illumination/ (one run folder per session, created
at the first save) with metadata.json and the LED patterns:
x00_y00/image/NNN.tiff and x00_y00/sequence_NNN/<pattern index>.tiff.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from spec import Run, Setup, patterns
from spec.grid import HERE
from spec.ui.gui import add_overview_option

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["full", "spot", "ring", "canonical", "hadamard", "file"]
COLORS = {"green": (0, 255, 0), "white": (255, 255, 255), "red": (255, 0, 0), "blue": (0, 0, 255)}
MAX_INDEX = 4095  # range of the index field; the real limit is the number of patterns


def color_value(text):
    if text in COLORS:
        return COLORS[text]
    rgb = tuple(int(v) for v in text.split(",")) if text.replace(",", "").isdigit() else ()
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(f"{text!r}: use {', '.join(COLORS)} or R,G,B (0..255)")
    return rgb


def color_name(rgb):
    return next((name for name, value in COLORS.items() if value == tuple(rgb)), ",".join(map(str, rgb)))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("kind", nargs="?", choices=KINDS, help="pattern type; without it: the window")
parser.add_argument("extra", nargs="*", metavar="INDEX|COLOR", help="pattern index and/or colour")
parser.add_argument("-b", "--brightness", type=int, default=255, help="0..255, scales the colour (default 255)")
parser.add_argument("-r", "--radius", type=float, default=15, help="spot radius in panel px (default 15)")
parser.add_argument("--inner", type=float, default=50, help="ring inner radius in panel px (default 50)")
parser.add_argument("--outer", type=float, default=90, help="ring outer radius in panel px (default 90)")
parser.add_argument("--segments", type=int, default=32, help="canonical/hadamard segments (default 32)")
parser.add_argument("--angle-offset", type=float, default=90, help="segment 0 centre in deg (default 90)")
parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                    help="centre in panel px (default: calibration/panel_camera_center.toml)")
parser.add_argument("--file", type=Path, help="type file: an image, or a folder of images (a sequence)")
parser.add_argument("--fps", type=float, help="play the sequence at this frame rate (window: the start value)")
parser.add_argument("--max-s", type=float, default=600,
                    help="panel off after this many seconds (default 600, 0: never)")
parser.add_argument("--gui", action="store_true", help="the window also when a type is given")
parser.add_argument("--no-camera", action="store_true", help="the window without camera")
parser.add_argument("--camera", default="camera", help="camera device (default: camera)")
parser.add_argument("-o", "--output", type=Path, default=ROOT / "results", help="parent folder for saved runs")
parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
parser.add_argument("--stage-z", default="stage_z", metavar="STAGE",
                    help="window: the z stage to move by hand (default: stage_z; never moved on its own)")
parser.add_argument("--no-stage-z", dest="stage_z", action="store_const", const=None, help="window: no z stage")
add_overview_option(parser)
args = parser.parse_args()

look = {  # what the panel shows; the window changes it
    "kind": args.kind or "spot", "index": 0, "color": COLORS["green"], "brightness": args.brightness,
    "radius": args.radius, "inner": args.inner, "outer": args.outer, "segments": args.segments,
    "angle_offset": args.angle_offset, "center": None,
}
try:
    for token in args.extra:
        if token.isdigit():
            look["index"] = int(token)
        else:
            look["color"] = color_value(token)
except ValueError as error:
    parser.error(str(error))
if not 0 <= args.brightness <= 255:
    parser.error("--brightness must be 0..255")
if args.kind is not None and (args.kind == "file") != (args.file is not None):
    parser.error("type file needs --file, and --file needs type file")
if args.fps is not None and args.fps <= 0:
    parser.error("--fps must be > 0")
if args.max_s < 0:
    parser.error("--max-s must be >= 0 (0: never)")
use_window = args.kind is None or args.gui or args.no_camera
use_camera = use_window and not args.no_camera
kinds = KINDS if args.file else KINDS[:-1]

overview = args.overview if use_camera and args.overview != args.camera else None
stage_z = args.stage_z if use_window else None
setup = Setup.from_folder(args.instrument, "panel", *[name for name in (args.camera, overview)
                                                      if use_camera and name], *filter(None, [stage_z]))
shape = setup.panel.shape
calibrated = setup.calibration.get("panel_camera_center", {}).get("center_px")
look["center"] = list(args.center or calibrated or (shape[1] // 2, shape[0] // 2))
if args.file is None:
    files = None
elif args.file.is_dir():
    files = patterns.load_folder(args.file, shape)
else:
    files = patterns.load_image(args.file, shape)[None]


# ---- patterns ----------------------------------------------------------------------

_cache = {}


def stack() -> np.ndarray:
    """All patterns of the current type (N, H, W, 3); one for full, spot and ring."""
    key = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in look.items() if k != "index")
    if key not in _cache:
        _cache.clear()
        rgb = np.round(np.asarray(look["color"]) * look["brightness"] / 255).astype(np.uint8)
        kind, center = look["kind"], look["center"]
        if kind == "full":
            result = np.broadcast_to(rgb, shape)[None].copy()
        elif kind == "spot":
            result = patterns.ring(shape, center, look["radius"], color=rgb)[None]
        elif kind == "ring":
            result = patterns.ring(shape, center, look["outer"], look["inner"], color=rgb)[None]
        elif kind in patterns.CODINGS:
            result = patterns.coded(kind, shape, center, look["outer"], look["inner"], look["segments"],
                                    color=rgb, angle_offset_deg=look["angle_offset"])
        else:  # file: brightness scales the images, the colour is not used
            result = np.round(files * (look["brightness"] / 255)).astype(np.uint8)
        _cache[key] = result
    return _cache[key]


def describe() -> str:
    kind, n, c = look["kind"], len(stack()), look["center"]
    at = f"at [{c[0]:g}, {c[1]:g}]"
    text = {
        "full": "full panel",
        "spot": f"spot r={look['radius']:g} px {at}",
        "ring": f"ring r={look['inner']:g}..{look['outer']:g} px {at}",
        "file": f"file {args.file}",
    }.get(kind, f"{kind} {n} segments, r={look['inner']:g}..{look['outer']:g} px {at}")
    if n > 1:
        text = f"{text}, pattern {look['index']}/{n - 1}"
    light = "" if kind == "file" else f"{color_name(look['color'])}, "
    return f"{text} ({light}brightness {look['brightness']})"


def parameters() -> dict:
    """The look for metadata: only what the current type uses."""
    used = {"full": [], "spot": ["radius", "center"], "ring": ["inner", "outer", "center"],
            "file": []}.get(look["kind"], ["inner", "outer", "segments", "angle_offset", "center"])
    extra = {"file": str(args.file)} if look["kind"] == "file" else {"color": list(look["color"])}
    return {"kind": look["kind"], "index": look["index"], "brightness": look["brightness"],
            **extra, **{k: look[k] for k in used}}


def show():
    """Show the current pattern (the index is checked against the type's patterns)."""
    all_ = stack()
    if not 0 <= look["index"] < len(all_):
        raise ValueError(f"{look['kind']} has patterns 0..{len(all_) - 1}, not {look['index']}")
    setup.panel.set(pattern=all_[look["index"]])


# ---- saving ------------------------------------------------------------------------

saved = {"run": None, "images": 0, "sequences": 0}


def run_folder() -> Run:
    if saved["run"] is None:
        saved["run"] = Run(setup, args.output, "illumination")
        print(f"output: {saved['run'].path}")
    return saved["run"]


def save_image(camera):
    run, k = run_folder(), saved["images"]
    [led] = run.led_patterns(f"image_{k:03d}", stack()[look["index"]], describe())
    path = f"{HERE.name}/image/{k:03d}.tiff"
    image = run.capture(camera, path, tile=HERE.name, illumination=parameters(), led_pattern=led)
    saved["images"] += 1
    return image, f"saved {run.path / path}: {describe()}"


def save_sequence(camera, on_image):
    """Like a measurement: every pattern of the current type, one image each."""
    run, k, all_ = run_folder(), saved["sequences"], stack()
    name = f"sequence_{k:03d}"
    leds = run.led_patterns(name, all_, describe())
    saved["sequences"] += 1
    setup.panel.preload(all_)
    shown = look["index"]
    try:
        for i, pattern in enumerate(all_):
            look["index"] = i
            setup.panel.set(pattern=pattern)
            image = run.capture(camera, f"{HERE.name}/{name}/{i:03d}.tiff", tile=HERE.name,
                                illumination=parameters(), led_pattern=leds[i])
            on_image(image, f"{name}: {i + 1}/{len(all_)}")
    finally:
        look["index"] = shown
        show()
    return f"saved {run.path / HERE.name / name}/ ({len(all_)} images): {describe()}"


# ---- sequence playback -------------------------------------------------------------

class Player:
    """Plays the current type's patterns at ``fps`` (preloaded on the panel). A pattern
    that is due while the panel is still busy is shown as soon as it is free (late, not
    skipped): above what the panel can do, it plays as fast as it can."""

    def __init__(self, fps):
        self.fps, self.playing = fps, False
        self.shown = 0
        self.started = self.next = 0.0

    def start(self):
        if len(stack()) < 2:
            raise ValueError(f"{look['kind']} has only one pattern: nothing to play")
        setup.panel.preload(stack())
        self.playing, self.shown = True, 0
        self.started = self.next = time.monotonic()

    def stop(self):
        self.playing = False

    def step(self) -> float:
        """Show the next pattern if it is due; returns the seconds until the next one."""
        now = time.monotonic()
        if now < self.next:
            return self.next - now
        if setup.panel.busy:
            return 0.002  # due, but the panel is not free yet
        look["index"] = (look["index"] + 1) % len(stack())
        setup.panel.set(pattern=stack()[look["index"]])
        self.shown += 1
        self.next = max(self.next + 1 / self.fps, now)  # late: no catching up
        return self.next - now

    def rate(self) -> str:
        elapsed = time.monotonic() - self.started
        actual = f"{self.shown / elapsed:.1f}" if elapsed > 1 else "..."
        return f"sequence {self.fps:g} fps (actual {actual})"


# ---- command line ------------------------------------------------------------------

def command_line():
    player = Player(args.fps) if args.fps else None
    try:
        show()
        if player:
            player.start()
    except ValueError as error:
        raise SystemExit(f"error: {error}") from None
    print(f"panel: {describe()}" + (f", {args.fps:g} fps" if player else ""))
    print("on until Ctrl-C" + (f", at most {args.max_s:g} s" if args.max_s else ""))
    deadline = time.monotonic() + args.max_s if args.max_s else float("inf")
    try:
        while (left := deadline - time.monotonic()) > 0:
            time.sleep(min(player.step() if player else 0.5, 0.5, left))
        print(f"{args.max_s:g} s reached: off")
    except KeyboardInterrupt:
        print("stopped")
    if player:
        print(player.rate())


# ---- window ------------------------------------------------------------------------

Z_STEPS_MM = ["0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1"]
Z_STEP_MM = "0.01"


def window_tool(camera, overview=None, stage=None):
    from spec.ui.gui import Window

    player = Player(args.fps or 5)
    timer = {"deadline": None, "off": False}
    fields = {}  # look key -> StringVar of its field
    show_gain = None

    def touch():  # every change: the light is on again, the time limit restarts
        timer["deadline"] = time.monotonic() + args.max_s if args.max_s else None
        timer["off"] = False

    def follow():  # the fields show the current look (also after keys, errors, playback)
        for key, variable in fields.items():
            value = look[key[:-2]][int(key[-1])] if key.startswith("center") else look[key]
            text = color_name(value) if key == "color" else value if isinstance(value, str) else f"{value:g}"
            if variable.get() != text:
                variable.set(text)

    def change(**values):
        old = dict(look)
        look.update(values)
        try:
            if player.playing and set(values) != {"index"}:
                player.stop()
            show()
        except Exception:
            look.update(old)
            raise
        finally:
            follow()
        touch()

    def toggle():
        if player.playing:
            player.stop()
        else:
            player.start()
            touch()

    with Window("illumination", camera, overview=overview) as window:
        def with_still(action):  # capture here, not in the live thread
            was_live = window.view is not None and window.view.live
            if window.view is not None:
                window.view.stop()
            player.stop()
            try:
                return action()
            finally:
                if was_live and not window.closed:
                    window.view.resume()

        def save_one():
            frame, text = with_still(lambda: save_image(camera))
            window.show(frame, text)
            print(text)

        def save_all():
            text = with_still(lambda: save_sequence(camera, window.show))
            window.text(text)
            print(text)

        def tick():
            if window.closed:
                return
            if timer["deadline"] and time.monotonic() > timer["deadline"] and not timer["off"]:
                player.stop()
                setup.panel.off()
                timer["off"] = True
                window.text(f"{args.max_s:g} s without a change: panel off (change anything to switch it on)")
            delay = player.step() if player.playing else 0.1
            if player.playing:
                follow()
            window.root.after(max(1, int(min(delay, 0.1) * 1000)), tick)

        def status():
            parts = [describe()]
            if player.playing:
                parts.append(player.rate())
            if timer["off"]:
                parts.append("panel OFF")
            elif timer["deadline"]:
                parts.append(f"off in {max(timer['deadline'] - time.monotonic(), 0):.0f} s")
            if show_gain is not None:
                show_gain(camera.state.gain)  # follows Auto exposure
                parts.append(f"gain {camera.state.gain:.1f} dB")
            if stage is not None:
                if stage.busy:
                    parts.append(f"z -> {stage.state.position_mm:.4f} mm (moving)")
                else:
                    z = stage.read().position_mm  # also follows moves made with the controller
                    if window.root.focus_get() is not z_field.widget:  # not while typing
                        z_field.set(f"{z:g}")
                    parts.append(f"z {z:.4f} mm")
            return "   ".join(parts)

        def move_z(direction):
            low, high = stage.limits["position_mm"]
            z = stage.state.position_mm + direction * float(z_step.get())
            stage.set(position_mm=float(np.clip(z, low, high)))
            z_field.set(f"{stage.state.position_mm:g}")

        def number(parent, label, key, low, high, step=1, integer=False, **reset):
            fields[key] = window.number(label, look[key], lambda v: change(**{key: v}, **reset), low, high,
                                        step=step, integer=integer, parent=parent)

        box = window.section("pattern")
        fields["kind"] = window.choice("type", kinds, look["kind"], lambda kind: change(kind=kind, index=0),
                                       parent=box)
        number(box, "index", "index", 0, MAX_INDEX, integer=True)
        box = window.section("geometry (panel px)")
        number(box, "spot radius", "radius", 0.5, 1000)
        number(box, "ring inner", "inner", 0, 1000)
        number(box, "ring outer", "outer", 0.5, 1000)
        number(box, "segments", "segments", 1, 1024, integer=True, index=0)
        number(box, "angle offset", "angle_offset", -360, 360, step=5)
        for axis, name in enumerate("xy"):
            fields[f"center_{axis}"] = window.number(
                f"centre {name}", look["center"][axis],
                lambda v, axis=axis: change(center=[v if i == axis else c for i, c in enumerate(look["center"])]),
                -1000, 1000, integer=False, parent=box)
        box = window.section("light")
        fields["color"] = window.choice("colour", list(COLORS), color_name(look["color"]),
                                        lambda name: change(color=COLORS[name]), parent=box)
        number(box, "brightness", "brightness", 0, 255, step=5, integer=True)
        box = window.section("sequence")
        window.number("fps", player.fps, lambda v: setattr(player, "fps", v), 0.1, 100, integer=False, parent=box)
        if stage is not None:  # moved only by hand: field (Enter, arrows) and Up/Down
            box = window.section(f"{stage.name} (mm)")
            low, high = stage.limits["position_mm"]
            z_field = window.number("z", stage.state.position_mm, lambda z: stage.set(position_mm=z), low, high,
                                    step=float(Z_STEP_MM), integer=False, parent=box)
            z_step = window.choice("step", Z_STEPS_MM, Z_STEP_MM,
                                   lambda step: z_field.widget.config(increment=float(step)), parent=box)
            window.key("<Up>", lambda: move_z(+1))
            window.key("<Down>", lambda: move_z(-1))

        window.button("Previous", lambda: change(index=(look["index"] - 1) % len(stack())), key="<Left>")
        window.button("Next", lambda: change(index=(look["index"] + 1) % len(stack())), key="<Right>")
        window.button("Sequence start/stop (Space)", toggle, key="<space>")
        for key, dx, dy in (("<Shift-Left>", -1, 0), ("<Shift-Right>", 1, 0),
                            ("<Shift-Up>", 0, -1), ("<Shift-Down>", 0, 1)):
            window.key(key, lambda dx=dx, dy=dy: change(center=[look["center"][0] + dx, look["center"][1] + dy]))
        if camera is not None:
            window.button("Save image (s)", save_one, key="s")
            window.button("Save sequence", save_all)
            if "gain" in camera.limits:
                low, high = camera.limits["gain"]
                show_gain = window.slider("gain dB", low, high, camera.state.gain, lambda g: camera.set(gain=g))
        window.status(status)
        z_keys = ", Up/Down: z" if stage is not None else ""
        window.text(f"Left/Right: index, Shift+arrows: centre, Space: sequence{z_keys}; close the window to end")
        show()
        touch()
        window.root.after(100, tick)
        window.wait(None)
    player.stop()


with setup:
    error = None
    try:
        if use_window:
            window_tool(setup[args.camera] if use_camera else None, setup[overview] if overview else None,
                        setup[stage_z] if stage_z else None)
        else:
            command_line()
    except BaseException as exception:
        error = exception
        raise
    finally:
        if saved["run"] is not None:
            saved["run"].close(error)  # metadata.json: completed or failed
        setup.panel.off()
