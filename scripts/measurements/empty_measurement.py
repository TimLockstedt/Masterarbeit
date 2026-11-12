#!/usr/bin/env -S uv run
"""Empty measurement: the same position measured repeatedly (noise, calibration measurements).

    scripts/measurements/empty_measurement.py [empty_measurement.toml] [-o OUTPUT]
                                              [--coding canonical|hadamard] [-t SECONDS]

Without a sample or with a reference (e.g. an opal glass diffuser): to
measure the noise or as a calibration measurement. ``type`` in [measurement]:

- sli_calibration: every sweep of [[measurement.sweep]] (coding, segments,
  exposure_ms) on every tile. Loop order: sweep -> tile -> pattern.
- pli_calibration: the same with LCVR states instead of illumination patterns
  (not implemented yet).

Tiles: ``repeats = [nx, ny]``, a tiled measurement with zero movement, named
x<ix>_y<iy> like every tiled measurement (as on other microscopes). For now
stage_x/stage_y are not installed, so only camera and panel are connected
and the tiles have no position (``move_to`` does nothing). Once the x/y
stages are installed it should become a real x/y tiling with a step of 0 mm:
same folders, same naming, but with the stage positions recorded.

Run folder: <OUTPUT>/<timestamp>_<type>/ (OUTPUT: -o, else 'output' of the toml)
Images: <run>/x<ix>_y<iy>/<type>/<coding>_<segments>/<pattern index>.tiff
LED patterns: <run>/led_pattern/<type>_<coding>_<segments>/<pattern index>.png
When the run ended without error, the whole panel is green for -t seconds.
"""

import argparse
import time
from pathlib import Path

from spec import Run, Setup, patterns
from spec.grid import Tile, move_to
from spec.run import add_finished_option, show_finished

TYPES = ["sli_calibration", "pli_calibration"]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
parser.add_argument("toml", nargs="?", default=Path(__file__).with_suffix(".toml"), type=Path)
parser.add_argument("-o", "--output", type=Path, help="parent folder for the run (overrides the toml)")
parser.add_argument("--coding", choices=list(patterns.CODINGS), help="sli_calibration: only this coding")
add_finished_option(parser)
args = parser.parse_args()

setup = Setup.from_toml(args.toml)
p = setup.measurement
kind = p.get("type")
if kind not in TYPES:
    raise SystemExit(f"{args.toml}: [measurement] type must be one of {', '.join(TYPES)}, got {kind!r}")
if kind == "pli_calibration":
    raise SystemExit("pli_calibration (LCVR states instead of patterns) is not implemented yet")

sweeps = [s for s in p["sweep"] if args.coding in (None, s["coding"])]
if not sweeps:
    raise SystemExit(f"{args.toml}: no sweep with coding {args.coding!r}")
for sweep in sweeps:
    if sweep["coding"] not in patterns.CODINGS:
        raise SystemExit(f"unknown coding {sweep['coding']!r}; use {', '.join(patterns.CODINGS)}")
nx, ny = p["repeats"]
tiles = [Tile(ix, iy, None, None) for iy in range(ny) for ix in range(nx)]  # zero movement, see above
total_images = len(tiles) * sum(s["segments"] for s in sweeps)
exposure_s = len(tiles) * sum(s["segments"] * s["exposure_ms"] for s in sweeps) / 1000
setup.measurement = {**p, "sweep": sweeps, "selected_coding": args.coding}  # recorded in metadata
print(f"{kind}: {len(sweeps)} sweeps x {len(tiles)} tiles: {total_images} images, "
      f"exposure alone {exposure_s / 3600:.1f} h")

with setup:
    with Run(setup, args.output, kind) as run:
        print(f"output: {run.path}")
        center = p.get("center_px") or setup.calibration["panel_camera_center"]["center_px"]
        done, start = 0, time.monotonic()
        for sweep in sweeps:
            coding, n, exposure = sweep["coding"], sweep["segments"], sweep["exposure_ms"]
            name = f"{coding}_{n}"
            images = patterns.coded(
                coding,
                setup.panel.shape,
                center,
                p["outer_radius_px"],
                p["inner_radius_px"],
                n,
                color=p["color_rgb"],
                angle_offset_deg=p.get("angle_offset_deg", 0),
            )
            leds = run.led_patterns(
                f"{kind}_{name}", images, f"{n} segments ({coding}), "
                f"r={p['inner_radius_px']}..{p['outer_radius_px']} px at {list(center)}"
            )
            setup.camera.set(exposure_ms=exposure)
            print(f"sweep {name}: {n} patterns, exposure {exposure:g} ms")
            for tile in tiles:
                move_to(setup, tile)
                for i, pattern in enumerate(images):
                    setup.panel.set(pattern=pattern)
                    image = run.capture(
                        setup.camera, f"{tile.name}/{kind}/{name}/{i:03d}.tiff",
                        type=kind, sweep=name, coding=coding, tile=tile.name, pattern=i,
                        led_pattern=leds[i],
                    )
                    done += 1
                    elapsed = time.monotonic() - start
                    left = elapsed / done * (total_images - done)
                    print(f"{done}/{total_images} {tile.name} {name} {i:03d}: mean {image.mean():.1f}"
                          f"   ~{left / 60:.0f} min left")
    show_finished(setup.panel, args.finished_s)
