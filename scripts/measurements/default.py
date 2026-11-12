#!/usr/bin/env -S uv run
"""Default measurement: a green spot on the panel, one image per tile.

    scripts/measurements/default.py [measurement.toml] [-t SECONDS]

Images: <run>/<tile>/default/<index>.tiff; LED pattern: <run>/led_pattern/default/000.png
When the run ended without error, the whole panel is green for -t seconds (default 5).
"""

import argparse
from pathlib import Path

from spec import Run, Setup, patterns
from spec.grid import move_to, tiles_from
from spec.run import add_finished_option, show_finished

parser = argparse.ArgumentParser(description="Default measurement")
parser.add_argument("toml", nargs="?", default=Path(__file__).with_suffix(".toml"), type=Path)
add_finished_option(parser)
args = parser.parse_args()

setup = Setup.from_toml(args.toml)
p = setup.measurement
center = p.get("center_px") or setup.calibration["panel_camera_center"]["center_px"]

with setup:
    with Run(setup) as run:
        print(f"output: {run.path}")
        spot = patterns.ring(setup.panel.shape, center, p["radius_px"], color=p["color_rgb"])
        [led] = run.led_patterns("default", spot, f"spot r={p['radius_px']} px at {list(center)}")
        setup.panel.set(pattern=spot)
        for tile in tiles_from(p, setup, run):
            move_to(setup, tile)
            for i in range(p.get("repeats", 1)):
                image = run.capture(setup.camera, f"{tile.name}/default/{i:03d}.tiff", led_pattern=led)
                print(f"{tile.name} {i:03d}: mean {image.mean():.1f}, max {image.max()}")
    show_finished(setup.panel, args.finished_s)
