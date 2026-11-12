#!/usr/bin/env -S uv run
"""z-scan: one image per z position, green spot on the panel.

    scripts/measurements/z_scan.py [z_scan.toml] [-t SECONDS]

Images: <run>/<tile>/z/<index>.tiff (the z position is in each TIFF's metadata);
LED pattern: <run>/led_pattern/z/000.png
When the run ended without error, the whole panel is green for -t seconds (default 5).
"""

import argparse
from pathlib import Path

import numpy as np

from spec import Run, Setup, patterns
from spec.grid import move_to, tiles_from
from spec.run import add_finished_option, show_finished

parser = argparse.ArgumentParser(description="z-scan")
parser.add_argument("toml", nargs="?", default=Path(__file__).with_suffix(".toml"), type=Path)
add_finished_option(parser)
args = parser.parse_args()

setup = Setup.from_toml(args.toml)
p = setup.measurement
center = p.get("center_px") or setup.calibration["panel_camera_center"]["center_px"]
z_values = np.arange(p["z_start_mm"], p["z_stop_mm"] + p["z_step_mm"] / 2, p["z_step_mm"])

with setup:
    with Run(setup) as run:
        print(f"output: {run.path}")
        spot = patterns.ring(setup.panel.shape, center, p["radius_px"], color=p["color_rgb"])
        [led] = run.led_patterns("z", spot, f"spot r={p['radius_px']} px at {list(center)}")
        setup.panel.set(pattern=spot)
        for tile in tiles_from(p, setup, run):
            move_to(setup, tile)
            for i, z in enumerate(z_values):
                setup.stage_z.set(position_mm=float(z))
                image = run.capture(setup.camera, f"{tile.name}/z/{i:03d}.tiff", z_mm=float(z), led_pattern=led)
                print(f"{tile.name} z={z:.3f} mm: mean {image.mean():.1f}")
    show_finished(setup.panel, args.finished_s)
