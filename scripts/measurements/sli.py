#!/usr/bin/env -S uv run
"""SLI: segmented ring illumination, one image per pattern and tile.

    scripts/measurements/sli.py [sli.toml] [-o OUTPUT] [-t SECONDS]

The run folder is <OUTPUT>/<timestamp>_sli/ (OUTPUT defaults to 'output' in the toml).
Images: <run>/<tile>/sli/<pattern index>.tiff; LED patterns: <run>/led_pattern/sli/<pattern index>.png
When the run ended without error, the whole panel is green for -t seconds (default 5).
"""

import argparse
from pathlib import Path

from spec import Run, Setup, patterns
from spec.grid import move_to, tiles_from
from spec.run import add_finished_option, show_finished

parser = argparse.ArgumentParser(description="SLI: segmented ring illumination")
parser.add_argument("toml", nargs="?", default=Path(__file__).with_suffix(".toml"), type=Path)
parser.add_argument("-o", "--output", type=Path, help="parent folder for the run (overrides the toml)")
add_finished_option(parser)
args = parser.parse_args()

setup = Setup.from_toml(args.toml)
p = setup.measurement
center = p.get("center_px") or setup.calibration["panel_camera_center"]["center_px"]
coding = p.setdefault("coding", "canonical")  # setdefault: always in metadata.json "parameters"
if coding not in patterns.CODINGS:
    raise SystemExit(f"{args.toml}: unknown coding {coding!r}; use {', '.join(patterns.CODINGS)}")

with setup:
    with Run(setup, output=args.output) as run:
        print(f"output: {run.path}")
        images = patterns.coded(
            coding,
            setup.panel.shape,
            center,
            p["outer_radius_px"],
            p["inner_radius_px"],
            p["segments"],
            color=p["color_rgb"],
            angle_offset_deg=p.get("angle_offset_deg", 0),
        )
        leds = run.led_patterns(
            "sli", images, f"{p['segments']} segments ({coding}), "
            f"r={p['inner_radius_px']}..{p['outer_radius_px']} px at {list(center)}"
        )
        for tile in tiles_from(p, setup, run):
            move_to(setup, tile)
            for i, pattern in enumerate(images):
                setup.panel.set(pattern=pattern)
                image = run.capture(setup.camera, f"{tile.name}/sli/{i:03d}.tiff", coding=coding, pattern=i,
                                    led_pattern=leds[i])
                print(f"{tile.name} pattern {i}: mean {image.mean():.1f}")
    show_finished(setup.panel, args.finished_s)
