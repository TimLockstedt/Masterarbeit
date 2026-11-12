#!/usr/bin/env -S uv run
"""PLI: ring illumination, one image per LCVR voltage state and tile.

    scripts/measurements/pli.py [pli.toml] [-t SECONDS]

Images: <run>/<tile>/pli/<state index>.tiff; LED pattern: <run>/led_pattern/pli/000.png
When the run ended without error, the whole panel is green for -t seconds (default 5).
"""

import argparse
from pathlib import Path

from spec import Run, Setup, patterns
from spec.grid import move_to, tiles_from
from spec.run import add_finished_option, show_finished

LCVRS = ["lcvr1", "lcvr2", "lcvr3", "lcvr4"]

parser = argparse.ArgumentParser(description="PLI: ring illumination")
parser.add_argument("toml", nargs="?", default=Path(__file__).with_suffix(".toml"), type=Path)
add_finished_option(parser)
args = parser.parse_args()

setup = Setup.from_toml(args.toml)
p = setup.measurement
center = p.get("center_px") or setup.calibration["panel_camera_center"]["center_px"]
states = p["retarder_states_v"]
if any(len(state) != len(LCVRS) for state in states):
    raise SystemExit(f"each retarder state needs {len(LCVRS)} voltages")

with setup:
    with Run(setup) as run:
        print(f"output: {run.path}")
        ring = patterns.ring(
            setup.panel.shape,
            center,
            p["outer_radius_px"],
            p.get("inner_radius_px", 0),
            color=p["color_rgb"],
        )
        [led] = run.led_patterns(
            "pli", ring, f"ring r={p.get('inner_radius_px', 0)}..{p['outer_radius_px']} px at {list(center)}"
        )
        setup.panel.set(pattern=ring)
        for tile in tiles_from(p, setup, run):
            move_to(setup, tile)
            for i, voltages in enumerate(states):
                for name, voltage in zip(LCVRS, voltages):
                    setup[name].set(voltage_v=voltage)
                image = run.capture(setup.camera, f"{tile.name}/pli/{i:03d}.tiff", state=i, led_pattern=led)
                print(f"{tile.name} state {i} {voltages} V: mean {image.mean():.1f}")
    show_finished(setup.panel, args.finished_s)
