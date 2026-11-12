#!/usr/bin/env -S uv run
"""Overview camera: live view to align it, then save one image.

    scripts/calibrate/overview_camera.py -o results/calibration   # options: --help

Panel: green spot r=15 at the camera centre (--light white / off).
"Save image" (Enter) saves; closing the window cancels.
Later: the overview-to-stage homography (docs/DESIGN.md, section 12).

Writes <output>/<time>_overview_camera/: overview.tiff, metadata.json.
"""

import argparse
from pathlib import Path

import numpy as np

from spec import Run, Setup, patterns
from spec.ui.gui import Window

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
    parser.add_argument("--camera", default="overview_camera", help="overview camera device")
    parser.add_argument(
        "--light", choices=["spot", "white", "off"], default="spot",
        help="panel: green disc at the camera centre (default), all white, or off",
    )
    parser.add_argument("--radius", type=float, default=15, help="spot radius in panel px (default 15)")
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                        help="spot centre in panel px (default: calibration/panel_camera_center.toml)")
    args = parser.parse_args()

    names = [args.camera] if args.light == "off" else [args.camera, "panel"]
    setup = Setup.from_folder(args.instrument, *names)
    with setup, Run(setup, args.output, "overview_camera") as run:
        camera = setup[args.camera]
        if args.light == "spot":
            center = args.center or setup.calibration["panel_camera_center"]["center_px"]
            setup.panel.set(pattern=patterns.ring(setup.panel.shape, center, args.radius + 0.5, color=(0, 255, 0)))
        elif args.light == "white":
            setup.panel.set(pattern=np.full(setup.panel.shape, 255, np.uint8))
        print("align the overview camera, then 'Save image' (or Enter)")
        with Window("overview camera", camera) as window:
            window.text("align the overview camera, then save one image")
            saved = window.wait("Save image")
        if saved:
            image = run.capture(camera, "overview.tiff")
        if args.light != "off":
            setup.panel.off()
        if not saved:
            raise SystemExit("cancelled (window closed without saving)")
    print(f"{image.shape[1]}x{image.shape[0]} {image.dtype}, saved {run.path / 'overview.tiff'}")


if __name__ == "__main__":
    main()
