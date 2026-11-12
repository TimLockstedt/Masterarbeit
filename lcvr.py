#!/usr/bin/env -S uv run
"""LCVR calibration: sweep one LCVR's voltage and record the camera intensity.

    scripts/calibrate/lcvr.py lcvr1 -c 139 138 -o results/calibration   # options: --help

Green spot at --center, the other LCVRs at 0 V, one frame per voltage step at
a fixed exposure (halved and restarted on saturation).

Writes <output>/<time>_lcvr_<name>/: sweep.csv, metadata.json.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from spec import Run, Setup, patterns

ROOT = Path(__file__).resolve().parents[2]
LCVRS = ["lcvr1", "lcvr2", "lcvr3", "lcvr4"]
SATURATION = 0.98
MIN_EXPOSURE_MS = 0.1
COLUMNS = ["step", "voltage_v", "exposure_ms", "mean", "max", "mean_r", "mean_g", "mean_b"]


class Saturated(Exception):
    pass


def channel_means(frame, pixel_format):
    """Mean R, G, B of a raw BayerRG frame; empty for other formats."""
    if not pixel_format.startswith("BayerRG"):
        return "", "", ""
    f = frame.astype(np.float64)
    green = (f[0::2, 1::2].mean() + f[1::2, 0::2].mean()) / 2
    return f[0::2, 0::2].mean(), green, f[1::2, 1::2].mean()


def sweep(setup, lcvr, voltages, exposure_ms, window=None):
    camera = setup.camera
    limit = SATURATION * camera.full_scale
    rows = []
    for step, voltage in enumerate(voltages):
        setup[lcvr].set(voltage_v=float(voltage))
        frame = camera.capture()  # waits for the LCVR (incl. its settle time)
        peak = int(frame.max())
        if peak >= limit:
            print(f"  {voltage:6.3f} V: saturated (max {peak})")
            raise Saturated
        mean = float(frame.mean(dtype=np.float64))
        rows.append([step, float(voltage), exposure_ms, mean, peak,
                     *channel_means(frame, camera.config.pixel_format)])
        print(f"  {voltage:6.3f} V: mean {mean:.3f}, max {peak}")
        if window is not None:
            window.plot({f"{exposure_ms:g} ms": [(row[1], row[3]) for row in rows]})
            window.show(frame, f"{lcvr} {voltage:6.3f} V @ {exposure_ms:g} ms: mean {mean:.3f}, max {peak}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lcvr", choices=LCVRS)
    parser.add_argument("-c", "--center", type=int, nargs=2, required=True, metavar=("X", "Y"),
                        help="camera centre on the panel (panel_camera_center.py)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
    parser.add_argument("-r", "--radius", type=int, default=15)
    parser.add_argument("-e", "--exposure-ms", type=float, default=500)
    parser.add_argument("-v", "--voltage-range", type=float, nargs=2, default=(0.0, 5.0), metavar=("MIN", "MAX"))
    parser.add_argument("-n", "--steps", type=int, default=101)
    parser.add_argument("--settle-ms", type=float, help="override the LCVR settle time")
    parser.add_argument("--no-window", action="store_true", help="no window")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("steps must be at least 2")

    config = {args.lcvr: {"settle_ms": args.settle_ms}} if args.settle_ms is not None else {}
    defaults = {name: {"voltage_v": 0} for name in LCVRS}
    setup = Setup.from_folder(args.instrument, "camera", "panel", *LCVRS, config=config, defaults=defaults)
    low, high = args.voltage_range
    voltages = np.linspace(low, high, args.steps)
    allowed = setup[args.lcvr].limits["voltage_v"]
    if not allowed[0] <= low < high <= allowed[1]:
        parser.error(f"{args.lcvr} is limited to {allowed[0]:g}..{allowed[1]:g} V")
    exposure = args.exposure_ms

    with setup, Run(setup, args.output, f"lcvr_{args.lcvr}") as run:
        window = None
        if not args.no_window:
            from spec.ui.gui import Window

            window = Window(f"{args.lcvr} calibration", setup.camera, plot=("voltage (V)", "mean"),
                            markers={"saturated": SATURATION * setup.camera.full_scale})
        setup.panel.set(pattern=patterns.ring(setup.panel.shape, args.center, args.radius + 0.5))
        print(f"calibrating {args.lcvr}: {low:g}..{high:g} V, {args.steps} steps")
        while True:
            setup.camera.set(exposure_ms=exposure)
            print(f"exposure {exposure:g} ms")
            try:
                rows = sweep(setup, args.lcvr, voltages, exposure, window)
                break
            except Saturated:
                exposure /= 2
                if exposure < MIN_EXPOSURE_MS:
                    raise RuntimeError("camera saturates even at minimum exposure")
        setup[args.lcvr].set(voltage_v=0)
        setup.panel.off()
        with (run.path / "sweep.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(rows)

    best = max(rows, key=lambda row: row[3])
    worst = min(rows, key=lambda row: row[3])
    message = f"max mean {best[3]:.3f} at {best[1]:.3f} V, min mean {worst[3]:.3f} at {worst[1]:.3f} V"
    print(f"\n{message}\nsaved {run.path}")
    if window is not None:
        window.finish(message)
        window.close()


if __name__ == "__main__":
    main()
