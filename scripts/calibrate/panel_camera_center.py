#!/usr/bin/env -S uv run
"""Camera centre: find the panel pixel imaged at the camera centre.

    scripts/calibrate/panel_camera_center.py -o results/calibration   # options: --help

Remove all optics from the beam path first (sample, LCVRs, polarizers, ...).
A green disc is moved by Nelder-Mead to maximize the mean green signal, with
the radius shrinking 15, 13, ..., 1, 0. Fixed gain; the exposure is reset per
radius (green p99.9 at half of full scale) and halved on saturation.

Writes <output>/<time>_panel_camera_center/: result.toml (center_px: copy to
config/<instrument>/calibration/panel_camera_center.toml yourself), radius_NN.tiff,
history.csv, metadata.json.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from spec import Run, Setup, patterns
from spec.bayer import green_pixels

ROOT = Path(__file__).resolve().parents[2]
SATURATION = 0.98  # fraction of full scale counted as saturated
SATURATED_FRACTION = 1e-4  # more saturated green pixels than this: the frame is saturated
TARGET = 0.5  # exposure target: green 99.9th percentile as a fraction of full scale
MIN_EXPOSURE_MS = 0.1
STEP_PX = 16  # initial simplex size for the first radius
MIN_STEP_PX = 2  # smallest initial simplex for later radii (step = radius // 2)
MAX_EVALUATIONS = 120
MAX_RESTARTS = 8


class Saturated(Exception):
    pass


def radii(start: int, step: int) -> list[int]:
    """start, start - step, ... down to 0 (always ends with 0: a single pixel)."""
    values = list(range(start, 0, -step))
    return values + [0]


class SpotSearch:
    def __init__(self, setup, exposure_ms, color, gain_db=0.0, window=None):
        self.setup, self.color = setup, color
        self.height, self.width = setup.panel.shape[:2]
        self.full_scale = setup.camera.full_scale
        self.limit = SATURATION * self.full_scale
        self.pixel_format = setup.camera.config.pixel_format
        low, high = setup.camera.limits.get("exposure_ms", (MIN_EXPOSURE_MS, 10_000))
        self.min_exposure, self.max_exposure = max(low, MIN_EXPOSURE_MS), high
        self.exposure_ms, self.gain_db = exposure_ms, gain_db
        self.window = window  # spec.ui.gui.Window or None
        self.radius = None
        self.cache: dict[tuple[int, int], float] = {}
        # (radius, x, y, exposure_ms, gain_db, mean, p99.9)
        self.history: list[tuple[int, int, int, float, float, float, float]] = []

    # ---- one position ---------------------------------------------------------

    def clip(self, x, y):
        r = self.radius
        return (
            int(np.clip(round(float(x)), r, self.width - 1 - r)),
            int(np.clip(round(float(y)), r, self.height - 1 - r)),
        )

    def pattern(self, x, y):
        return patterns.ring(self.setup.panel.shape, (x, y), self.radius + 0.5, color=self.color)

    def green(self, x, y) -> np.ndarray:
        self.setup.panel.set(pattern=self.pattern(x, y))
        frame = self.setup.camera.capture()  # waits for the panel
        green = green_pixels(frame, self.pixel_format)
        if self.window is not None:
            self.window.show(
                frame,
                f"r={self.radius} ({x}, {y}) @ {self.setting()}\n"
                f"green: mean {green.mean(dtype=np.float64):.3f}, p99.9 {np.percentile(green, 99.9):.0f}",
            )
        return green

    def saturated(self, green) -> bool:
        return np.count_nonzero(green >= self.limit) > SATURATED_FRACTION * green.size

    def measure(self, x, y) -> float:
        x, y = self.clip(x, y)
        if (x, y) in self.cache:
            return self.cache[(x, y)]
        green = self.green(x, y)
        if self.saturated(green):
            print(f"  ({x:3d}, {y:3d}): saturated")
            raise Saturated
        score = float(green.mean(dtype=np.float64))
        p999 = float(np.percentile(green, 99.9))
        self.cache[(x, y)] = score
        self.history.append((self.radius, x, y, self.exposure_ms, self.gain_db, score, p999))
        print(f"  r={self.radius:2d} ({x:3d}, {y:3d}) @ {self.setting()}: mean {score:.3f}, p99.9 {p999:.0f}")
        return score

    # ---- exposure ---------------------------------------------------------------

    def setting(self) -> str:
        return f"{self.exposure_ms:g} ms, {self.gain_db:.1f} dB"

    def set_exposure(self, exposure_ms):
        exposure_ms = float(np.clip(exposure_ms, self.min_exposure, self.max_exposure))
        if exposure_ms != self.exposure_ms:
            self.exposure_ms = exposure_ms
            print(f"exposure -> {self.setting()}")
        self.setup.camera.set(exposure_ms=exposure_ms, gain=self.gain_db)
        self.cache.clear()

    def reduce_exposure(self):
        if self.exposure_ms <= self.min_exposure:
            raise RuntimeError("camera saturates even at minimum exposure")
        self.set_exposure(self.exposure_ms / 2)

    def expose(self, x, y):
        """Set the exposure so the green 99.9th percentile at (x, y) is TARGET of full scale."""
        for _ in range(5):
            green = self.green(x, y)
            if self.saturated(green):
                wanted = self.exposure_ms / 4
            else:
                p999 = max(float(np.percentile(green, 99.9)), 1.0)
                wanted = self.exposure_ms * TARGET * self.full_scale / p999
            wanted = float(np.clip(wanted, self.min_exposure, self.max_exposure))
            if abs(wanted / self.exposure_ms - 1) < 0.2 or wanted == self.exposure_ms:
                break
            self.set_exposure(wanted)
        if self.exposure_ms >= self.max_exposure:
            print(f"warning: exposure at its limit ({self.setting()}); the signal is weak")

    def unsaturated(self, x, y):
        while True:
            try:
                return self.measure(x, y)
            except Saturated:
                self.reduce_exposure()

    # ---- search -----------------------------------------------------------------

    def best(self):
        return max(self.cache, key=self.cache.get)

    def probe(self, start, distance):
        """Measure ``start`` and the four points ``distance`` px away in +-x, +-y."""
        x, y = start
        points = [(x, y), (x + distance, y), (x, y + distance), (x - distance, y), (x, y - distance)]
        while True:
            try:
                for point in points:
                    self.measure(*point)
                return
            except Saturated:
                self.reduce_exposure()  # clears the cache: probe again at the new exposure

    def search(self, start, step, probe_px=None):
        """Nelder-Mead at the current radius, from ``start``; returns the best position."""
        self.cache.clear()
        if probe_px:
            self.probe(start, probe_px)
            print(f"probe: best {self.best()}")
        else:
            self.unsaturated(*start)
        for _ in range(MAX_RESTARTS):
            x0 = np.array(self.best(), dtype=float)
            simplex = np.array([x0, x0 + (step, 0), x0 + (0, step)])
            try:
                minimize(
                    lambda point: -self.measure(*point),
                    x0,
                    method="Nelder-Mead",
                    options={"initial_simplex": simplex, "xatol": 0.5, "fatol": 0, "maxfev": MAX_EVALUATIONS},
                )
                return self.best()
            except Saturated:
                best = self.best()
                self.reduce_exposure()
                self.unsaturated(*best)
        raise RuntimeError("search kept saturating; too many restarts")

    def run(self, start, radius, radius_step, probe_px, on_found=None):
        """Search at every radius, each starting where the previous one ended."""
        point, stages = start, []
        self.set_exposure(self.exposure_ms)
        for index, r in enumerate(radii(radius, radius_step)):
            self.radius = r
            point = self.clip(*point)
            print(f"\nradius {r} px, from {point}")
            self.expose(*point)
            first = index == 0
            point = self.search(
                point,
                STEP_PX if first else max(MIN_STEP_PX, r // 2),
                probe_px if first else None,
            )
            stage = {
                "radius_px": r,
                "center_px": list(point),
                "exposure_ms": self.exposure_ms,
                "gain_db": self.gain_db,
                "mean": self.cache[point],
                "evaluations": sum(1 for h in self.history if h[0] == r),
            }
            stages.append(stage)
            print(f"radius {r} px: centre {point} @ {self.setting()}, mean {stage['mean']:.3f}")
            if on_found is not None:
                on_found(stage)
        return point, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
    parser.add_argument("-e", "--exposure-ms", type=float, default=500, help="start exposure")
    parser.add_argument("-g", "--gain-db", type=float, default=12, help="fixed camera gain (dB)")
    parser.add_argument("-r", "--radius", type=int, default=15, help="first spot radius in panel px")
    parser.add_argument("--radius-step", type=int, default=2, help="radius decrease per stage (px)")
    parser.add_argument("--probe-px", type=int, default=32, help="probe distance in +-x, +-y (panel px)")
    parser.add_argument("--start", type=int, nargs=2, metavar=("X", "Y"), help="start position (default: current calibration, else panel centre)")
    parser.add_argument("--color", type=int, nargs=3, default=(0, 255, 0), metavar=("R", "G", "B"))
    parser.add_argument("--no-window", "--no-histogram", action="store_true", help="no window")
    args = parser.parse_args()
    if args.radius < 0 or args.radius_step < 1:
        parser.error("--radius must be >= 0 and --radius-step >= 1")

    print("note: no optical elements in the beam path (sample, LCVRs, polarizers, filters, ...)")
    setup = Setup.from_folder(args.instrument, "camera", "panel")
    with setup, Run(setup, args.output, "panel_camera_center") as run:
        window = None
        if not args.no_window:
            from spec.ui.gui import Window

            full_scale = setup.camera.full_scale
            window = Window(
                "camera centre", setup.camera,
                markers={"target": TARGET * full_scale, "saturated": SATURATION * full_scale},
            )
        search = SpotSearch(setup, args.exposure_ms, args.color, args.gain_db, window)

        def save_image(stage):
            setup.panel.set(pattern=search.pattern(*stage["center_px"]))
            run.capture(setup.camera, f"radius_{stage['radius_px']:02d}.tiff", **stage)

        height, width = setup.panel.shape[:2]
        start = args.start or setup.calibration.get("panel_camera_center", {}).get("center_px") or (width // 2, height // 2)
        print(f"start at {tuple(start)}")
        (x, y), stages = search.run(
            tuple(start),
            args.radius,
            args.radius_step,
            args.probe_px,
            on_found=save_image,
        )
        run.record("stages", stages)
        setup.panel.off()

    np.savetxt(
        run.path / "history.csv",
        np.array(search.history),
        delimiter=",",
        header="radius_px,x_px,y_px,exposure_ms,gain_db,mean,p99_9",
        comments="",
        fmt=["%d", "%d", "%d", "%g", "%.2f", "%.6f", "%.1f"],
    )
    lines = [
        f"# camera centre on the panel, {run.metadata['started']}",
        f"# radii {', '.join(str(s['radius_px']) for s in stages)} px, probe {args.probe_px} px, "
        f"green pixels, {len(search.history)} evaluations",
        f"center_px = [{x}, {y}]",
        "",
    ]
    for stage in stages:
        lines += [
            "[[stages]]",
            f"radius_px = {stage['radius_px']}",
            f"center_px = [{stage['center_px'][0]}, {stage['center_px'][1]}]",
            f"exposure_ms = {stage['exposure_ms']:g}",
            f"gain_db = {stage['gain_db']:.2f}",
            f"mean = {stage['mean']:.6f}",
            f"evaluations = {stage['evaluations']}",
            "",
        ]
    (run.path / "result.toml").write_text("\n".join(lines))
    message = f"camera centre at panel pixel x={x}, y={y}"
    print(f"\n{message}\nsaved {run.path}")
    if window is not None:
        window.finish(message)
        window.close()


if __name__ == "__main__":
    main()
