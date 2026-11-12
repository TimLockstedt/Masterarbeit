#!/usr/bin/env -S uv run
"""Focus calibration: autofocus to find the z (stage_z) of a sharp image.

    scripts/calibrate/camera_focus.py [-o results/calibration]   # options: --help

1. Live view: focus roughly (controller, Up/Down, the z field or the slider
   marked at the calibrated focus, "Calibrated focus"), optionally draw a ROI and
   set the exposure ("Auto exposure" or the field); Enter starts.
2. Autofocus from there (green spot r=15 at the camera centre): Nelder-Mead
   on the score (variance of the Laplacian of the green pixels). The focus is
   the maximum of a split Gaussian fitted to all images (else the best image).
   The exposure is never changed; if an image saturates, back to 1.
3. Check: sharpness (G1+G2 vs G1-G2 score, ~1 = noise) >= --min-sharpness,
   else back to 1.

The stage stays at the focus. With -o it writes <output>/<time>_camera_focus/:
focus.tiff, history.csv, result.toml (z_mm: copy to
config/<instrument>/calibration/camera_focus.toml yourself), metadata.json.
Without -o nothing is written (just focusing); the result is printed.
"""

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import curve_fit, minimize

from spec import Run, Setup, patterns
from spec.bayer import GREEN_SITES, green_image, green_pixels
from spec.ui.gui import add_overview_option

ROOT = Path(__file__).resolve().parents[2]
SATURATION = 0.98  # fraction of full scale counted as saturated
SATURATION_PERCENTILE = 99.7  # saturated if this percentile of the green pixels is (single bright pixels are fine)


class Saturated(Exception):
    def __init__(self, z, level):
        super().__init__(f"saturated at z = {z:.4f} mm ({SATURATION_PERCENTILE:g}th percentile of the "
                         f"green pixels at {level:.0%} of full scale)")
        self.z, self.level = z, level


# ---- image measures ----------------------------------------------------------------


def focus_score(green: np.ndarray, blur_px: int) -> float:
    """Variance of the Laplacian of the (median-blurred) green image."""
    if blur_px > 1:
        green = cv2.medianBlur(green, blur_px)
    return float(cv2.Laplacian(green, cv2.CV_64F).var())


def noise_image(frame: np.ndarray, pixel_format: str) -> np.ndarray | None:
    """G1 - G2 (+ offset, uint16): the two green sites' difference, i.e. pixel noise only."""
    sites = GREEN_SITES.get(pixel_format[5:7]) if pixel_format.startswith("Bayer") else None
    if sites is None:
        return None
    (r1, c1), (r2, c2) = sites
    height, width = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
    frame = frame[:height, :width].astype(np.int32)
    return (frame[r1::2, c1::2] - frame[r2::2, c2::2] + 4096).astype(np.uint16)


def sharpness(frame: np.ndarray, pixel_format: str, blur_px: int) -> float | None:
    """Score of G1 + G2 / score of G1 - G2: about 1 without structure, >> 1 when sharp."""
    noise = noise_image(frame, pixel_format)
    if noise is None:
        return None
    return focus_score(green_image(frame, pixel_format), blur_px) / max(focus_score(noise, blur_px), 1e-9)


def crop(frame: np.ndarray, roi) -> np.ndarray:
    """``frame[y0:y1, x0:x1]`` with the start on even pixels (keeps the Bayer phase); None: all."""
    if roi is None:
        return frame
    x0, y0, x1, y1 = (int(v) for v in roi)
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise ValueError(f"ROI {roi} is too small (at least 16 x 16 px)")
    return frame[y0:y1, x0:x1]


# ---- peak fit ----------------------------------------------------------------------


def peak_model(z, mu, amplitude, width_low, width_high, baseline):
    """Split Gaussian: a different width below and above ``mu`` (z scans are asymmetric)."""
    z = np.asarray(z, float)
    width = np.where(z < mu, width_low, width_high)
    return baseline + amplitude * np.exp(-((z - mu) ** 2) / (2 * width**2))


def fit_peak(z, score, fraction=0.2) -> dict | None:
    """Split Gaussian through the score peak; None if that does not work.

    Baseline: the lowest score (fixed). Only points at least ``fraction`` of the
    peak height above it are used. None with fewer than 4 different z, no
    convergence, or a maximum at the edge of the points used.
    """
    z, score = np.asarray(z, float), np.asarray(score, float)
    baseline = float(score.min())
    height = float(score.max()) - baseline
    if height <= 0:
        return None
    used = score - baseline >= fraction * height
    zu, su = z[used], score[used]
    low, high = float(zu.min()), float(zu.max())
    if len(np.unique(np.round(zu, 6))) < 4:
        return None
    width = (high - low) / 2
    try:
        (mu, amplitude, width_low, width_high), _ = curve_fit(
            lambda x, mu, a, wl, wh: peak_model(x, mu, a, wl, wh, baseline),
            zu, su,
            p0=[float(z[np.argmax(score)]), height, width, width],
            bounds=([low, 0, 1e-5, 1e-5], [high, 10 * height, 10 * width, 10 * width]),
        )
    except (RuntimeError, ValueError):
        return None
    margin = 0.01 * (high - low)  # a maximum on the edge is no peak
    if not low + margin < mu < high - margin:
        return None
    return {"z_mm": float(mu), "amplitude": float(amplitude), "baseline": baseline,
            "fwhm_low_mm": 1.1774 * float(width_low), "fwhm_high_mm": 1.1774 * float(width_high),
            "points": int(used.sum()), "z_range_mm": [low, high]}


def fit_curve(fit, points=80) -> list[tuple[float, float]]:
    """The fitted peak over the points used, for the plot."""
    z = np.linspace(*fit["z_range_mm"], points)
    widths = (fit["fwhm_low_mm"] / 1.1774, fit["fwhm_high_mm"] / 1.1774)
    y = peak_model(z, fit["z_mm"], fit["amplitude"], *widths, fit["baseline"])
    return list(zip(z.tolist(), y.tolist(), strict=True))


# ---- autofocus ---------------------------------------------------------------------


def move(stage, z, backlash_mm):
    """Move to z, always arriving from below (backlash)."""
    low = stage.limits["position_mm"][0]
    if z < stage.state.position_mm:
        stage.set(position_mm=max(z - backlash_mm, low))
    stage.set(position_mm=z)


def autofocus(setup, p, history, window=None, roi=None):
    """Nelder-Mead from the current z. Appends (z, score, green mean, sharpness) to
    ``history``; returns (z, best measured z, fit or None). Raises Saturated."""
    stage, camera = setup.stage_z, setup.camera
    pixel_format = camera.config.pixel_format
    z0 = stage.read().position_mm  # includes moves made with the controller
    low, high = stage.limits["position_mm"]
    low, high = max(low, z0 - p["max_distance_mm"]), min(high, z0 + p["max_distance_mm"])

    def plot(mark, curves=None):
        if window is not None:
            window.plot({"images": [(h[0], h[1]) for h in history]}, mark=mark, curves=curves)

    def negative_score(x):
        z = float(np.clip(x[0], low, high))
        move(stage, z, p["backlash_mm"])
        full = camera.capture()  # waits for the stage
        frame = crop(full, roi)
        green = green_pixels(frame, pixel_format)
        level = float(np.percentile(green, SATURATION_PERCENTILE)) / camera.full_scale
        if level >= SATURATION:
            raise Saturated(z, level)
        score = focus_score(green_image(frame, pixel_format), p["blur_px"])
        ratio = sharpness(frame, pixel_format, p["blur_px"])
        history.append((z, score, float(green.mean(dtype=np.float64)), ratio))
        print(f"  z={z:8.4f} mm: score {score:12.2f}  sharpness {ratio:.2f}")
        if window is not None:
            best = max(history, key=lambda h: h[1])
            plot(best[0])
            window.show(full, f"image {len(history)}: z={z:.4f} mm  score {score:.2f}  sharpness {ratio:.2f}\n"
                              f"best: z={best[0]:.4f} mm  score {best[1]:.2f}")
        return -score

    second = z0 + p["step_mm"] if z0 + p["step_mm"] <= high else z0 - p["step_mm"]
    print(f"Nelder-Mead from z = {z0:.4f} mm, exposure {camera.state.exposure_ms:g} ms")
    minimize(
        negative_score, [z0], method="Nelder-Mead", bounds=[(low, high)],
        options={"initial_simplex": [[z0], [second]], "xatol": p["tolerance_mm"],
                 "fatol": np.inf, "maxfev": p["max_images"]},
    )
    best = max(history, key=lambda h: h[1])[0]
    fit = fit_peak([h[0] for h in history], [h[1] for h in history], p["fit_fraction"])
    z = fit["z_mm"] if fit else best
    print(f"peak fit: z = {z:.4f} mm" if fit else f"peak fit failed: best image at z = {z:.4f} mm")
    plot(z, {"fit": fit_curve(fit)} if fit else None)
    return z, best, fit


# ---- window ------------------------------------------------------------------------


def focus_window(setup, p, overview=None):
    """The run's window (spec.ui.gui): live view with z Up/Down and its step, z field and a
    slider marked at the calibrated focus, score vs z."""
    from spec.ui.gui import Window

    stage, camera = setup.stage_z, setup.camera
    low, high = stage.limits["position_mm"]
    window = Window("focus calibration", camera, overview=overview, plot=("z (mm)", "score"),
                    exposure_target=p["exposure_target"])
    step = window.entry("step mm", f"{p['manual_step_mm']:g}", width=7)

    # z by hand: field (type + Enter, arrows by the step) and a slider around the calibrated focus
    reference = setup.calibration.get("camera_focus", {}).get("z_mm")  # calibration/camera_focus.toml
    middle = reference if reference is not None else stage.state.position_mm
    box = window.section(f"{stage.name} (mm)")
    z_field = window.number("z", stage.state.position_mm, lambda z: stage.set(position_mm=z), low, high,
                            step=p["manual_step_mm"], integer=False, parent=box)

    def step_changed(*_):  # the field's arrows follow the step
        try:
            z_field.widget.config(increment=float(step.get()))
        except ValueError:
            pass

    step.trace_add("write", step_changed)
    show_slider = window.marked_slider(
        f"z, calibrated focus +-{p['slider_span_mm']:g} mm", max(low, middle - p["slider_span_mm"]),
        min(high, middle + p["slider_span_mm"]), stage.state.position_mm,
        lambda z: stage.set(position_mm=z), marks={"calibrated": reference} if reference is not None else {},
        parent=box)
    if reference is not None:
        window.button("Calibrated focus", lambda: stage.set(position_mm=float(np.clip(reference, low, high))))

    def manual(direction):
        try:
            distance = float(step.get())
        except ValueError:
            raise ValueError("step must be a number") from None
        z = stage.state.position_mm + direction * distance
        stage.set(position_mm=float(np.clip(z, low, high)))

    def status():
        z = stage.read().position_mm  # also follows moves made with the controller
        show_slider(z)
        if window.root.focus_get() is not z_field.widget:  # not while typing
            z_field.set(f"{z:g}")
        text, roi = f"z = {z:.4f} mm", window.roi
        if window.view.frame is not None:
            ratio = sharpness(crop(window.view.frame, roi), camera.config.pixel_format, p["blur_px"])
            if ratio is not None:
                text += f"   sharpness {ratio:.2f} (need {p['min_sharpness']:g})"
        text += f"   ROI {roi[2] - roi[0]}x{roi[3] - roi[1]}" if roi else "   ROI: whole frame (drag to draw)"
        return text + "   Up/Down: move z"

    window.button("Down", lambda: manual(-1), key="<Down>")
    window.button("Up", lambda: manual(+1), key="<Up>")
    window.status(status)
    return window


# ---- main --------------------------------------------------------------------------


def write_history(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["attempt", "z_mm", "exposure_ms", "score", "green_mean", "sharpness"])
        for attempt, exposure, z, score, mean, ratio in rows:
            writer.writerow([attempt, f"{z:.5f}", f"{exposure:g}", f"{score:.4f}", f"{mean:.3f}", f"{ratio:.3f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path,
                        help="output folder (default: nothing is written, just focus)")
    parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
    parser.add_argument("-e", "--exposure-ms", type=float, default=500, help="start exposure")
    parser.add_argument("-g", "--gain", type=float, help="camera gain (default: camera toml)")
    parser.add_argument("--settle-ms", type=float, default=200,
                        help="stage_z wait after every move; vibrations lower the score (2026-11-05)")
    parser.add_argument("--manual-step-mm", type=float, default=0.05, help="Up/Down step in the live view")
    parser.add_argument("--slider-span-mm", type=float, default=1.0,
                        help="the z slider covers the calibrated focus +- this (default 1)")
    parser.add_argument("--step-mm", type=float, default=0.05, help="first step of the optimizer")
    parser.add_argument("--tolerance-mm", type=float, default=0.01,
                        help="optimizer stops when z is known to this; the peak fit then does better "
                        "(simulated 2026-11-06: 10 um -> ~11 images, fit within 1.5 um; 2 um -> ~15 images)")
    parser.add_argument("--max-images", type=int, default=40, help="optimizer images at most")
    parser.add_argument("--max-distance-mm", type=float, default=1.0, help="optimizer stays this close to the start")
    parser.add_argument("--fit-fraction", type=float, default=0.2,
                        help="peak fit uses the images this far up the peak (0..1, above the lowest score)")
    parser.add_argument("--min-sharpness", type=float, default=10.0,
                        help="fail unless the focus image is this sharp (noise only ~2; "
                        "2026-11-05: sharp 47, 0.5 mm off 6-10)")
    parser.add_argument("--backlash-mm", type=float, default=0.05,
                        help="a move downwards first goes this much further down")
    parser.add_argument("--exposure-target", type=float, default=0.75,
                        help="'Auto exposure' button: raw 99.9th percentile, fraction of full scale")
    parser.add_argument("--blur-px", type=int, choices=[1, 3, 5], default=5,
                        help="median blur before the Laplacian (1: none)")
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                        help="spot centre in panel px (default: calibration/panel_camera_center.toml)")
    parser.add_argument("-r", "--radius", type=float, default=15, help="spot radius in panel px")
    parser.add_argument("--color", type=int, nargs=3, default=(0, 255, 0), metavar=("R", "G", "B"))
    add_overview_option(parser)
    args = parser.parse_args()

    camera = {key: value for key, value in (("exposure_ms", args.exposure_ms), ("gain", args.gain))
              if value is not None}
    setup = Setup.from_folder(
        args.instrument, "camera", "panel", "stage_z", *filter(None, [args.overview]),
        config={"stage_z": {"settle_ms": args.settle_ms}}, defaults={"camera": camera},
    )
    p = {key: getattr(args, key) for key in (
        "manual_step_mm", "slider_span_mm", "step_mm", "tolerance_mm", "max_images", "max_distance_mm", "fit_fraction",
        "min_sharpness", "backlash_mm", "exposure_target", "blur_px", "settle_ms")}
    center = args.center or setup.calibration["panel_camera_center"]["center_px"]
    p.update(center_px=list(center), radius_px=args.radius, color_rgb=list(args.color))
    setup.measurement = p  # recorded as "parameters" in metadata.json

    with setup, (Run(setup, args.output, "camera_focus") if args.output else nullcontext()) as run:
        record = run.record if run else lambda key, value: None
        print(f"output: {run.path}" if run else "no output folder (-o): nothing is written")
        setup.panel.set(pattern=patterns.ring(setup.panel.shape, center, args.radius + 0.5, color=p["color_rgb"]))
        window = focus_window(setup, p, setup[args.overview] if args.overview else None)
        rows, attempts = [], []  # every image / every attempt
        print("focus roughly (Zaber controller or Up/Down), then Enter or 'Start autofocus'")
        while True:
            if not window.wait("Start autofocus"):
                record("attempts", attempts)
                raise SystemExit("cancelled (window closed without start)")
            roi, attempt = window.roi, len(attempts) + 1  # fixed for this attempt
            exposure = setup.camera.state.exposure_ms
            window.text(f"attempt {attempt}: autofocus running ...")
            print(f"\nattempt {attempt}, ROI {list(roi) if roi else 'whole frame'}")
            history = []
            try:
                z, best, fit = autofocus(setup, p, history, window, roi)
            except Saturated as error:
                attempts.append({"error": str(error), "exposure_ms": exposure})
                message = f"attempt {attempt}: {error}. Lower the exposure, Enter: try again"
            else:
                move(setup.stage_z, z, p["backlash_mm"])
                full = setup.camera.capture()
                frame = crop(full, roi)
                ratio = sharpness(frame, setup.camera.config.pixel_format, p["blur_px"])
                score = focus_score(green_image(frame, setup.camera.config.pixel_format), p["blur_px"])
                sharp = ratio is None or ratio >= p["min_sharpness"]
                name = "focus.tiff" if sharp else f"failed_{attempt}.tiff"
                if run:
                    run.save(full, name, z_mm=z, measurement_time=setup.camera.last_capture_time, camera="camera")
                window.show(full)
                attempts.append({"z_mm": z, "z_best_image_mm": best, "fit": fit, "score": score,
                                 "sharpness": ratio, "exposure_ms": exposure, "images": len(history),
                                 "roi_px": list(roi) if roi else None, "image": name})
                if sharp:
                    break
                message = (f"attempt {attempt}: not sharp (sharpness {ratio:.2f} at z = {z:.4f} mm, "
                           f"need {p['min_sharpness']:g}). Focus roughly, check the ROI, Enter: try again")
            finally:
                rows += [(attempt, exposure, *h) for h in history]
                if run:
                    write_history(run.path / "history.csv", rows)  # also after an error
            print(message)
            if window.closed:
                record("attempts", attempts)
                raise RuntimeError(message)
            window.text(message)  # shown in the next wait()
        record("attempts", attempts)
        record("camera_focus", attempts[-1])
        setup.panel.off()

    how = "peak of a split-Gaussian fit" if fit else "best image (peak fit failed)"
    message = f"focus at z = {z:.4f} mm (score {score:.2f}, sharpness {ratio:.2f})"
    if run is None:
        print(f"\n{message}; stage_z stays there (nothing saved: no -o)")
        window.finish(message)
        window.close()
        return
    (run.path / "result.toml").write_text(
        f"# focus calibration, {run.metadata['started']}; copy to config/<instrument>/calibration/camera_focus.toml\n"
        f"# attempt {attempt}, {len(history)} images, ROI {list(roi) if roi else 'whole frame'}, "
        f"blur {p['blur_px']} px; z_mm: {how}, best image at {best:.5f} mm\n"
        f"z_mm = {z:.5f}\nscore = {score:.4f}\nsharpness = {ratio:.3f}\nexposure_ms = {exposure:g}\n"
    )
    print(f"\n{message}\nsaved {run.path}")
    window.finish(message)
    window.close()


if __name__ == "__main__":
    main()
