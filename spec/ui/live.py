"""Live view of a camera: raw or debayered image, zoom/pan, ROI and histogram (tk).

    uv run spec-live [device toml]          # default: config/sp01/camera.toml
    spec.ui.gui.live_view(setup.camera)     # from a script, setup connected

The building blocks of spec.ui.gui.Window (which scripts use).

Mouse: wheel zooms at the cursor, middle button (or Ctrl + left) drags the
view, left button draws a rectangular ROI, right click clears it. The
histogram and the statistics use the ROI, or the visible part of the image
when there is no ROI. Bayer data: R, G (both green sites), B separately, raw
counts. Keys: f = fit, 1 = 100 %, Escape = clear ROI.
"""

import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_TOMLS = [Path(__file__).resolve().parents[3] / "config" / "sp01" / f"{name}.toml"
                 for name in ("camera", "overview_camera")]
CHANNEL_COLORS = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "I": "#555555"}
HISTOGRAM_BINS = 256


def is_bayer(frame: np.ndarray, pixel_format: str | None) -> bool:
    return frame.ndim == 2 and bool(pixel_format) and pixel_format.startswith("BayerRG")


def debayer(frame: np.ndarray) -> np.ndarray:
    """BayerRG mosaic (R at row 0, column 0) -> RGB (same dtype).

    OpenCV names Bayer patterns by a shifted 2x2 block, so a sensor's "RG"
    pattern is OpenCV's "BG"."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BayerBG2RGB)


def full_scale_of(camera, frame: np.ndarray) -> int:
    value = getattr(camera, "full_scale", None)
    if value:
        return int(value)
    return 255 if frame.dtype == np.uint8 else int(np.iinfo(frame.dtype).max)


class View:
    """Mapping between image pixels and canvas pixels (zoom and pan)."""

    MAX_SCALE = 64.0

    def __init__(self, image_shape, canvas_size):
        self.height, self.width = image_shape[:2]
        self.canvas_width, self.canvas_height = canvas_size
        self.fit()

    @property
    def fit_scale(self) -> float:
        return min(self.canvas_width / self.width, self.canvas_height / self.height)

    def fit(self):
        self.set_scale(self.fit_scale, (self.width / 2, self.height / 2))

    def set_scale(self, scale, center=None):
        """Zoom to ``scale`` (canvas px per image px) around image point ``center``."""
        if center is None:
            center = self.canvas_to_image(self.canvas_width / 2, self.canvas_height / 2)
        self.scale = float(np.clip(scale, self.fit_scale / 4, self.MAX_SCALE))
        self.x0 = center[0] - self.canvas_width / 2 / self.scale
        self.y0 = center[1] - self.canvas_height / 2 / self.scale

    def zoom_at(self, canvas_x, canvas_y, factor):
        x, y = self.canvas_to_image(canvas_x, canvas_y)
        self.scale = float(np.clip(self.scale * factor, self.fit_scale / 4, self.MAX_SCALE))
        self.x0, self.y0 = x - canvas_x / self.scale, y - canvas_y / self.scale

    def pan(self, d_canvas_x, d_canvas_y):
        self.x0 -= d_canvas_x / self.scale
        self.y0 -= d_canvas_y / self.scale

    def resize(self, canvas_size):
        center = self.canvas_to_image(self.canvas_width / 2, self.canvas_height / 2)
        self.canvas_width, self.canvas_height = canvas_size
        self.set_scale(self.scale, center)

    def canvas_to_image(self, cx, cy):
        return self.x0 + cx / self.scale, self.y0 + cy / self.scale

    def image_to_canvas(self, x, y):
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale

    def region(self) -> tuple[int, int, int, int]:
        """Visible image pixels (x0, y0, x1, y1), clipped; may be empty."""
        x0 = max(0, math.floor(self.x0))
        y0 = max(0, math.floor(self.y0))
        x1 = min(self.width, math.ceil(self.x0 + self.canvas_width / self.scale))
        y1 = min(self.height, math.ceil(self.y0 + self.canvas_height / self.scale))
        return x0, y0, max(x0, x1), max(y0, y1)


def to_uint8(image: np.ndarray, full_scale: int, auto: bool) -> np.ndarray:
    if auto:
        sample = image[:: max(1, image.shape[0] // 512), :: max(1, image.shape[1] // 512)]
        low, high = np.percentile(sample, (0.5, 99.5))
    else:
        low, high = 0, full_scale
    scaled = (image.astype(np.float32) - low) * (255.0 / max(high - low, 1e-9))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def render(source: np.ndarray, view: View, full_scale: int, auto: bool):
    """Visible part of ``source`` (raw or RGB) as (uint8 array, canvas size, canvas x, y)."""
    x0, y0, x1, y1 = view.region()
    if x1 <= x0 or y1 <= y0:
        return None
    step = max(1, int(1 / view.scale))  # skip pixels that would not be shown anyway
    crop = source[y0:y1:step, x0:x1:step]
    size = (max(1, round((x1 - x0) * view.scale)), max(1, round((y1 - y0) * view.scale)))
    cx, cy = view.image_to_canvas(x0, y0)
    return to_uint8(crop, full_scale, auto), size, cx, cy


def channels(frame: np.ndarray, region, bayer: bool) -> dict[str, np.ndarray]:
    """Raw samples per channel inside ``region`` (Bayer: both greens pooled)."""
    x0, y0, x1, y1 = region
    if bayer:
        x0, y0 = x0 - x0 % 2, y0 - y0 % 2  # keep the RG/GB phase
        sub = frame[y0:y1, x0:x1]
        green = np.concatenate([sub[0::2, 1::2].ravel(), sub[1::2, 0::2].ravel()])
        return {"R": sub[0::2, 0::2].ravel(), "G": green, "B": sub[1::2, 1::2].ravel()}
    sub = frame[y0:y1, x0:x1]
    if sub.ndim == 3:
        return {name: sub[..., i].ravel() for i, name in enumerate("RGB")}
    return {"I": sub.ravel()}


def histogram(frame, region, bayer, full_scale, bins=HISTOGRAM_BINS):
    """{channel: (counts, mean, max)} with ``bins`` equal bins over 0..full_scale."""
    result = {}
    for name, values in channels(frame, region, bayer).items():
        if values.size == 0:
            continue
        index = np.minimum(values.astype(np.int64) * bins // (full_scale + 1), bins - 1)
        result[name] = (np.bincount(index, minlength=bins), float(values.mean()), int(values.max()))
    return result


SERIES_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def draw_scan(canvas, series: dict, mark: float | None = None, x_label="x", y_label="score", curves=None):
    """Plot ``series`` = {name: [(x, y), ...]} as points on a tk canvas.

    One colour per series, the latest point of each circled, ``mark`` (e.g.
    the best x so far) as a dashed line, ``curves`` ({name: [(x, y), ...]},
    e.g. a fit) as lines."""
    curves = curves or {}
    c = canvas
    c.delete("all")
    width, height = int(c["width"]), int(c["height"])
    left, right, top, bottom = 60, 10, 24, 30  # top: room for the y label
    points = [p for values in (*series.values(), *curves.values()) for p in values]
    c.create_text(4, 2, text=y_label, anchor="nw", fill="#555555")
    if not points:
        return
    xs, ys = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    x1, y1 = (x1 if x1 > x0 else x0 + 1), (y1 if y1 > y0 else y0 + 1)

    def to_canvas(x, y):
        return (
            left + (x - x0) / (x1 - x0) * (width - left - right),
            height - bottom - (y - y0) / (y1 - y0) * (height - top - bottom),
        )

    c.create_rectangle(left, top, width - right, height - bottom, outline="#999999")
    for x, anchor in ((x0, "nw"), (x1, "ne")):
        c.create_text(to_canvas(x, y0)[0], height - bottom + 3, text=f"{x:.4g}", anchor=anchor)
    c.create_text((left + width - right) / 2, height - 2, text=x_label, anchor="s")
    for y in (y0, y1):
        c.create_text(left - 4, to_canvas(x0, y)[1], text=f"{y:.4g}", anchor="e")
    if mark is not None:
        x = to_canvas(mark, y0)[0]
        c.create_line(x, top, x, height - bottom, dash=(4, 3), fill="#888888")
    for index, (name, values) in enumerate(curves.items(), start=len(series)):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        if len(values) > 1:
            c.create_line(*[v for x, y in values for v in to_canvas(x, y)], fill=color, width=2)
        c.create_text(width - right - 4, top + 3 + 13 * index, text=name, anchor="ne", fill=color)
    for index, (name, values) in enumerate(series.items()):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        for x, y in values:
            cx, cy = to_canvas(x, y)
            c.create_oval(cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5, fill=color, outline=color)
        if values:
            cx, cy = to_canvas(*values[-1])
            c.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, outline=color, width=2)  # latest
        c.create_text(width - right - 4, top + 3 + 13 * index, text=name, anchor="ne", fill=color)


class LiveView:
    """Image, histogram and statistics of a camera in ``root`` (tk).

    ``live``: capture continuously from the start (else only ``show()`` and
    ``resume()``). ``stop()``/``resume()`` switch the capture off and on.
    ``overview``: a further camera (e.g. the overview camera); "histogram /
    <overview>" above the histogram shows its image there instead, captured in
    its own thread while the live capture runs and it is selected."""

    def __init__(self, root, camera, canvas_size=(900, 750), live=True, overview=None):
        import tkinter as tk

        self.root, self.camera = root, camera
        self.pixel_format = getattr(camera.config, "pixel_format", None)
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.running = threading.Event()
        self.running.set()
        self.paused = tk.BooleanVar(value=False)
        self.pause_event = threading.Event()  # the grab thread must not touch tk
        self.frame = None  # latest raw frame
        self.rgb = None  # its debayered version (lazy)
        self.view: View | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_enabled = True  # False: the ROI is shown but can't be changed
        self.markers: dict[str, float] = {}  # {label: counts} as dashed lines in the histogram
        self.drag = None
        self.photo = None
        self.count, self.since = 0, time.monotonic()

        root.title(f"{camera.name} live")
        main = tk.Frame(root)
        main.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(main, width=canvas_size[0], height=canvas_size[1], bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        side = self.side = tk.Frame(main)  # scripts may add widgets here
        side.pack(side="right", fill="y", padx=6, pady=6)
        self.overview = overview
        self.overview_on = threading.Event()  # the overview image is selected (for the thread)
        self.overview_frames: queue.Queue = queue.Queue(maxsize=1)
        self.overview_thread = None
        self.overview_photo = None
        self.side_mode = tk.StringVar(value="histogram")
        if overview is not None:
            self._side_row = row = tk.Frame(side)
            row.pack(fill="x")
            for value, text in (("histogram", "histogram"), ("overview", overview.name)):
                tk.Radiobutton(row, text=text, value=value, variable=self.side_mode,
                               command=self.switch_side).pack(side="left")
            self.overview_canvas = tk.Canvas(side, width=320, height=220, bg="black", highlightthickness=0)
        self.hist = tk.Canvas(side, width=320, height=220, bg="white")
        self.hist.pack()
        self.log = tk.BooleanVar(value=False)
        tk.Checkbutton(side, text="log counts", variable=self.log, command=self.redraw).pack(anchor="w")
        self.stats = tk.Label(side, justify="left", anchor="w", font=("TkFixedFont", 9))
        self.stats.pack(fill="x", pady=4)

        self.mode = tk.StringVar(value="debayered")
        tk.Radiobutton(side, text="debayered", value="debayered", variable=self.mode,
                       command=self.redraw).pack(anchor="w")
        tk.Radiobutton(side, text="raw", value="raw", variable=self.mode,
                       command=self.redraw).pack(anchor="w")
        self.auto = tk.BooleanVar(value=False)
        tk.Checkbutton(side, text="auto contrast", variable=self.auto, command=self.redraw).pack(anchor="w")
        tk.Checkbutton(side, text="pause", variable=self.paused, command=lambda: (
            self.pause_event.set() if self.paused.get() else self.pause_event.clear())).pack(anchor="w")
        row = tk.Frame(side)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Fit", command=lambda: self.zoom("fit")).pack(side="left")
        tk.Button(row, text="1:1", command=lambda: self.zoom(1.0)).pack(side="left")
        tk.Button(row, text="Clear ROI", command=self.clear_roi).pack(side="left")

        self.exposure = tk.StringVar()
        self.exposure_entry = None
        if camera.state is not None and hasattr(camera.state, "exposure_ms"):
            self.exposure.set(f"{camera.state.exposure_ms:g}")
            row = tk.Frame(side)
            row.pack(fill="x")
            tk.Label(row, text="exposure ms").pack(side="left")
            entry = self.exposure_entry = tk.Entry(row, textvariable=self.exposure, width=9)
            entry.pack(side="left", padx=4)
            entry.bind("<Return>", self.apply_exposure)
        self.status = tk.Label(root, anchor="w")
        self.status.pack(fill="x", padx=6)

        c = self.canvas
        c.bind("<MouseWheel>", lambda e: self.wheel(e, 1 if e.delta > 0 else -1))
        c.bind("<Button-4>", lambda e: self.wheel(e, 1))
        c.bind("<Button-5>", lambda e: self.wheel(e, -1))
        c.bind("<ButtonPress-2>", self.pan_start)
        c.bind("<B2-Motion>", self.pan_move)
        c.bind("<Control-ButtonPress-1>", self.pan_start)
        c.bind("<Control-B1-Motion>", self.pan_move)
        c.bind("<ButtonPress-1>", self.roi_start)
        c.bind("<B1-Motion>", self.roi_move)
        c.bind("<ButtonRelease-1>", self.roi_end)
        c.bind("<Button-3>", lambda e: self.clear_roi())
        c.bind("<Motion>", self.hover)
        c.bind("<Configure>", self.resized)
        root.bind("f", lambda e: self.zoom("fit"))
        root.bind("1", lambda e: self.zoom(1.0))
        root.bind("<Escape>", lambda e: self.clear_roi())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.thread = None
        self.poll_id = None
        if live:
            self.resume()

    # ---- frames -------------------------------------------------------------

    def grab(self):
        while self.running.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            try:
                item = self.camera.capture()
            except Exception as error:  # show it, keep the window
                item = error
                time.sleep(0.5)
            try:
                self.frames.put(item, timeout=1)
            except queue.Full:
                pass

    def grab_overview(self):
        while self.running.is_set() and self.overview_on.is_set():
            try:
                item = self.overview.capture()
            except Exception as error:  # show it, keep going
                item = error
                time.sleep(1)
            try:
                self.overview_frames.get_nowait()  # keep only the latest
            except queue.Empty:
                pass
            self.overview_frames.put(item)

    def poll(self):
        try:
            item = self.frames.get_nowait()
        except queue.Empty:
            item = None
        if isinstance(item, Exception):
            self.status.config(text=f"error: {item}")
        elif item is not None:
            self.show(item)
        try:
            self.draw_overview(self.overview_frames.get_nowait())
        except queue.Empty:
            pass
        self.poll_id = self.root.after(30, self.poll) if self.running.is_set() else None

    def show(self, frame: np.ndarray):
        """Display a new frame (also used by tests)."""
        if self.view is None or self.frame is None or frame.shape[:2] != self.frame.shape[:2]:
            self.view = View(frame.shape, (self.canvas.winfo_width() or 900, self.canvas.winfo_height() or 750))
        self.frame, self.rgb = frame, None
        self.full_scale = full_scale_of(self.camera, frame)
        self.count += 1
        self.redraw()

    @property
    def bayer(self) -> bool:
        return self.frame is not None and is_bayer(self.frame, self.pixel_format)

    def source(self) -> np.ndarray:
        if self.bayer and self.mode.get() == "debayered":
            if self.rgb is None:
                self.rgb = debayer(self.frame)
            return self.rgb
        return self.frame

    # ---- drawing ------------------------------------------------------------

    def redraw(self):
        if self.frame is None:
            return
        from PIL import Image, ImageTk

        c = self.canvas
        c.delete("all")
        rendered = render(self.source(), self.view, self.full_scale, self.auto.get())
        if rendered is not None:
            pixels, size, x, y = rendered
            resample = Image.NEAREST if self.view.scale >= 1 else Image.BILINEAR
            self.photo = ImageTk.PhotoImage(Image.fromarray(pixels).resize(size, resample), master=self.root)
            c.create_image(x, y, image=self.photo, anchor="nw")
        cx, cy = self.view.image_to_canvas(self.frame.shape[1] / 2, self.frame.shape[0] / 2)
        c.create_line(cx, 0, cx, self.view.canvas_height, fill="red", dash=(4, 4))
        c.create_line(0, cy, self.view.canvas_width, cy, fill="red", dash=(4, 4))
        if self.roi is not None:
            x0, y0 = self.view.image_to_canvas(*self.roi[:2])
            x1, y1 = self.view.image_to_canvas(*self.roi[2:])
            c.create_rectangle(x0, y0, x1, y1, outline="yellow", width=2)
        self.draw_histogram()
        rate = self.count / max(time.monotonic() - self.since, 1e-6)
        self.status.config(text=f"{self.frame.shape[1]}x{self.frame.shape[0]} {self.frame.dtype}  "
                                f"zoom {self.view.scale * 100:.0f} %  {rate:.1f} fps")

    def draw_histogram(self):
        h = self.hist
        h.delete("all")
        region = self.roi or self.view.region()
        data = histogram(self.frame, region, self.bayer, self.full_scale)
        width, height, margin = int(h["width"]), int(h["height"]), 16
        lines = [f"{'ROI' if self.roi else 'view'} {region[2] - region[0]}x{region[3] - region[1]}"]
        top = max((counts.max() for counts, _, _ in data.values()), default=1)
        for name, (counts, mean, peak) in data.items():
            values = np.log1p(counts) / np.log1p(top) if self.log.get() else counts / max(top, 1)
            xs = np.linspace(0, width, len(values))
            ys = height - margin - values * (height - 2 * margin)
            h.create_line(*np.column_stack([xs, ys]).ravel(), fill=CHANNEL_COLORS[name])
            lines.append(f"{name}: mean {mean:8.1f}  max {peak}")
        for label, value in self.markers.items():
            x = value / self.full_scale * width
            h.create_line(x, 0, x, height - margin, dash=(4, 3), fill="#888888")
            h.create_text(x + 3, 2, text=label, anchor="nw", fill="#555555")
        h.create_text(2, height - 2, text="0", anchor="sw")
        h.create_text(width - 2, height - 2, text=str(self.full_scale), anchor="se")
        self.stats.config(text="\n".join(lines))

    def draw_overview(self, item):
        """The overview image (or its error), fitted into the histogram's place."""
        from PIL import Image, ImageTk

        c = self.overview_canvas
        c.delete("all")
        width, height = int(c["width"]), int(c["height"])
        if isinstance(item, Exception):
            c.create_text(4, 4, text=f"{self.overview.name}: {item}", anchor="nw", fill="red",
                          width=width - 8)
            return
        pixel_format = getattr(self.overview.config, "pixel_format", None)
        image = debayer(item) if is_bayer(item, pixel_format) else item
        pixels = to_uint8(image, full_scale_of(self.overview, item), auto=False)
        picture = Image.fromarray(pixels)
        picture.thumbnail((width, height))
        self.overview_photo = ImageTk.PhotoImage(picture, master=self.root)
        c.create_image(width // 2, height // 2, image=self.overview_photo)
        c.create_text(4, height - 4, text=f"{self.overview.name} {item.shape[1]}x{item.shape[0]}",
                      anchor="sw", fill="yellow")

    def switch_side(self):
        """Histogram or overview image (the radio buttons above them)."""
        if self.side_mode.get() == "overview":
            self.hist.pack_forget()
            self.overview_canvas.pack(after=self._side_row)
            self.overview_on.set()
            self._start_overview()
        else:
            self.overview_on.clear()
            self.overview_canvas.pack_forget()
            self.hist.pack(after=self._side_row)

    def _start_overview(self):
        alive = self.overview_thread is not None and self.overview_thread.is_alive()
        if self.overview_on.is_set() and self.running.is_set() and not alive:
            self.overview_thread = threading.Thread(target=self.grab_overview, daemon=True)
            self.overview_thread.start()

    # ---- interaction ----------------------------------------------------------

    def zoom(self, how):
        if self.view is None:
            return
        if how == "fit":
            self.view.fit()
        else:
            self.view.set_scale(how)
        self.redraw()

    def wheel(self, event, direction):
        if self.view is not None:
            self.view.zoom_at(event.x, event.y, 1.25 if direction > 0 else 0.8)
            self.redraw()

    def pan_start(self, event):
        self.drag = ("pan", event.x, event.y)

    def pan_move(self, event):
        if self.view is not None and self.drag and self.drag[0] == "pan":
            self.view.pan(event.x - self.drag[1], event.y - self.drag[2])
            self.drag = ("pan", event.x, event.y)
            self.redraw()

    def image_point(self, event):
        x, y = self.view.canvas_to_image(event.x, event.y)
        return (int(np.clip(x, 0, self.frame.shape[1])), int(np.clip(y, 0, self.frame.shape[0])))

    def roi_start(self, event):
        if self.frame is not None and self.roi_enabled:
            self.drag = ("roi", *self.image_point(event))

    def roi_move(self, event):
        if self.frame is not None and self.drag and self.drag[0] == "roi":
            x, y = self.image_point(event)
            x0, y0 = self.drag[1:]
            self.roi = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self.redraw()

    def roi_end(self, event):
        if self.drag and self.drag[0] == "roi" and self.roi is not None:
            x0, y0, x1, y1 = self.roi
            if x1 - x0 < 2 or y1 - y0 < 2:
                self.roi = None
            self.redraw()
        self.drag = None

    def set_roi(self, roi):
        self.roi = roi
        self.redraw()

    def clear_roi(self):
        if self.roi_enabled:
            self.set_roi(None)

    def hover(self, event):
        if self.frame is None:
            return
        x, y = self.view.canvas_to_image(event.x, event.y)
        if 0 <= x < self.frame.shape[1] and 0 <= y < self.frame.shape[0]:
            value = self.frame[int(y), int(x)]
            text = self.status.cget("text").split("  |")[0]
            self.status.config(text=f"{text}  |  x {int(x)} y {int(y)}: {value}")

    def resized(self, event):
        if self.view is not None:
            self.view.resize((event.width, event.height))
            self.redraw()

    def apply_exposure(self, _event=None):
        try:
            self.camera.set(exposure_ms=float(self.exposure.get()))
        except (TypeError, ValueError) as error:
            self.status.config(text=str(error))

    @property
    def live(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        """Stop the live capture; the window stays (e.g. for a script's own frames via show())."""
        self.running.clear()
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        if self.thread is not None:
            self.thread.join(timeout=60)
        while not self.frames.empty():  # no old live frame after the script's own
            self.frames.get_nowait()

    def resume(self):
        """Start (or restart after stop()) the live capture."""
        if self.live:
            return
        self.running.set()
        self.thread = threading.Thread(target=self.grab, daemon=True)
        self.thread.start()
        self._start_overview()  # not joined in stop(): it ends after its current (slow) capture
        self.poll()

    def close(self):
        self.running.clear()
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)  # no callback into a destroyed window
            self.poll_id = None
        self.root.destroy()


def main():
    """``spec-live [camera.toml [overview.toml]]``: the first camera live, the second's
    image on request instead of the histogram (default: camera and overview_camera)."""
    from ..device import Device
    from ..setup import Setup
    from .gui import live_view

    tomls = [Path(arg) for arg in sys.argv[1:3]] or DEFAULT_TOMLS
    cameras = [Device.from_toml(toml) for toml in tomls]
    with Setup(*cameras):
        live_view(cameras[0], overview=cameras[1] if len(cameras) > 1 else None)


if __name__ == "__main__":
    main()
