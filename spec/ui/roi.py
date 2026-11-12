"""Draw a polygon region of interest on an image (tk).

    polygon_px = select_polygon(overview_image)        # None if cancelled
    polygon_mm = to_stage(polygon_px, calibration)     # overview pixel -> stage mm

Left click adds a point, right click or Backspace removes the last point,
Enter or "Start" accepts (at least 3 points), Escape or closing cancels.
"""

import numpy as np

MAX_SIZE = (1200, 800)


def display_image(image: np.ndarray) -> np.ndarray:
    """Any camera frame -> uint8 gray or RGB for display (robust contrast)."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    low, high = np.percentile(image, (0.5, 99.5))
    scaled = (image.astype(np.float64) - low) / max(high - low, 1e-9)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def fit_affine(pixels, stage_mm) -> np.ndarray:
    """Least-squares 2x3 affine matrix mapping pixel (x, y) to stage (x, y) mm."""
    pixels, stage_mm = np.asarray(pixels, float), np.asarray(stage_mm, float)
    if pixels.shape != stage_mm.shape or pixels.shape[0] < 3 or pixels.shape[1] != 2:
        raise ValueError("need at least 3 matching (x, y) point pairs")
    design = np.column_stack([pixels, np.ones(len(pixels))])
    solution, *_ = np.linalg.lstsq(design, stage_mm, rcond=None)
    return solution.T


def to_stage(points_px, affine) -> np.ndarray:
    """Apply a 2x3 affine matrix (e.g. from a calibration toml) to (N, 2) points."""
    affine = np.asarray(affine, float)
    if affine.shape != (2, 3):
        raise ValueError("affine matrix must be 2x3")
    points = np.asarray(points_px, float)
    return points @ affine[:, :2].T + affine[:, 2]


class PolygonEditor:
    """Polygon state in image pixels; independent of tk so it can be tested."""

    def __init__(self, scale: float):
        self.scale = scale  # canvas px per image px
        self.points: list[tuple[float, float]] = []

    def add(self, canvas_x, canvas_y):
        self.points.append((canvas_x / self.scale, canvas_y / self.scale))

    def undo(self):
        if self.points:
            self.points.pop()

    def canvas_points(self) -> list[float]:
        return [c * self.scale for point in self.points for c in point]

    @property
    def complete(self) -> bool:
        return len(self.points) >= 3


def select_polygon(image, title="Draw the region of interest", max_size=MAX_SIZE, on_ready=None):
    """Show ``image``; return the drawn polygon as (N, 2) image pixels or None.

    ``on_ready(root, canvas)`` is called once the window exists (for tests)."""
    import tkinter as tk

    from PIL import Image, ImageTk

    shown = Image.fromarray(display_image(image))
    scale = min(max_size[0] / shown.width, max_size[1] / shown.height, 1.0)
    shown = shown.resize((max(1, round(shown.width * scale)), max(1, round(shown.height * scale))))
    editor = PolygonEditor(scale)
    result = {"polygon": None}

    root = tk.Tk()
    root.title(title)
    photo = ImageTk.PhotoImage(shown, master=root)
    canvas = tk.Canvas(root, width=shown.width, height=shown.height, cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor="nw")
    bar = tk.Frame(root)
    bar.pack(fill="x")
    status = tk.Label(bar, anchor="w")
    status.pack(side="left", fill="x", expand=True, padx=8)
    start = tk.Button(bar, text="Start", state="disabled")
    start.pack(side="right", padx=4, pady=4)
    tk.Button(bar, text="Cancel", command=root.destroy).pack(side="right", pady=4)

    def redraw():
        canvas.delete("roi")
        points = editor.canvas_points()
        if len(points) >= 4:
            canvas.create_line(*points, *(points[:2] if editor.complete else []),
                               fill="yellow", width=2, tags="roi")
        for x, y in zip(points[::2], points[1::2]):
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, outline="yellow", tags="roi")
        start.config(state="normal" if editor.complete else "disabled")
        status.config(text=f"{len(editor.points)} points - left click: add, right click: undo, Enter: start")

    def accept(_event=None):
        if editor.complete:
            result["polygon"] = np.array(editor.points)
            root.destroy()

    canvas.bind("<Button-1>", lambda e: (editor.add(e.x, e.y), redraw()))
    canvas.bind("<Button-3>", lambda e: (editor.undo(), redraw()))
    root.bind("<BackSpace>", lambda e: (editor.undo(), redraw()))
    root.bind("<Return>", accept)
    root.bind("<Escape>", lambda e: root.destroy())
    start.config(command=accept)
    redraw()
    if on_ready is not None:
        root.after(0, on_ready, root, canvas)
    root.mainloop()
    return result["polygon"]


def roi_tiles(setup, tile_size_mm, overlap_mm=0.0, camera="overview_camera"):
    """Overview image -> drawn polygon -> stage tiles.

    Needs ``config/<instrument>/calibration/overview_to_stage.toml`` with
    ``matrix = [[a, b, c], [d, e, f]]``. Returns (tiles, polygon_mm).
    """
    from ..grid import tiles_in_polygon

    if "overview_to_stage" not in setup.calibration:
        raise RuntimeError(
            "no overview_to_stage calibration in the instrument folder "
            "(calibration/overview_to_stage.toml)"
        )
    if camera not in setup:
        raise RuntimeError(f"the ROI needs the {camera!r} device in the setup")
    if "panel" in setup:  # light the sample for the overview image
        previous = setup.panel.state.pattern
        setup.panel.set(pattern=np.full(setup.panel.shape, 255, np.uint8))
        image = setup[camera].capture()
        setup.panel.set(pattern=previous)
    else:
        image = setup[camera].capture()
    polygon_px = select_polygon(image, title="Draw the region to measure, then Start")
    if polygon_px is None:
        raise SystemExit("ROI selection cancelled")
    polygon_mm = to_stage(polygon_px, setup.calibration["overview_to_stage"]["matrix"])
    return tiles_in_polygon(polygon_mm, tile_size_mm, overlap_mm), polygon_mm
