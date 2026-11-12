"""One simple window for scripts: camera image, buttons, text, optional plot (tk).

    from spec.ui.gui import Window

    with Window("focus calibration", setup.camera, plot=("z (mm)", "score")) as window:
        window.button("Up", lambda: ..., key="<Up>")
        window.status(lambda: f"z = {setup.stage_z.read().position_mm:.4f} mm")
        if not window.wait("Start"):     # live view + controls until Start/Enter
            raise SystemExit("cancelled")
        for ...:                         # the script's own frames and results
            window.show(frame, "text")
            window.plot({"coarse": [(z, score), ...]}, mark=best_z)
        window.finish("done")            # stays open until it is closed

Layout: the image on the left (zoom, pan, ROI, histogram; spec.ui.live), the
plot under the histogram on the right, a text line and a row of buttons below.
``plot=(x label, y label)`` adds the plot, ``markers={label: counts}`` dashed
lines in the histogram (e.g. a target and the saturation level).

The window has two modes:

- ``wait()``: interactive. The camera runs live, the buttons, keys, exposure
  field and ROI drawing work, ``status()`` is updated twice a second. Ends with
  the OK button / Enter (True) or when the window is closed (False).
- everything else: the script runs. No live capture (the script can use the
  camera), controls are off, the window shows what the script gives it
  (``show``, ``plot``, ``text``); every call also handles the window's events,
  so no mainloop is needed.

With a camera that has an exposure time, wait() adds an "Auto exposure"
button (``adjust_exposure``: gain 0, then the exposure so the 99.9th
percentile of the raw pixels in the ROI, else the whole frame, is at
``exposure_target`` of full scale).

After the window has been closed every call does nothing (``window.closed``),
so a script carries on without it.
"""

from collections.abc import Callable

import numpy as np

from .live import draw_scan, full_scale_of

EXPOSURE_TARGET = 0.75  # usual: brightest pixels at 70-80 % of full scale
SATURATION = 0.98  # fraction of full scale counted as saturated
MIN_EXPOSURE_MS = 0.1


def crop(frame: np.ndarray, roi) -> np.ndarray:
    """``frame[y0:y1, x0:x1]`` (start on even pixels: keeps the Bayer phase); None: all."""
    if roi is None:
        return frame
    x0, y0, x1, y1 = (int(v) for v in roi)
    return frame[y0 - y0 % 2 : y1, x0 - x0 % 2 : x1]


def adjust_exposure(camera, target=EXPOSURE_TARGET, roi=None, on_frame=None, steps=10):
    """Gain 0, then scale the exposure until the 99.9th percentile of the raw pixels
    (``roi``, else the whole frame) is at ``target`` of full scale.

    ``on_frame(frame, text)`` is called with every image. Returns (ok, text):
    ok is False at an exposure limit or if it did not settle in ``steps`` images."""
    low, high = camera.limits.get("exposure_ms", (MIN_EXPOSURE_MS, 10_000))
    low = max(low, MIN_EXPOSURE_MS)
    if hasattr(camera.state, "gain"):
        camera.set(gain=0)
    for _ in range(steps):
        exposure = camera.state.exposure_ms
        frame = camera.capture()
        full_scale = full_scale_of(camera, frame)
        level = float(np.percentile(crop(frame, roi), 99.9))
        text = f"{exposure:g} ms, gain 0: 99.9th percentile at {level / full_scale:.0%} of full scale"
        if on_frame is not None:
            on_frame(frame, f"auto exposure: {text} (target {target:.0%})")
        saturated = level >= SATURATION * full_scale  # true level unknown: halve
        factor = 0.5 if saturated else min(target * full_scale / max(level, 1.0), 10.0)
        wanted = float(np.clip(exposure * factor, low, high))
        if not saturated and abs(factor - 1) < 0.05:
            return True, text
        if wanted == exposure:
            limit = "maximum" if factor > 1 else "minimum"
            return False, f"at the {limit} exposure: {text}"
        camera.set(exposure_ms=wanted)
    return False, f"did not settle in {steps} images: {text}"


class Window:
    """See the module docstring. ``camera``: None for a window without image.

    ``overview``: a further connected camera (e.g. the overview camera); its image
    can be shown instead of the histogram (live in wait(), see spec.ui.live)."""

    def __init__(self, title, camera=None, *, overview=None, plot=None, markers=None,
                 exposure_target=EXPOSURE_TARGET):
        import tkinter as tk

        from .live import LiveView

        self.tk, self.camera, self.exposure_target = tk, camera, exposure_target
        self.closed = False
        self.waiting = False  # in wait(): live and controls on
        self.busy = False  # a control is running (e.g. auto exposure)
        self._ok = False
        self._text = ""
        self._status: Callable[[], str] | None = None
        self._status_id = None
        self._controls = []  # widgets that are only enabled in wait()
        self._keys = {}  # key -> command, only active in wait()

        self.root = root = tk.Tk()
        if camera is not None:
            self.view = LiveView(root, camera, live=False, overview=overview)
            self.view.roi_enabled = False
            self.view.markers = dict(markers or {})  # {label: counts}: lines in the histogram
            side = self.view.side
            if self.view.exposure_entry is not None:
                self._controls.append(self.view.exposure_entry)
        else:
            self.view = None
            side = tk.Frame(root)
            side.pack(fill="both", expand=True, padx=6, pady=6)
        self.side = side  # right column: histogram (with camera), choice(), number(), plot
        root.title(title)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._plot_labels = plot
        self._plot = None
        if plot is not None:
            tk.Label(side, text=f"{plot[1]} vs {plot[0]}", anchor="w").pack(fill="x", pady=(8, 0))
            self._plot = tk.Canvas(side, width=320, height=220, bg="white")
            self._plot.pack()

        self._label = tk.Label(root, anchor="w", justify="left", font="TkFixedFont")
        self._label.pack(fill="x", padx=8)
        self.bar = tk.Frame(root)
        self.bar.pack(fill="x", padx=4, pady=4)
        self._ok_button = tk.Button(self.bar, text="OK", command=self._accept)
        self._ok_button.pack(side="right", padx=2)
        self._ok_button.pack_forget()  # shown in wait(ok=...)
        self._auto_button = None  # added in the first wait()
        self._set_controls(False)
        self.update()

    # ---- building ------------------------------------------------------------------

    def button(self, text, command, key=None):
        """A button (and optional key, e.g. "<Up>"); both only work in wait()."""
        button = self.tk.Button(self.bar, text=text, command=lambda: self._run(command))
        button.pack(side="left", padx=2)
        self._controls.append(button)
        if key is not None:
            self.key(key, command)
        button.config(state="normal" if self.waiting else "disabled")
        return button

    def key(self, key, command):
        """A key (e.g. "<Shift-Left>") without a button; only works in wait(), not while
        typing in a field."""
        self._keys[key] = command
        self.root.bind(key, lambda event: self._key(event, command))

    def entry(self, label, value="", width=8):
        """A labelled text field (StringVar); only editable in wait()."""
        variable = self.tk.StringVar(value=str(value))
        self.tk.Label(self.bar, text=label).pack(side="left", padx=(6, 0))
        widget = self.tk.Entry(self.bar, textvariable=variable, width=width)
        widget.pack(side="left")
        self._controls.append(widget)
        widget.config(state="normal" if self.waiting else "disabled")
        return variable

    def slider(self, label, low, high, value, command, resolution=0.1):
        """A labelled slider with a field to type the value; only works in wait().

        ``command(value)`` runs when the slider is released or Enter is pressed in
        the field (a typed value outside ``low..high`` is refused). Returns
        ``show(value)``: moves slider and field without calling ``command``, unless
        the user is dragging or typing (e.g. to follow a value changed elsewhere)."""
        tk = self.tk
        number = tk.DoubleVar(value=value)
        text = tk.StringVar(value=f"{value:g}")
        state = {"dragging": False}
        tk.Label(self.bar, text=label).pack(side="left", padx=(6, 0))
        scale = tk.Scale(self.bar, from_=low, to=high, resolution=resolution, orient="horizontal",
                         variable=number, showvalue=False, length=160,
                         command=lambda v: text.set(f"{float(v):g}"))  # follows while dragging
        scale.pack(side="left")
        field = tk.Entry(self.bar, textvariable=text, width=6)
        field.pack(side="left")

        def apply(new):
            number.set(new)
            text.set(f"{new:g}")
            command(new)

        def pressed(_event):
            state["dragging"] = True

        def released(_event):
            state["dragging"] = False
            self._run(lambda: apply(number.get()))

        def typed(_event):
            def check():
                try:
                    new = float(text.get())
                except ValueError:
                    text.set(f"{number.get():g}")
                    raise ValueError(f"{label}: {text.get()!r} is not a number") from None
                if not low <= new <= high:
                    text.set(f"{number.get():g}")
                    raise ValueError(f"{label} must be within {low:g}..{high:g}, got {new:g}")
                apply(new)

            self._run(check)

        scale.bind("<ButtonPress-1>", pressed)
        scale.bind("<ButtonRelease-1>", released)
        field.bind("<Return>", typed)
        for widget in (scale, field):
            self._controls.append(widget)
            widget.config(state="normal" if self.waiting else "disabled")

        def show(new):
            if self.closed or state["dragging"] or self.root.focus_get() is field:
                return
            number.set(new)
            text.set(f"{new:g}")

        return show

    def section(self, title):
        """A titled group in the right column, as ``parent`` for choice()/number()."""
        frame = self.tk.LabelFrame(self.side, text=title, padx=4, pady=2)
        frame.pack(fill="x", pady=(6, 0))
        return frame

    def _row(self, parent, label):
        row = self.tk.Frame(parent or self.side)
        row.pack(fill="x", pady=1)
        self.tk.Label(row, text=label, width=12, anchor="w").pack(side="left")
        return row

    def choice(self, label, options, value, command, parent=None):
        """A labelled drop-down in the right column; ``command(option)`` when one is
        picked (only in wait()). Returns its StringVar (set() it to follow changes)."""
        variable = self.tk.StringVar(value=value)
        menu = self.tk.OptionMenu(self._row(parent, label), variable, *options,
                                  command=lambda option: self._run(lambda: command(option)))
        menu.pack(side="left", fill="x", expand=True)
        self._controls.append(menu)
        menu.config(state="normal" if self.waiting else "disabled")
        return variable

    def number(self, label, value, command, low, high, step=1, integer=True, parent=None):
        """A labelled number field with arrows in the right column; ``command(value)`` on
        an arrow or Enter, after checking ``low..high`` (only in wait()). Returns its
        StringVar (set() it to follow changes; ``command`` is not called then); the
        field is its ``.widget`` (e.g. ``.widget.config(increment=...)``)."""
        tk = self.tk
        cast = int if integer else float
        variable = tk.StringVar(value=f"{value:g}")
        last = {"value": cast(value)}

        def apply():
            def check():
                text = variable.get()
                try:
                    new = float(text)
                    if integer and not new.is_integer():
                        raise ValueError
                except ValueError:
                    variable.set(f"{last['value']:g}")
                    raise ValueError(f"{label}: {text!r} is not {'an integer' if integer else 'a number'}") from None
                new = cast(new)
                if not low <= new <= high:
                    variable.set(f"{last['value']:g}")
                    raise ValueError(f"{label} must be within {low:g}..{high:g}, got {new:g}")
                command(new)
                last["value"] = new

            self._run(check)

        box = tk.Spinbox(self._row(parent, label), from_=low, to=high, increment=step, width=8,
                         textvariable=variable, command=apply)
        variable.set(f"{value:g}")  # the Spinbox resets it to from_
        box.pack(side="left")
        box.bind("<Return>", lambda _event: apply())
        self._controls.append(box)
        box.config(state="normal" if self.waiting else "disabled")
        variable.widget = box
        return variable

    def marked_slider(self, label, low, high, value, command, marks=None, resolution=0.001,
                      parent=None, width=320):
        """A slider drawn on a canvas with labelled marks (e.g. a calibrated position),
        in the right column; only works in wait().

        Click or drag the handle; ``command(value)`` runs when the mouse is released
        (rounded to ``resolution``). ``marks``: {label: value}, red ticks. Returns
        ``show(value)``: moves the handle without calling ``command`` (not while
        dragging), e.g. to follow the real position."""
        tk = self.tk
        frame = tk.Frame(parent or self.side)
        frame.pack(fill="x", pady=(4, 0))
        tk.Label(frame, text=label, anchor="w").pack(fill="x")
        height, margin = 50, 12
        canvas = tk.Canvas(frame, width=width, height=height, bg="white", highlightthickness=0)
        canvas.pack()
        state = {"value": float(value), "dragging": False}
        digits = max(0, -int(np.floor(np.log10(resolution)))) if resolution < 1 else 0

        def to_x(v):
            return margin + (np.clip(v, low, high) - low) / (high - low) * (width - 2 * margin)

        def to_value(x):
            v = low + (x - margin) / (width - 2 * margin) * (high - low)
            return float(np.clip(round(v / resolution) * resolution, low, high))

        def draw():
            canvas.delete("all")
            y = 30
            canvas.create_line(margin, y, width - margin, y, width=3, fill="#999999")
            canvas.create_text(margin, height - 2, text=f"{low:.{digits}f}", anchor="sw", font=("TkDefaultFont", 7))
            canvas.create_text(width - margin, height - 2, text=f"{high:.{digits}f}", anchor="se",
                               font=("TkDefaultFont", 7))
            for name, mark in (marks or {}).items():
                if low <= mark <= high:
                    x = to_x(mark)
                    canvas.create_line(x, y - 9, x, y + 9, fill="red", width=2)
                    canvas.create_text(x, y - 10, text=f"{name} {mark:.{digits}f}", anchor="s", fill="red",
                                       font=("TkDefaultFont", 7))
            x = to_x(state["value"])
            canvas.create_rectangle(x - 4, y - 7, x + 4, y + 7, fill="#3060c0", outline="")
            canvas.create_text(width / 2, height - 2, text=f"{state['value']:.{digits}f}", anchor="s")

        def drag(event):
            if not self.waiting or self.busy:
                return
            state["dragging"] = True
            state["value"] = to_value(event.x)
            draw()

        def release(event):
            if not state["dragging"]:
                return
            state["dragging"] = False
            state["value"] = to_value(event.x)
            draw()
            self._run(lambda: command(state["value"]))

        canvas.bind("<ButtonPress-1>", drag)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)
        self._controls.append(canvas)
        canvas.config(state="normal" if self.waiting else "disabled")
        draw()

        def show(new):
            if self.closed or state["dragging"]:
                return
            state["value"] = float(new)
            draw()

        return show

    def status(self, function: Callable[[], str] | None):
        """``function()`` -> text shown under the text line, twice a second in wait()."""
        self._status = function

    # ---- script mode ---------------------------------------------------------------

    @property
    def roi(self):
        """The ROI drawn in wait() (x0, y0, x1, y1) or None."""
        return None if self.view is None else self.view.roi

    def text(self, text: str):
        """The text line (e.g. what the script does, a result, why something failed)."""
        if self.closed:
            return
        self._text = text
        self._render(self._status_text() if self.waiting else "")
        self.update()

    def show(self, frame: np.ndarray, text: str | None = None):
        """Show the script's own image (and text)."""
        if self.closed:
            return
        if self.view is not None:
            self.view.show(frame)
        if text is not None:
            self._text = text
            self._render("")
        self.update()

    def plot(self, series: dict, mark: float | None = None, curves: dict | None = None):
        """{name: [(x, y), ...]} as points, the latest of each circled, ``mark`` dashed,
        ``curves`` {name: [(x, y), ...]} as lines (e.g. a fit)."""
        if self.closed or self._plot is None:
            return
        draw_scan(self._plot, series, mark, x_label=self._plot_labels[0], y_label=self._plot_labels[1],
                  curves=curves)
        self.update()

    def update(self):
        if self.closed:
            return
        try:
            self.root.update()
        except self.tk.TclError:  # destroyed meanwhile
            self.closed = True

    # ---- interactive mode ------------------------------------------------------------

    def wait(self, ok: str | None = "OK") -> bool:
        """Live view and controls until ``ok`` (button or Enter): True; closed: False.

        ``ok=None``: no OK button, until the window is closed (returns False)."""
        if self.closed:
            return False
        self._ok, self.waiting = False, True
        if self._auto_button is None and self.camera is not None and hasattr(self.camera.state, "exposure_ms"):
            self._auto_button = self.button("Auto exposure", self.auto_exposure)
        if ok is not None:
            self._ok_button.config(text=f"{ok} (Enter)")
            self._ok_button.pack(side="right", padx=2)
            self.root.bind("<Return>", self._enter)
        self._set_controls(True)
        if self.view is not None:
            if self.view.exposure_entry is not None:  # the script may have changed it
                self.view.exposure.set(f"{self.camera.state.exposure_ms:g}")
            self.view.roi_enabled = True
            self.view.resume()
        self._tick()
        self.root.mainloop()
        if self.closed:
            return False
        self.waiting = False
        if self._status_id is not None:
            self.root.after_cancel(self._status_id)
            self._status_id = None
        if self.view is not None:
            self.view.stop()
            self.view.roi_enabled = False
        self._set_controls(False)
        self._ok_button.pack_forget()
        self.root.unbind("<Return>")
        self._render("")
        self.update()
        return self._ok

    def auto_exposure(self):
        """The "Auto exposure" button: see ``adjust_exposure``."""
        live = self.view.live
        self.view.stop()  # capture here, not in the live thread
        try:
            ok, text = adjust_exposure(self.camera, self.exposure_target, self.view.roi, self.show)
            text = f"auto exposure: {text}"
        except Exception as error:
            text = f"auto exposure failed: {error}"
        print(text)
        if self.closed:
            return
        self.view.exposure.set(f"{self.camera.state.exposure_ms:g}")
        self.text(text)
        if live:
            self.view.resume()

    # ---- end -------------------------------------------------------------------------

    def finish(self, text: str):
        """Show ``text`` and keep the window open (no live view) until it is closed."""
        if self.closed:
            return
        self.text(f"{text}   (close the window to end)")
        self.root.mainloop()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._status_id is not None:
            self.root.after_cancel(self._status_id)
        if self.view is not None:
            self.view.stop()
        self.root.destroy()  # also ends a running mainloop

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---- internals -------------------------------------------------------------------

    def _run(self, command):
        """Run a control's command; one at a time, others are off meanwhile."""
        if self.busy or not self.waiting:
            return
        self.busy = True
        self._set_controls(False)
        try:
            command()
        except Exception as error:  # show it, keep the window
            self.text(f"error: {error}")
        finally:
            self.busy = False
            if not self.closed and self.waiting:
                self._set_controls(True)

    def _key(self, event, command):
        if self.waiting and not isinstance(event.widget, self.tk.Entry):  # typing in a field
            self._run(command)

    def _enter(self, event):
        if not isinstance(event.widget, self.tk.Entry):  # Enter in a field only applies it
            self._accept()

    def _accept(self):
        if self.busy or not self.waiting:
            return
        self._ok = True
        self.root.quit()  # leaves mainloop; the window stays

    def _set_controls(self, enabled):
        for widget in self._controls:
            widget.config(state="normal" if enabled else "disabled")
        self._ok_button.config(state="normal" if enabled else "disabled")

    def _status_text(self) -> str:
        if self._status is None:
            return ""
        try:
            return self._status()
        except Exception as error:  # show it, keep the window
            return f"error: {error}"

    def _render(self, status: str):
        self._label.config(text="\n".join(t for t in (self._text, status) if t))

    def _tick(self):
        if self.closed or not self.waiting:
            return
        if not self.busy:
            self._render(self._status_text())
        self._status_id = self.root.after(500, self._tick)


OVERVIEW_CAMERA = "overview_camera"


def add_overview_option(parser):
    """``--overview CAMERA`` / ``--no-overview``: the camera whose image the window can
    show instead of the histogram (scripts with a live phase)."""
    parser.add_argument("--overview", default=OVERVIEW_CAMERA, metavar="CAMERA",
                        help=f"camera shown instead of the histogram on request (default: {OVERVIEW_CAMERA})")
    parser.add_argument("--no-overview", dest="overview", action="store_const", const=None,
                        help="no overview camera (not connected)")


def live_view(camera, title=None, overview=None):
    """Show a connected camera live until the window is closed (``overview``: see Window)."""
    with Window(title or f"{camera.name} live", camera, overview=overview) as window:
        window.wait(None)
