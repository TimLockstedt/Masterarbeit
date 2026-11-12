"""Illumination patterns for the rspanel display: uint8 RGB arrays (H, W, 3).

``shape`` is the display shape, ``(height, width)`` or ``panel.shape``.
Coordinates: ``x`` is the column, ``y`` the row; pixel centres are integers.
A pixel is lit when ``inner <= r < outer`` (``r`` = distance of its centre from
``center``). Angles are ``atan2(y - cy, x - cx)`` in degrees, from +x towards
+y; segment ``k`` of ``n`` (width ``w = 360/n``) is centred on ``offset + k*w`` and
covers ``[offset + (k - 1/2)*w, offset + (k + 1/2)*w)``.
"""

import re
from pathlib import Path

import numpy as np

GREEN = (0, 255, 0)


def _grid(shape, center):
    height, width = shape[:2]
    y, x = np.indices((height, width), dtype=float)
    return x - center[0], y - center[1]


def colorize(mask: np.ndarray, color=GREEN) -> np.ndarray:
    """Boolean mask (..., H, W) -> uint8 RGB (..., H, W, 3)."""
    return mask[..., None] * np.asarray(color, dtype=np.uint8)


def ring_mask(shape, center, outer, inner=0.0) -> np.ndarray:
    if not 0 <= inner < outer:
        raise ValueError("require 0 <= inner < outer")
    dx, dy = _grid(shape, center)
    r = np.hypot(dx, dy)
    mask = (r >= inner) & (r < outer)
    if not mask.any():
        raise ValueError("ring contains no pixels")
    return mask


def ring(shape, center, outer, inner=0.0, color=GREEN) -> np.ndarray:
    """Filled disc (inner=0) or annulus."""
    return colorize(ring_mask(shape, center, outer, inner), color)


def segment_masks(shape, center, outer, inner, count, angle_offset_deg=0.0) -> np.ndarray:
    """``count`` boolean masks (count, H, W): the ring cut into equal angular segments."""
    if count < 1:
        raise ValueError("count must be >= 1")
    mask = ring_mask(shape, center, outer, inner)
    dx, dy = _grid(shape, center)
    half_width = 180.0 / count  # segment k is centred on angle_offset_deg + k * width
    angle = np.mod(np.degrees(np.arctan2(dy, dx)) - angle_offset_deg + half_width, 360.0)
    index = np.minimum((angle * count / 360.0).astype(int), count - 1)
    masks = np.stack([mask & (index == k) for k in range(count)])
    empty = [k for k in range(count) if not masks[k].any()]
    if empty:
        raise ValueError(f"segments contain no pixels: {empty}")
    return masks


def canonical(shape, center, outer, inner, count, color=GREEN, angle_offset_deg=0.0):
    """Canonical coding: one pattern per segment, pattern k lights segment k: (count, H, W, 3)."""
    return colorize(segment_masks(shape, center, outer, inner, count, angle_offset_deg), color)


def hadamard(shape, center, outer, inner, count, color=GREEN, angle_offset_deg=0.0):
    """Hadamard-coded segments (Sylvester): pattern i lights segment k if H[i, k] = +1."""
    if count < 2 or count & (count - 1):
        raise ValueError("Hadamard count must be a power of two >= 2")
    from scipy.linalg import hadamard as sylvester

    masks = segment_masks(shape, center, outer, inner, count, angle_offset_deg)
    rows = sylvester(count) > 0
    return colorize(np.stack([masks[row].any(axis=0) for row in rows]), color)


# SLI codings by name (the name is also what the toml and the metadata use)
CODINGS = {"canonical": canonical, "hadamard": hadamard}


def coded(coding, shape, center, outer, inner, count, color=GREEN, angle_offset_deg=0.0):
    """The patterns of SLI ``coding`` ("canonical" or "hadamard"): (count, H, W, 3)."""
    if coding not in CODINGS:
        raise ValueError(f"unknown coding {coding!r}; use {', '.join(CODINGS)}")
    return CODINGS[coding](shape, center, outer, inner, count, color, angle_offset_deg)


def load_image(path, shape) -> np.ndarray:
    """Load an 8-bit grayscale or RGB image of exactly the display size."""
    from PIL import Image

    path = Path(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or image.mode not in ("L", "RGB"):
            raise ValueError(f"{path}: need a single-frame 8-bit L or RGB image")
        array = np.asarray(image)
    if array.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"{path}: size {array.shape[:2]} differs from display {shape[:2]}")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return np.ascontiguousarray(array, dtype=np.uint8)


def load_folder(folder, shape) -> np.ndarray:
    """All images of a folder in natural order (img2 before img10): (N, H, W, 3)."""
    def natural(path):
        return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]

    files = sorted((p for p in Path(folder).iterdir() if not p.name.startswith(".")), key=natural)
    if not files:
        raise ValueError(f"{folder}: no images")
    return np.stack([load_image(p, shape) for p in files])
