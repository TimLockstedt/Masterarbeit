"""Green pixels of raw Bayer frames (e.g. BayerRG12 from the main camera).

Each 2x2 block has two green sites on one diagonal. ``green_pixels`` returns
all of them as a flat array (for means and percentiles); ``green_image`` adds
the two sites of each block into one half-resolution image on a regular grid
(for filters such as the Laplacian). Mono formats: all pixels.
"""

import numpy as np

# (row, column) of the green sites in each 2x2 block, by Bayer pattern.
GREEN_SITES = {
    "RG": ((0, 1), (1, 0)),
    "BG": ((0, 1), (1, 0)),
    "GR": ((0, 0), (1, 1)),
    "GB": ((0, 0), (1, 1)),
}


def _sites(pixel_format: str):
    if not pixel_format.startswith("Bayer"):
        return None
    return GREEN_SITES[pixel_format[5:7]]


def green_pixels(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """1-D array of the green pixels (all pixels if not Bayer)."""
    sites = _sites(pixel_format)
    if sites is None:
        return frame.ravel()
    return np.concatenate([frame[r::2, c::2].ravel() for r, c in sites])


def green_image(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """Sum of the two green sites of each 2x2 block: (H/2, W/2) uint16 (frame itself if not Bayer).

    A 12-bit frame gives values up to 8190."""
    sites = _sites(pixel_format)
    if sites is None:
        return frame
    (r1, c1), (r2, c2) = sites
    height, width = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
    frame = frame[:height, :width]
    return frame[r1::2, c1::2].astype(np.uint16) + frame[r2::2, c2::2].astype(np.uint16)
