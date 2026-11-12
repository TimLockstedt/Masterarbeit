"""Tiles for scanning with the x/y stage. Stage x/y is the camera field-of-view centre.

Tiles are visited in serpentine order: even rows by increasing x index, odd
rows by decreasing x index. ``tile.name`` (``x00_y00``) is meant for folders.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    ix: int
    iy: int
    x_mm: float | None  # None: stay where the stage is
    y_mm: float | None

    @property
    def name(self) -> str:
        return f"x{self.ix:02d}_y{self.iy:02d}"


HERE = Tile(0, 0, None, None)


def _pair(value, name):
    value = tuple(float(v) for v in (value if np.ndim(value) else (value, value)))
    if len(value) != 2:
        raise ValueError(f"{name} must be one number or [x, y]")
    return value


def grid(start_mm, count, tile_size_mm, overlap_mm=0.0) -> list[Tile]:
    """``count = [nx, ny]`` tiles; ``start_mm`` is the centre of tile (0, 0)."""
    sx, sy = _pair(start_mm, "start_mm")
    nx, ny = (int(c) for c in _pair(count, "count"))
    w, h = _pair(tile_size_mm, "tile_size_mm")
    ox, oy = _pair(overlap_mm, "overlap_mm")
    if nx < 1 or ny < 1 or w <= ox or h <= oy or min(ox, oy) < 0:
        raise ValueError("need count >= 1 and 0 <= overlap < tile size")
    tiles = []
    for iy in range(ny):
        columns = range(nx) if iy % 2 == 0 else range(nx - 1, -1, -1)
        tiles += [Tile(ix, iy, sx + ix * (w - ox), sy + iy * (h - oy)) for ix in columns]
    return tiles


def inside(points, polygon) -> np.ndarray:
    """Even-odd rule: which (N, 2) points lie inside the polygon (M, 2)."""
    points, polygon = np.asarray(points, float), np.asarray(polygon, float)
    x, y = points[:, 0:1], points[:, 1:2]
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (crosses & (x < x_cross)).sum(axis=1) % 2 == 1


def tiles_in_polygon(polygon_mm, tile_size_mm, overlap_mm=0.0) -> list[Tile]:
    """Tiles of a grid over the polygon's bounding box that touch the polygon."""
    polygon = np.asarray(polygon_mm, float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon needs at least 3 [x, y] points")
    w, h = _pair(tile_size_mm, "tile_size_mm")
    ox, oy = _pair(overlap_mm, "overlap_mm")
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    nx = max(1, int(np.ceil((high[0] - low[0] - ox) / (w - ox))))
    ny = max(1, int(np.ceil((high[1] - low[1] - oy) / (h - oy))))
    start = (low[0] + w / 2, low[1] + h / 2)
    kept = []
    for tile in grid(start, (nx, ny), (w, h), (ox, oy)):
        # tile rectangle sampled on a 5x5 raster, plus polygon vertices inside the tile
        u = np.linspace(-0.5, 0.5, 5)
        samples = np.array([(tile.x_mm + a * w, tile.y_mm + b * h) for a in u for b in u])
        vertex_in_tile = (
            (abs(polygon[:, 0] - tile.x_mm) <= w / 2) & (abs(polygon[:, 1] - tile.y_mm) <= h / 2)
        ).any()
        if vertex_in_tile or inside(samples, polygon).any():
            kept.append(tile)
    return kept


def tiles_from(parameters: dict, setup=None, run=None) -> list[Tile]:
    """Tiles from a ``[measurement.grid]`` table, or the current position if absent.

    ``grid = {start_mm, count, tile_size_mm, overlap_mm}``,
    ``grid = {polygon_mm, tile_size_mm, overlap_mm}`` or
    ``grid = {roi = "overview", tile_size_mm, overlap_mm, camera}`` (draw on
    the image of ``camera``, default "overview_camera", e.g. "webcam"; needs
    ``setup`` and the overview_to_stage calibration of that camera; the
    polygon is recorded in ``run``'s metadata).
    """
    table = parameters.get("grid")
    if not table:
        return [HERE]
    table = dict(table)
    if table.get("roi") == "overview":
        from .ui.roi import roi_tiles

        if setup is None:
            raise ValueError("roi = 'overview' needs the setup: tiles_from(p, setup, run)")
        tiles, polygon_mm = roi_tiles(
            setup,
            table["tile_size_mm"],
            table.get("overlap_mm", 0.0),
            camera=table.get("camera", "overview_camera"),
        )
        if run is not None:
            run.record("roi_polygon_mm", polygon_mm.tolist())
        return tiles
    if "polygon_mm" in table:
        return tiles_in_polygon(
            table.pop("polygon_mm"), table.pop("tile_size_mm"), table.pop("overlap_mm", 0.0)
        )
    return grid(**table)


def move_to(setup, tile: Tile):
    """Start moving stage_x/stage_y (if they are in the setup) to the tile."""
    for name, value in (("stage_x", tile.x_mm), ("stage_y", tile.y_mm)):
        if value is None:
            continue
        if name not in setup:
            raise ValueError(f"tile {tile.name} needs {name}, which is not in the setup")
        setup[name].set(position_mm=value)
