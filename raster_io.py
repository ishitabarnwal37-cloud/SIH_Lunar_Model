"""
Raster I/O helpers for the preprocessing scripts.

Uses rasterio (its wheels bundle GDAL), so no osgeo bindings and no
gdal_translate / gdalinfo executables are needed.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# PDS4 products and the plain-array outputs below carry no geotransform.
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def open_raster(path: Path):
    """Open a raster (GeoTIFF or PDS4 .xml label); raises with a clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected raster does not exist: {path}")
    return rasterio.open(path)


def read_row(ds, x: int, y: int, width: int, band: int = 1) -> np.ndarray:
    """Read `width` pixels of line `y` starting at column `x`."""
    return ds.read(band, window=Window(x, y, width, 1))[0]


def read_array(path: Path, band: int = 1) -> np.ndarray:
    with open_raster(path) as ds:
        return ds.read(band)


def write_raster(path: Path, array: np.ndarray, compress: str = "LZW") -> Path:
    """
    Write a 2-D array as a single-band GeoTIFF. The file is written under a
    temporary name and renamed on success, so a failed run never leaves a
    partial file that looks like a finished output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(
        tmp, "w", driver="GTiff", width=array.shape[1], height=array.shape[0],
        count=1, dtype=array.dtype, compress=compress,
    ) as dst:
        dst.write(array, 1)
    os.replace(tmp, path)
    return path


def copy_band(src_path: Path, dst_path: Path, band: int = 1,
              window: Window | None = None, block_rows: int = 1024) -> tuple[int, int]:
    """
    Copy one band (optionally a window) to a tiled, LZW-compressed BigTIFF,
    streaming `block_rows` lines at a time. Replaces `gdal_translate -b N
    [-srcwin ...]`. Returns (height, width) written.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.name + ".part")

    with open_raster(src_path) as src:
        window = window or Window(0, 0, src.width, src.height)
        x, y, width, height = (int(v) for v in (
            window.col_off, window.row_off, window.width, window.height))
        if x < 0 or y < 0 or x + width > src.width or y + height > src.height:
            raise ValueError(
                f"Window (x={x}, y={y}, w={width}, h={height}) lies outside "
                f"{src_path.name} ({src.width} x {src.height})"
            )
        profile = dict(
            driver="GTiff", width=width, height=height, count=1,
            dtype=src.dtypes[band - 1], tiled=True, blockxsize=256, blockysize=256,
            compress="LZW", BIGTIFF="YES",
        )
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, height, block_rows):
                rows = min(block_rows, height - row)
                data = src.read(band, window=Window(x, y + row, width, rows))
                dst.write(data, 1, window=Window(0, row, width, rows))

    os.replace(tmp, dst_path)
    return height, width


def write_preview(src_path: Path, dst_path: Path, band: int = 1,
                  window: Window | None = None, out_width: int | None = None) -> Path:
    """8-bit min/max-stretched preview (replaces `gdal_translate -scale -outsize`)."""
    with open_raster(src_path) as src:
        window = window or Window(0, 0, src.width, src.height)
        width = int(window.width)
        height = int(window.height)
        if out_width and out_width < width:
            out_shape = (max(1, round(height * out_width / width)), out_width)
        else:
            out_shape = (height, width)
        data = src.read(band, window=window, out_shape=out_shape).astype(np.float32)

    low, high = float(data.min()), float(data.max())
    if high <= low:
        high = low + 1.0
    scaled = np.round((data - low) / (high - low) * 255.0).astype(np.uint8)
    return write_raster(dst_path, scaled)
