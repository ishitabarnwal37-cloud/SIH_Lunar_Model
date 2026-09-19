import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import REGISTRATION_DIR
from raster_io import read_array, write_raster

DIR = REGISTRATION_DIR

files = {
    "OHRC": DIR / "OHRC_common_grid.tif",
    "TMC": DIR / "TMC_common_grid.tif"
}

def gradient(src):
    a = read_array(src).astype(np.float32)

    mask = a > 0
    if not mask.any():
        raise SystemExit(f"ERROR: {src.name} has no valid (non-zero) pixels.")

    lo, hi = np.percentile(a[mask], (2, 98))
    a = np.clip((a - lo) / (hi - lo), 0, 1)

    gy, gx = np.gradient(a)
    g = np.sqrt(gx * gx + gy * gy)

    g[~mask] = 0

    if np.any(g > 0):
        p = np.percentile(g[g > 0], 99)
        g = np.clip(g / p, 0, 1)

    return (g * 255).astype(np.uint8)

for name, src in files.items():
    img = gradient(src)

    out_path = DIR / f"{name}_gradient.tif"

    write_raster(out_path, img)

    print(name, img.shape, "->", out_path)