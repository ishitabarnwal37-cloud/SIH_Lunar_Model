import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import REGISTRATION_DIR
from raster_io import read_array, write_raster

DIR = REGISTRATION_DIR

def smooth(a, radius=2):
    p = np.pad(a.astype(np.float32), radius, mode="reflect")
    out = np.zeros_like(a, dtype=np.float32)

    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out += p[dy:dy+a.shape[0], dx:dx+a.shape[1]]

    return out / (2 * radius + 1) ** 2

for name in ["OHRC", "TMC"]:
    src = DIR / f"{name}_gradient.tif"
    dst = DIR / f"{name}_gradient_smooth.tif"

    a = read_array(src)

    result = smooth(a, radius=2)

    result = np.clip(result, 0, 255).astype(np.uint8)

    write_raster(dst, result)

    print(name, result.shape, "->", dst)