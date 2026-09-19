import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import COMMON_GRID_HEIGHT, COMMON_GRID_WIDTH, REGISTRATION_DIR
from raster_io import read_array, write_raster

SRC = REGISTRATION_DIR / "TMC_overlap_corridor.tif"
OUT = REGISTRATION_DIR / "TMC_common_grid.tif"

out_h = COMMON_GRID_HEIGHT
out_w = COMMON_GRID_WIDTH

# Valid TMC width at the top / bottom of the common grid (columns).
TOP_VALID_WIDTH = 124
BOTTOM_VALID_WIDTH = 141

src = read_array(SRC)

if src.shape[0] < out_h or src.shape[1] < out_w:
    sys.exit(
        f"ERROR: TMC corridor is {src.shape[1]} x {src.shape[0]} (w x h) but the "
        f"common grid needs at least {out_w} x {out_h}. Re-run script 11."
    )

result = np.zeros((out_h, out_w), dtype=np.uint16)

for y in range(out_h):
    f = y / (out_h - 1)

    valid_w = round(TOP_VALID_WIDTH + f * (BOTTOM_VALID_WIDTH - TOP_VALID_WIDTH))
    valid_w = min(valid_w, out_w, src.shape[1])

    result[y, :valid_w] = src[y, :valid_w]

if not result.any():
    sys.exit("ERROR: TMC common grid is empty; the corridor contained no data.")

write_raster(OUT, result)

print("Shape:", result.shape)
print(f"Top valid width: {TOP_VALID_WIDTH}")
print(f"Bottom valid width: {BOTTOM_VALID_WIDTH}")
print("Saved:", OUT)
