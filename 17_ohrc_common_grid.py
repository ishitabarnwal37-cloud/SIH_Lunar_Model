import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import (
    COMMON_GRID_HEIGHT, COMMON_GRID_WIDTH, PRODUCTS, PROCESSED_DIR,
    REGISTRATION_DIR, parse_label,
)
from raster_io import open_raster, read_row, write_raster

SRC = PROCESSED_DIR / "OHRC_2024.tif"
OUT = REGISTRATION_DIR / "OHRC_common_grid.tif"

out_h = COMMON_GRID_HEIGHT
out_w = COMMON_GRID_WIDTH

UL = 23.472939
UR = 23.593739
LL = 23.454679
LR = 23.575386

TMC_TOP = 23.570453
TMC_BOTTOM = 23.548896

# The raster must be the full OHRC 2024 product described by its label.
label = parse_label(PRODUCTS["OHRC_2024"].label)

with open_raster(SRC) as ds:
    src_h = ds.height
    src_w = ds.width
    if (src_w, src_h) != (label.samples, label.lines):
        sys.exit(
            f"ERROR: {SRC.name} is {src_w} x {src_h} but the OHRC 2024 label "
            f"says {label.samples} x {label.lines}."
        )

    result = np.zeros((out_h, out_w), dtype=np.uint8)

    for y in range(out_h):
        f = y / (out_h - 1)

        left_lon = UL + f * (LL - UL)
        right_lon = UR + f * (LR - UR)
        tmc_lon = TMC_TOP + f * (TMC_BOTTOM - TMC_TOP)

        overlap_fraction = (right_lon - tmc_lon) / (right_lon - left_lon)
        valid_w = round(out_w * overlap_fraction / 0.03120)
        valid_w = max(1, min(out_w, valid_w))

        x0 = (tmc_lon - left_lon) / (right_lon - left_lon) * (src_w - 1)
        src_y = f * (src_h - 1)

        x0 = int(round(x0))
        src_y = int(round(src_y))

        src_pixels = src_w - x0

        if src_pixels <= 0:
            continue

        row = read_row(ds, x0, src_y, src_pixels)

        edges = np.linspace(0, len(row), valid_w + 1).astype(int)

        for x in range(valid_w):
            block = row[edges[x]:edges[x + 1]]
            if len(block):
                result[y, x] = round(np.mean(block))

if not result.any():
    sys.exit("ERROR: OHRC common grid is empty; no OHRC data was sampled.")

write_raster(OUT, result)

print("Shape:", result.shape)
print("Saved:", OUT)
