import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import COMMON_GRID_HEIGHT, PRODUCTS, PROCESSED_DIR, REGISTRATION_DIR, parse_label
from raster_io import open_raster, read_row, write_raster

SRC = PROCESSED_DIR / "OHRC_2024.tif"
OUT = REGISTRATION_DIR / "OHRC_overlap_corridor.tif"

# OHRC corner longitudes
UL_lon = 23.472939
UR_lon = 23.593739
LL_lon = 23.454679
LR_lon = 23.575386

# TMC left-edge longitude at OHRC top/bottom
tmc_top = 23.570453
tmc_bottom = 23.548896

# OHRC reduced to approximately TMC along-track scale
out_h = COMMON_GRID_HEIGHT
out_w = 180

# The raster must be the full OHRC 2024 product described by its label.
label = parse_label(PRODUCTS["OHRC_2024"].label)

with open_raster(SRC) as ds:
    w, h = ds.width, ds.height
    if (w, h) != (label.samples, label.lines):
        sys.exit(
            f"ERROR: {SRC.name} is {w} x {h} but the OHRC 2024 label says "
            f"{label.samples} x {label.lines}."
        )

    result = np.zeros((out_h, out_w), dtype=np.uint8)

    for j in range(out_h):
        f = j / (out_h - 1)

        left_lon = UL_lon + f * (LL_lon - UL_lon)
        right_lon = UR_lon + f * (LR_lon - UR_lon)
        tmc_lon = tmc_top + f * (tmc_bottom - tmc_top)

        x0 = int(round((tmc_lon - left_lon) / (right_lon - left_lon) * (w - 1)))
        x0 = max(0, min(w - 1, x0))

        src_y = int(round(f * (h - 1)))

        # ~18.5 OHRC pixels per TMC-scale pixel
        src_width = w - x0
        row = read_row(ds, x0, src_y, src_width)

        if len(row) == 0:
            continue

        # average into out_w bins
        edges = np.linspace(0, len(row), out_w + 1).astype(int)

        for x in range(out_w):
            block = row[edges[x]:edges[x + 1]]
            if len(block):
                result[j, x] = int(np.mean(block))

if not result.any():
    sys.exit("ERROR: OHRC corridor is empty; the OHRC raster contains no data in the corridor.")

write_raster(OUT, result)

print("Shape:", result.shape)
print("Saved:", OUT)
