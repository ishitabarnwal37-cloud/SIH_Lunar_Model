import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import COMMON_GRID_HEIGHT, PROCESSED_DIR, REGISTRATION_DIR
from raster_io import open_raster, read_row, write_raster

SRC = PROCESSED_DIR / "TMC.tif"
OUT = REGISTRATION_DIR / "TMC_overlap_corridor.tif"

# Corridor window in the TMC raster (derived from the OAT / footprint
# geometry, see scripts 07 and 10). These are tied to the TMC product and
# are validated against the real raster size below.
y0 = 43290
h = COMMON_GRID_HEIGHT
search_x0 = 4000
search_w = 5600
out_w = 180

MIN_ROW_COVERAGE = 0.90   # fraction of corridor rows that must contain data

with open_raster(SRC) as ds:
    if ds.count < 1:
        sys.exit(f"ERROR: {SRC} has no bands.")
    print(f"TMC raster: {ds.width} x {ds.height}")

    if y0 + h > ds.height or search_x0 + search_w > ds.width:
        sys.exit(
            f"ERROR: corridor window (x {search_x0}..{search_x0 + search_w}, "
            f"y {y0}..{y0 + h}) lies outside the TMC raster "
            f"({ds.width} x {ds.height}). This script's constants belong to a "
            "different product or the raster is wrong."
        )

    result = np.zeros((h, out_w), dtype=np.uint16)

    for i in range(h):
        row = read_row(ds, search_x0, y0 + i, search_w)

        nz = np.flatnonzero(row > 0)
        if len(nz) == 0:
            continue

        left = nz[0]
        end = min(left + out_w, search_w)
        result[i, :end - left] = row[left:end]

covered = float(np.mean(result.any(axis=1)))
print(f"Corridor rows containing data: {covered:.1%}")
if covered < MIN_ROW_COVERAGE:
    sys.exit(
        f"ERROR: only {covered:.1%} of the corridor rows contain TMC data "
        f"(need >= {MIN_ROW_COVERAGE:.0%}). The window y={y0}..{y0 + h}, "
        f"x>={search_x0} does not hit the TMC swath; check the source raster "
        "(truncated file?) and the overlap geometry."
    )

write_raster(OUT, result)

print("Shape:", result.shape)
print("Saved:", OUT)
