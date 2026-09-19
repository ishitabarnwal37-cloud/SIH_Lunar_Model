from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/OHRC_2024.tif"
OUT = ROOT / "outputs/registration/OHRC_common_grid.tif"

ds = gdal.Open(str(SRC))
band = ds.GetRasterBand(1)

src_h = ds.RasterYSize
src_w = ds.RasterXSize

out_h = 4340
out_w = 141

UL = 23.472939
UR = 23.593739
LL = 23.454679
LR = 23.575386

TMC_TOP = 23.570453
TMC_BOTTOM = 23.548896

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

    row = band.ReadAsArray(x0, src_y, src_pixels, 1)[0]

    edges = np.linspace(0, len(row), valid_w + 1).astype(int)

    for x in range(valid_w):
        block = row[edges[x]:edges[x + 1]]
        if len(block):
            result[y, x] = round(np.mean(block))

driver = gdal.GetDriverByName("GTiff")

out = driver.Create(
    str(OUT),
    out_w,
    out_h,
    1,
    gdal.GDT_Byte,
    options=["COMPRESS=LZW"]
)

out.GetRasterBand(1).WriteArray(result)
out.FlushCache()

out = None
ds = None

print("Shape:", result.shape)
print("Saved:", OUT)