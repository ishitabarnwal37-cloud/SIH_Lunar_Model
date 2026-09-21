from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/OHRC_2024.tif"
OUT = ROOT / "outputs/registration/OHRC_overlap_corridor.tif"

ds = gdal.Open(str(SRC))
band = ds.GetRasterBand(1)

h = ds.RasterYSize
w = ds.RasterXSize

# OHRC corner longitudes
UL_lon = 23.472939
UR_lon = 23.593739
LL_lon = 23.454679
LR_lon = 23.575386

# TMC left-edge longitude at OHRC top/bottom
tmc_top = 23.570453
tmc_bottom = 23.548896

# OHRC reduced to approximately TMC along-track scale
out_h = 4340
out_w = 180

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
    row = band.ReadAsArray(x0, src_y, src_width, 1)[0]

    if len(row) == 0:
        continue

    # average into out_w bins
    edges = np.linspace(0, len(row), out_w + 1).astype(int)

    for x in range(out_w):
        block = row[edges[x]:edges[x + 1]]
        if len(block):
            result[j, x] = int(np.mean(block))

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
