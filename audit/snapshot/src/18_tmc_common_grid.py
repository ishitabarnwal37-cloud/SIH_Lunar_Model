from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/registration/TMC_overlap_corridor.tif"
OUT = ROOT / "outputs/registration/TMC_common_grid.tif"

ds = gdal.Open(str(SRC))
band = ds.GetRasterBand(1)

src = band.ReadAsArray()

out_h = 4340
out_w = 141

result = np.zeros((out_h, out_w), dtype=np.uint16)

for y in range(out_h):
    f = y / (out_h - 1)

    valid_w = round(124 + f * (141 - 124))
    valid_w = min(valid_w, out_w, src.shape[1])

    result[y, :valid_w] = src[y, :valid_w]

driver = gdal.GetDriverByName("GTiff")

out = driver.Create(
    str(OUT),
    out_w,
    out_h,
    1,
    gdal.GDT_UInt16,
    options=["COMPRESS=LZW"]
)

out.GetRasterBand(1).WriteArray(result)
out.FlushCache()

out = None
ds = None

print("Shape:", result.shape)
print("Top valid width: 124")
print("Bottom valid width: 141")
print("Saved:", OUT)
