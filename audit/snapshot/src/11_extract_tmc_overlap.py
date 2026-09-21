from pathlib import Path
from osgeo import gdal
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/TMC.tif"
OUT = ROOT / "outputs/registration/TMC_overlap_corridor.tif"

ds = gdal.Open(str(SRC))
band = ds.GetRasterBand(1)

y0 = 43290
h = 4340
search_x0 = 4000
search_w = 5600
out_w = 180

result = np.zeros((h, out_w), dtype=np.uint16)

for i in range(h):
    row = band.ReadAsArray(search_x0, y0 + i, search_w, 1)[0]

    nz = np.flatnonzero(row > 0)
    if len(nz) == 0:
        continue

    left = nz[0]
    end = min(left + out_w, search_w)
    result[i, :end-left] = row[left:end]

driver = gdal.GetDriverByName("GTiff")
out = driver.Create(
    str(OUT), out_w, h, 1, gdal.GDT_UInt16,
    options=["COMPRESS=LZW"]
)

out.GetRasterBand(1).WriteArray(result)
out.FlushCache()

out = None
ds = None

print("Shape:", result.shape)
print("Saved:", OUT)
