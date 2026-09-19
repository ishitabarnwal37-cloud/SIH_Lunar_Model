from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "outputs/registration"

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

    ds = gdal.Open(str(src))
    a = ds.GetRasterBand(1).ReadAsArray()

    result = smooth(a, radius=2)

    result = np.clip(result, 0, 255).astype(np.uint8)

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        str(dst),
        result.shape[1],
        result.shape[0],
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"]
    )

    out.GetRasterBand(1).WriteArray(result)
    out.FlushCache()

    out = None
    ds = None

    print(name, result.shape, "->", dst)